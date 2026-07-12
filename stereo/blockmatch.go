package stereo

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// BlockMatcher is a full-featured local block matcher, the counterpart to
// OpenCV's cv::StereoBM. It extends the bare [StereoBM] with the complete
// post-processing pipeline that OpenCV runs: an optional intensity pre-filter
// ([PrefilterType]) to normalise the inputs, a non-zero MinDisparity, the
// texture and uniqueness rejection tests, speckle removal, and a left-right
// consistency check (Disp12MaxDiff). The result is a considerably cleaner map
// than the raw matcher on real imagery.
//
// The zero value is not useful; set at least NumDisparities and BlockSize.
// Remaining fields default when non-positive.
type BlockMatcher struct {
	// MinDisparity is the smallest disparity searched (usually 0).
	MinDisparity int
	// NumDisparities is the width of the search range in pixels. Defaults to 16.
	NumDisparities int
	// BlockSize is the odd side length of the SAD window. Defaults to 9.
	BlockSize int
	// UniquenessRatio is the percent margin by which the best match must beat the
	// second-best non-adjacent match. Defaults to 10.
	UniquenessRatio int
	// TextureThreshold is the minimum left-window intensity range. Defaults to 4.
	TextureThreshold int
	// PreFilterType selects the input pre-filter. The zero value is [PrefilterNone].
	PreFilterType PrefilterType
	// PreFilterSize is the odd box size for [PrefilterNormalizedResponse]. Defaults to 9.
	PreFilterSize int
	// PreFilterCap clamps the pre-filter response. Defaults to 31.
	PreFilterCap int
	// SpeckleWindowSize is the maximum size of a disparity blob removed by speckle
	// filtering; non-positive disables speckle removal.
	SpeckleWindowSize int
	// SpeckleRange is the maximum disparity variation within a speckle blob.
	SpeckleRange int
	// Disp12MaxDiff is the maximum left-right disparity disagreement; negative
	// disables the consistency check.
	Disp12MaxDiff int
}

// Compute matches left against right and returns a single-channel 8-bit
// disparity map. The pipeline is: pre-filter both images, block-match with SAD
// under the texture and uniqueness tests, run the left-right consistency check,
// then remove speckles. Unmatched pixels hold [InvalidDisparity].
//
// It panics on empty input, a size or channel mismatch, or an even BlockSize.
func (b BlockMatcher) Compute(left, right *cv.Mat) *cv.Mat {
	minD := b.MinDisparity
	numD := b.NumDisparities
	if numD <= 0 {
		numD = 16
	}
	block := b.BlockSize
	if block <= 0 {
		block = 9
	}
	requireOdd(block, "BlockMatcher.BlockSize")
	uniq := b.UniquenessRatio
	if uniq <= 0 {
		uniq = 10
	}
	tex := b.TextureThreshold
	if tex <= 0 {
		tex = 4
	}

	gl := applyPrefilter(left, b.PreFilterType, b.PreFilterSize, b.PreFilterCap)
	gr := applyPrefilter(right, b.PreFilterType, b.PreFilterSize, b.PreFilterCap)
	if gl.Rows != gr.Rows || gl.Cols != gr.Cols {
		panic(fmt.Sprintf("stereo: BlockMatcher.Compute size mismatch left %dx%d right %dx%d",
			gl.Rows, gl.Cols, gr.Rows, gr.Cols))
	}
	rows, cols, li := matToIntGrid(gl)
	_, _, ri := matToIntGrid(gr)

	out := matchBlock(li, ri, rows, cols, minD, numD, block, uniq, tex, false)

	if b.Disp12MaxDiff >= 0 {
		right := matchBlock(ri, li, rows, cols, minD, numD, block, uniq, tex, true)
		out = ValidateDisparity(out, right, b.Disp12MaxDiff, InvalidDisparity)
	}
	if b.SpeckleWindowSize > 0 {
		FilterSpecklesDisparity(out, InvalidDisparity, b.SpeckleWindowSize, b.SpeckleRange)
	}
	return out
}

// matchBlock performs winner-take-all SAD block matching with texture and
// uniqueness rejection. When rightView is false the reference is the left image
// and the match for column x is searched at x-d; when true the reference is the
// right image and the match is searched at x+d, producing a right-referenced
// disparity map suitable for the consistency check.
func matchBlock(base, other []int, rows, cols, minD, numD, block, uniq, tex int, rightView bool) *cv.Mat {
	half := block / 2
	out := cv.NewMat(rows, cols, 1)
	sads := make([]int, numD)
	borderLimit := minD + numD - 1
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if rightView {
				if x >= cols-borderLimit {
					continue
				}
			} else if x < borderLimit {
				continue
			}
			if windowRange(base, rows, cols, y, x, half) < tex {
				continue
			}
			bestIdx, bestSad := 0, 1<<62
			for idx := 0; idx < numD; idx++ {
				d := minD + idx
				var ocx int
				if rightView {
					ocx = x + d
				} else {
					ocx = x - d
				}
				s := sadAt(base, other, rows, cols, y, x, ocx, half)
				sads[idx] = s
				if s < bestSad {
					bestSad, bestIdx = s, idx
				}
			}
			secondSad := 1 << 62
			for idx := 0; idx < numD; idx++ {
				if idx < bestIdx-1 || idx > bestIdx+1 {
					if sads[idx] < secondSad {
						secondSad = sads[idx]
					}
				}
			}
			if secondSad != 1<<62 && secondSad*100 <= bestSad*(100+uniq) {
				continue
			}
			out.Data[y*cols+x] = uint8(clampInt(minD+bestIdx, 0, 255))
		}
	}
	return out
}

// sadAt returns the sum of absolute differences between the base window centred
// at (y, x) and the other-image window whose centre column is ocx (same row).
// Coordinates outside the image are replicated.
func sadAt(base, other []int, rows, cols, y, x, ocx, half int) int {
	s := 0
	for dy := -half; dy <= half; dy++ {
		yy := clampInt(y+dy, 0, rows-1)
		rowBase := yy * cols
		for dx := -half; dx <= half; dx++ {
			bx := clampInt(x+dx, 0, cols-1)
			ox := clampInt(ocx+dx, 0, cols-1)
			diff := base[rowBase+bx] - other[rowBase+ox]
			if diff < 0 {
				diff = -diff
			}
			s += diff
		}
	}
	return s
}
