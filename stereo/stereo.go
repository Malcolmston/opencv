package stereo

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// InvalidDisparity is the reserved disparity value stored for pixels with no
// reliable match (uniform regions, the unsearchable left border band, or
// ambiguous matches). It is 0, so callers should treat 0 in a disparity map as
// "no data" rather than as a true zero disparity.
const InvalidDisparity uint8 = 0

// StereoBM computes a disparity map by local block matching, mirroring OpenCV's
// cv::StereoBM. For each left-image pixel it slides a BlockSize×BlockSize window
// across the horizontal search range [0, NumDisparities) in the right image and
// keeps the disparity with the smallest sum of absolute differences (SAD),
// subject to a texture threshold and a uniqueness ratio.
//
// The zero value is not useful; set at least NumDisparities and BlockSize. The
// remaining fields default when left non-positive.
type StereoBM struct {
	// NumDisparities is the width of the disparity search range in pixels; the
	// matcher considers disparities d in [0, NumDisparities). Must be positive.
	// Defaults to 16 when non-positive.
	NumDisparities int
	// BlockSize is the odd side length of the square matching window. Larger
	// windows are smoother but blur depth edges. Defaults to 9 when non-positive.
	BlockSize int
	// UniquenessRatio is the margin, in percent, by which the best match must beat
	// the second-best (non-adjacent) match; otherwise the pixel is marked invalid.
	// Defaults to 10 when non-positive.
	UniquenessRatio int
	// TextureThreshold is the minimum intensity range (max-min) inside the left
	// window; flatter windows are marked invalid. Defaults to 4 when non-positive.
	TextureThreshold int
}

// Compute matches left against right and returns a single-channel 8-bit
// disparity map the same size as the inputs. Inputs may be single-channel
// (grayscale) or three-channel (RGB, converted to gray); they must share the
// same dimensions. Pixels with no reliable match hold [InvalidDisparity].
//
// It panics if either image is empty, the images differ in size, an image has an
// unsupported channel count, or BlockSize is not a positive odd integer.
func (bm StereoBM) Compute(left, right *cv.Mat) *cv.Mat {
	numDisp := bm.NumDisparities
	if numDisp <= 0 {
		numDisp = 16
	}
	block := bm.BlockSize
	if block <= 0 {
		block = 9
	}
	requireOdd(block, "StereoBM.BlockSize")
	uniq := bm.UniquenessRatio
	if uniq <= 0 {
		uniq = 10
	}
	tex := bm.TextureThreshold
	if tex <= 0 {
		tex = 4
	}

	rows, cols, gl := toGrayGrid(left)
	rrows, rcols, gr := toGrayGrid(right)
	if rows != rrows || cols != rcols {
		panic(fmt.Sprintf("stereo: StereoBM.Compute size mismatch left %dx%d right %dx%d", rows, cols, rrows, rcols))
	}

	half := block / 2
	out := cv.NewMat(rows, cols, 1)
	sads := make([]int, numDisp)

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			// Left border band: the full search range is unavailable.
			if x < numDisp-1 {
				continue
			}
			// Texture test: reject flat windows.
			if windowRange(gl, rows, cols, y, x, half) < tex {
				continue
			}
			maxD := numDisp - 1
			bestD, bestSad := 0, 1<<62
			for d := 0; d <= maxD; d++ {
				s := blockSAD(gl, gr, rows, cols, y, x, d, half)
				sads[d] = s
				if s < bestSad {
					bestSad, bestD = s, d
				}
			}
			// Uniqueness: the second-best non-adjacent match must be worse by the
			// configured margin.
			secondSad := 1 << 62
			for d := 0; d <= maxD; d++ {
				if d < bestD-1 || d > bestD+1 {
					if sads[d] < secondSad {
						secondSad = sads[d]
					}
				}
			}
			if secondSad != 1<<62 && secondSad*100 <= bestSad*(100+uniq) {
				continue // ambiguous
			}
			out.Data[y*cols+x] = uint8(bestD)
		}
	}
	return out
}

// toGrayGrid extracts a single-channel intensity grid from m as a flat []int in
// row-major order. Three-channel input is converted with the root package's
// RGB->Gray. It panics on empty or unsupported input.
func toGrayGrid(m *cv.Mat) (rows, cols int, g []int) {
	if m == nil || m.Empty() {
		panic("stereo: nil or empty input Mat")
	}
	src := m
	switch m.Channels {
	case 1:
		// use as-is
	case 3:
		src = cv.CvtColor(m, cv.ColorRGB2Gray)
	default:
		panic(fmt.Sprintf("stereo: input must be 1- or 3-channel, got %d", m.Channels))
	}
	rows, cols = src.Rows, src.Cols
	g = make([]int, rows*cols)
	for i := 0; i < rows*cols; i++ {
		g[i] = int(src.Data[i])
	}
	return rows, cols, g
}

// clampInt clamps v to [lo, hi].
func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// blockSAD returns the sum of absolute differences between the left window
// centred at (y, x) and the right window centred at (y, x-d). Coordinates
// outside the image are clamped to the edge (border replication).
func blockSAD(gl, gr []int, rows, cols, y, x, d, half int) int {
	s := 0
	for dy := -half; dy <= half; dy++ {
		yy := clampInt(y+dy, 0, rows-1)
		rowBase := yy * cols
		for dx := -half; dx <= half; dx++ {
			lx := clampInt(x+dx, 0, cols-1)
			rx := clampInt(x-d+dx, 0, cols-1)
			diff := gl[rowBase+lx] - gr[rowBase+rx]
			if diff < 0 {
				diff = -diff
			}
			s += diff
		}
	}
	return s
}

// windowRange returns the intensity range (max-min) of the left window centred
// at (y, x), a cheap texture measure. Borders are replicated.
func windowRange(g []int, rows, cols, y, x, half int) int {
	mn, mx := 1<<62, -(1 << 62)
	for dy := -half; dy <= half; dy++ {
		yy := clampInt(y+dy, 0, rows-1)
		rowBase := yy * cols
		for dx := -half; dx <= half; dx++ {
			xx := clampInt(x+dx, 0, cols-1)
			v := g[rowBase+xx]
			if v < mn {
				mn = v
			}
			if v > mx {
				mx = v
			}
		}
	}
	return mx - mn
}

// requireOdd panics unless ksize is a positive odd integer.
func requireOdd(ksize int, name string) {
	if ksize <= 0 || ksize%2 == 0 {
		panic(fmt.Sprintf("stereo: %s requires a positive odd size, got %d", name, ksize))
	}
}
