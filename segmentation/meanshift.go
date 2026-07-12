package segmentation

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// meanShiftMaxIter caps the number of mean-shift iterations per pixel, matching
// the default termination of cv2.pyrMeanShiftFiltering.
const meanShiftMaxIter = 5

// meanShiftEps is the convergence threshold on the combined spatial-plus-range
// shift magnitude; iteration for a pixel stops once a step moves less than this.
const meanShiftEps = 1.0

// MeanShiftFiltering performs edge-preserving mean-shift smoothing of the
// three-channel image img and returns a new Mat of the same size, mirroring a
// single pyramid level of cv2.pyrMeanShiftFiltering.
//
// Each pixel is treated as a point in the joint 5-D spatial-range space
// (x, y, R, G, B). Starting from the pixel's own position and colour, the mean
// of all pixels within the spatial window radius sp whose colour lies within the
// range radius sr is computed repeatedly; the point is moved to that mean until
// it converges (or [meanShiftMaxIter] is reached). The pixel in the output takes
// the converged colour, which collapses noisy regions onto a shared mode while
// preserving the strong colour boundaries that the range window refuses to cross.
//
// sp and sr must be positive. It panics if img is empty or not three-channel.
func MeanShiftFiltering(img *cv.Mat, sp, sr float64) *cv.Mat {
	if img.Empty() {
		panic("segmentation: MeanShiftFiltering on empty image")
	}
	if img.Channels != 3 {
		panic(fmt.Sprintf("segmentation: MeanShiftFiltering requires a 3-channel image, got %d channels", img.Channels))
	}
	if sp <= 0 || sr <= 0 {
		panic("segmentation: MeanShiftFiltering requires positive sp and sr")
	}

	rows, cols := img.Rows, img.Cols
	radius := int(sp)
	if radius < 1 {
		radius = 1
	}
	sr2 := sr * sr
	out := cv.NewMat(rows, cols, 3)

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			cx, cy := float64(x), float64(y)
			b := (y*cols + x) * 3
			cr := float64(img.Data[b+0])
			cg := float64(img.Data[b+1])
			cb := float64(img.Data[b+2])

			for iter := 0; iter < meanShiftMaxIter; iter++ {
				var sX, sY, sR, sG, sB float64
				count := 0
				iCx, iCy := int(cx+0.5), int(cy+0.5)
				for wy := iCy - radius; wy <= iCy+radius; wy++ {
					if wy < 0 || wy >= rows {
						continue
					}
					for wx := iCx - radius; wx <= iCx+radius; wx++ {
						if wx < 0 || wx >= cols {
							continue
						}
						wb := (wy*cols + wx) * 3
						pr := float64(img.Data[wb+0])
						pg := float64(img.Data[wb+1])
						pb := float64(img.Data[wb+2])
						dr, dg, db := pr-cr, pg-cg, pb-cb
						if dr*dr+dg*dg+db*db > sr2 {
							continue
						}
						sX += float64(wx)
						sY += float64(wy)
						sR += pr
						sG += pg
						sB += pb
						count++
					}
				}
				if count == 0 {
					break
				}
				nCx := sX / float64(count)
				nCy := sY / float64(count)
				nR := sR / float64(count)
				nG := sG / float64(count)
				nB := sB / float64(count)

				shift := math.Abs(nCx-cx) + math.Abs(nCy-cy) +
					math.Abs(nR-cr) + math.Abs(nG-cg) + math.Abs(nB-cb)
				cx, cy, cr, cg, cb = nCx, nCy, nR, nG, nB
				if shift <= meanShiftEps {
					break
				}
			}

			out.Data[b+0] = clampU8(cr)
			out.Data[b+1] = clampU8(cg)
			out.Data[b+2] = clampU8(cb)
		}
	}
	return out
}

// PyrMeanShiftFiltering applies mean-shift smoothing over a Gaussian pyramid,
// the multi-scale variant of [MeanShiftFiltering] modelled on
// cv2.pyrMeanShiftFiltering's maxLevel argument.
//
// The image is downsampled maxLevel times with [cv.PyrDown]; mean-shift
// filtering is applied at the coarsest level, the result is upsampled with
// [cv.PyrUp] to seed the next finer level, and filtering is re-applied there.
// Coarse levels let modes form from large neighbourhoods cheaply, while finer
// levels sharpen the boundaries. When maxLevel is zero this reduces to a single
// [MeanShiftFiltering] pass.
//
// sp and sr have the same meaning as in [MeanShiftFiltering]. maxLevel is
// clamped to a non-negative value that keeps every pyramid level at least one
// pixel across. It panics if img is empty or not three-channel.
func PyrMeanShiftFiltering(img *cv.Mat, sp, sr float64, maxLevel int) *cv.Mat {
	if img.Empty() {
		panic("segmentation: PyrMeanShiftFiltering on empty image")
	}
	if img.Channels != 3 {
		panic(fmt.Sprintf("segmentation: PyrMeanShiftFiltering requires a 3-channel image, got %d channels", img.Channels))
	}
	if maxLevel < 0 {
		maxLevel = 0
	}

	// Build the pyramid, stopping early if a level would collapse below 2px.
	pyr := []*cv.Mat{img}
	for l := 0; l < maxLevel; l++ {
		top := pyr[len(pyr)-1]
		if top.Rows < 2 || top.Cols < 2 {
			break
		}
		pyr = append(pyr, cv.PyrDown(top))
	}

	// Filter the coarsest level from scratch.
	filtered := MeanShiftFiltering(pyr[len(pyr)-1], sp, sr)

	// Walk back up: upsample the coarse result to seed each finer level, then
	// refine that level with another mean-shift pass.
	for l := len(pyr) - 2; l >= 0; l-- {
		target := pyr[l]
		up := cv.PyrUp(filtered)
		seed := resizeTo(up, target.Rows, target.Cols)
		// Blend the upsampled coarse estimate with the true detail at this level
		// by averaging, then run mean shift to lock onto the local mode.
		blended := cv.NewMat(target.Rows, target.Cols, 3)
		for i := range blended.Data {
			blended.Data[i] = uint8((int(seed.Data[i]) + int(target.Data[i])) / 2)
		}
		filtered = MeanShiftFiltering(blended, sp, sr)
	}
	return filtered
}

// resizeTo crops or edge-replicates src to exactly rows x cols. PyrUp doubles a
// dimension, which may overshoot the odd original size by one pixel; this trims
// or pads to match without importing a full resize routine.
func resizeTo(src *cv.Mat, rows, cols int) *cv.Mat {
	if src.Rows == rows && src.Cols == cols {
		return src
	}
	dst := cv.NewMat(rows, cols, src.Channels)
	ch := src.Channels
	for y := 0; y < rows; y++ {
		sy := y
		if sy >= src.Rows {
			sy = src.Rows - 1
		}
		for x := 0; x < cols; x++ {
			sx := x
			if sx >= src.Cols {
				sx = src.Cols - 1
			}
			si := (sy*src.Cols + sx) * ch
			di := (y*cols + x) * ch
			copy(dst.Data[di:di+ch], src.Data[si:si+ch])
		}
	}
	return dst
}
