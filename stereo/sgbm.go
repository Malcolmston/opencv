package stereo

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// NumPaths is the number of aggregation paths used by [StereoSGBM]. This
// implementation is an SGM-lite that aggregates along the four cardinal
// directions (left, right, up, down); the four diagonal paths of the full
// eight-path OpenCV SGBM are deferred.
const NumPaths = 4

// StereoSGBM computes a disparity map with semi-global-matching-lite, a reduced
// form of OpenCV's cv::StereoSGBM. A pixelwise (BlockSize) SAD cost volume is
// aggregated along [NumPaths] cardinal paths using the SGM smoothness penalties
// P1 and P2, and the winner-take-all disparity of the summed cost is returned.
// Aggregating across four paths propagates confident matches into weakly
// textured regions, so it typically recovers more of a scene than [StereoBM].
//
// The zero value is not useful; set at least NumDisparities and BlockSize. The
// remaining fields default when left non-positive.
type StereoSGBM struct {
	// NumDisparities is the width of the disparity search range in pixels; the
	// matcher considers disparities d in [0, NumDisparities). Must be positive.
	// Defaults to 16 when non-positive.
	NumDisparities int
	// BlockSize is the odd side length of the SAD window used for the data cost.
	// A value of 1 gives a strictly pixelwise cost. Defaults to 5 when non-positive.
	BlockSize int
	// P1 is the penalty added when the disparity of neighbouring pixels changes by
	// one. Defaults to 8*BlockSize when non-positive.
	P1 int
	// P2 is the penalty added for larger disparity changes; it should exceed P1.
	// Defaults to 32*BlockSize when non-positive.
	P2 int
	// UniquenessRatio is the margin, in percent, by which the best aggregated cost
	// must beat the second-best (non-adjacent) cost; otherwise the pixel is marked
	// invalid. Defaults to 10 when non-positive.
	UniquenessRatio int
}

// aggregation path directions: left->right, right->left, top->bottom, bottom->top.
var sgbmPaths = [NumPaths][2]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}

// Compute matches left against right and returns a single-channel 8-bit
// disparity map the same size as the inputs. Inputs may be single-channel
// (grayscale) or three-channel (RGB, converted to gray) and must share the same
// dimensions. Pixels with no reliable match hold [InvalidDisparity].
//
// It panics if either image is empty, the images differ in size, an image has an
// unsupported channel count, or BlockSize is not a positive odd integer.
func (sg StereoSGBM) Compute(left, right *cv.Mat) *cv.Mat {
	numDisp := sg.NumDisparities
	if numDisp <= 0 {
		numDisp = 16
	}
	block := sg.BlockSize
	if block <= 0 {
		block = 5
	}
	requireOdd(block, "StereoSGBM.BlockSize")
	p1 := sg.P1
	if p1 <= 0 {
		p1 = 8 * block
	}
	p2 := sg.P2
	if p2 <= 0 {
		p2 = 32 * block
	}
	if p2 < p1 {
		p2 = p1
	}
	uniq := sg.UniquenessRatio
	if uniq <= 0 {
		uniq = 10
	}

	rows, cols, gl := toGrayGrid(left)
	rrows, rcols, gr := toGrayGrid(right)
	if rows != rrows || cols != rcols {
		panic(fmt.Sprintf("stereo: StereoSGBM.Compute size mismatch left %dx%d right %dx%d", rows, cols, rrows, rcols))
	}

	half := block / 2
	n := rows * cols

	// Data cost volume C[pixel*numDisp + d].
	cost := make([]int, n*numDisp)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			base := (y*cols + x) * numDisp
			for d := 0; d < numDisp; d++ {
				cost[base+d] = blockSAD(gl, gr, rows, cols, y, x, d, half)
			}
		}
	}

	// Aggregate along each path, summing into S.
	agg := make([]int, n*numDisp)
	lr := make([]int, n*numDisp) // per-path cost, reused each direction
	minLr := make([]int, n)      // per-pixel path minimum, reused each direction
	for _, dir := range sgbmPaths {
		aggregatePath(cost, lr, minLr, rows, cols, numDisp, dir, p1, p2)
		for i := range agg {
			agg[i] += lr[i]
		}
	}

	// Winner-take-all with border, and uniqueness filtering.
	out := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if x < numDisp-1 {
				continue
			}
			base := (y*cols + x) * numDisp
			bestD, bestC := 0, 1<<62
			for d := 0; d < numDisp; d++ {
				if agg[base+d] < bestC {
					bestC, bestD = agg[base+d], d
				}
			}
			secondC := 1 << 62
			for d := 0; d < numDisp; d++ {
				if d < bestD-1 || d > bestD+1 {
					if agg[base+d] < secondC {
						secondC = agg[base+d]
					}
				}
			}
			if secondC != 1<<62 && secondC*100 <= bestC*(100+uniq) {
				continue // ambiguous
			}
			out.Data[y*cols+x] = uint8(bestD)
		}
	}
	return out
}

// aggregatePath fills lr with the SGM path cost for a single direction and
// records each pixel's minimum path cost in minLr. The recurrence is
//
//	Lr(p,d) = C(p,d) + min( Lr(p-r,d),
//	                        Lr(p-r,d-1)+P1,
//	                        Lr(p-r,d+1)+P1,
//	                        min_k Lr(p-r,k)+P2 ) - min_k Lr(p-r,k)
//
// where p-r is the predecessor along direction r. Subtracting the predecessor
// minimum keeps the accumulated costs bounded. Pixels are visited so that each
// predecessor is computed before its successor.
func aggregatePath(cost, lr, minLr []int, rows, cols, numDisp int, dir [2]int, p1, p2 int) {
	dy, dx := dir[0], dir[1]

	ys := iterOrder(rows, dy)
	xs := iterOrder(cols, dx)

	for _, y := range ys {
		for _, x := range xs {
			idx := y*cols + x
			base := idx * numDisp
			py, px := y-dy, x-dx

			if py < 0 || py >= rows || px < 0 || px >= cols {
				// Path start: seed with the raw data cost.
				mn := 1 << 62
				for d := 0; d < numDisp; d++ {
					v := cost[base+d]
					lr[base+d] = v
					if v < mn {
						mn = v
					}
				}
				minLr[idx] = mn
				continue
			}

			pBase := (py*cols + px) * numDisp
			prevMin := minLr[py*cols+px]
			mn := 1 << 62
			for d := 0; d < numDisp; d++ {
				best := lr[pBase+d]
				if d > 0 {
					if c := lr[pBase+d-1] + p1; c < best {
						best = c
					}
				}
				if d < numDisp-1 {
					if c := lr[pBase+d+1] + p1; c < best {
						best = c
					}
				}
				if c := prevMin + p2; c < best {
					best = c
				}
				v := cost[base+d] + best - prevMin
				lr[base+d] = v
				if v < mn {
					mn = v
				}
			}
			minLr[idx] = mn
		}
	}
}

// iterOrder returns the indices 0..n-1 in ascending order when step >= 0 and in
// descending order when step < 0, so that path predecessors are visited first.
func iterOrder(n, step int) []int {
	out := make([]int, n)
	if step < 0 {
		for i := 0; i < n; i++ {
			out[i] = n - 1 - i
		}
	} else {
		for i := 0; i < n; i++ {
			out[i] = i
		}
	}
	return out
}
