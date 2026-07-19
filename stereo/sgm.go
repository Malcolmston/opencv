package stereo

import (
	cv "github.com/malcolmston/opencv"
)

// SGMMode selects the set of aggregation paths used by [StereoSGM].
type SGMMode int

// ModeSGBM and ModeHH are the supported aggregation-path sets.
const (
	// ModeSGBM aggregates along the four cardinal directions (left, right, up,
	// down), like the lighter [StereoSGBM].
	ModeSGBM SGMMode = iota
	// ModeHH aggregates along all eight directions — the four cardinals plus the
	// four diagonals — matching OpenCV's MODE_HH full semi-global matching. It is
	// the most accurate mode and the default for [StereoSGM].
	ModeHH
)

// eightPaths lists all eight aggregation directions {dy, dx}: the four cardinals
// followed by the four diagonals.
var eightPaths = [8][2]int{
	{0, 1}, {0, -1}, {1, 0}, {-1, 0},
	{1, 1}, {1, -1}, {-1, 1}, {-1, -1},
}

// StereoSGM is a full semi-global matcher. Unlike the four-path [StereoSGBM],
// it optionally aggregates the SGM smoothness cost along all eight directions
// (diagonals included, [ModeHH]), supports an arbitrary matching [CostType]
// including illumination-robust census, applies a left-right consistency check
// via Disp12MaxDiff, and can refine the winner to sub-pixel precision with an
// equiangular parabola fit ([StereoSGM.ComputeFloat]).
//
// The zero value aggregates four paths with SAD costs; set at least
// NumDisparities and BlockSize. Remaining fields default when non-positive.
type StereoSGM struct {
	// MinDisparity is the smallest disparity searched (usually 0).
	MinDisparity int
	// NumDisparities is the width of the search range; disparities d in
	// [MinDisparity, MinDisparity+NumDisparities) are considered. Defaults to 16.
	NumDisparities int
	// BlockSize is the odd side length of the SAD/census data window. Defaults to 5.
	BlockSize int
	// P1 penalises a disparity change of one between neighbours. Defaults to 8*BlockSize.
	P1 int
	// P2 penalises larger disparity jumps and must exceed P1. Defaults to 32*BlockSize.
	P2 int
	// UniquenessRatio is the percent margin by which the best aggregated cost must
	// beat the second-best non-adjacent cost. Defaults to 10.
	UniquenessRatio int
	// Disp12MaxDiff is the maximum allowed left-right disparity disagreement, in
	// pixels, in the consistency check; a negative value disables the check.
	// The zero value enables a strict (0-pixel) check.
	Disp12MaxDiff int
	// Mode selects the aggregation-path layout. The zero value is [ModeSGBM]
	// (four paths); set [ModeHH] for full eight-path aggregation.
	Mode SGMMode
	// CostType selects the matching cost. The zero value is [CostSAD].
	CostType CostType
}

func (s StereoSGM) resolved() (minD, numD, block, p1, p2, uniq int) {
	minD = s.MinDisparity
	numD = s.NumDisparities
	if numD <= 0 {
		numD = 16
	}
	block = s.BlockSize
	if block <= 0 {
		block = 5
	}
	requireOdd(block, "StereoSGM.BlockSize")
	p1 = s.P1
	if p1 <= 0 {
		p1 = 8 * block
	}
	p2 = s.P2
	if p2 <= 0 {
		p2 = 32 * block
	}
	if p2 < p1 {
		p2 = p1
	}
	uniq = s.UniquenessRatio
	if uniq <= 0 {
		uniq = 10
	}
	return minD, numD, block, p1, p2, uniq
}

// Aggregate builds the semi-global aggregated cost volume for a rectified pair.
// It first forms the raw matching cost with the configured [CostType] and block
// window, then sums the SGM path recurrence over four or eight directions
// depending on Mode. The returned volume shares MinDisparity/NumDisparities with
// the data volume and is what both [StereoSGM.Compute] and
// [StereoSGM.ComputeFloat] decode.
func (s StereoSGM) Aggregate(left, right *cv.Mat) *CostVolume {
	minD, numD, block, p1, p2, _ := s.resolved()
	data := MatchingCostVolume(left, right, minD, numD, block, s.CostType)
	rows, cols := data.Rows, data.Cols
	n := rows * cols

	cost := make([]int, n*numD)
	for i, c := range data.Data {
		cost[i] = int(c)
	}

	paths := eightPaths[:4]
	if s.Mode == ModeHH {
		paths = eightPaths[:]
	}

	agg := make([]int, n*numD)
	lr := make([]int, n*numD)
	minLr := make([]int, n)
	for _, dir := range paths {
		aggregatePath(cost, lr, minLr, rows, cols, numD, dir, p1, p2)
		for i := range agg {
			agg[i] += lr[i]
		}
	}

	out := newCostVolume(rows, cols, minD, numD)
	for i := range out.Data {
		out.Data[i] = int32(agg[i])
	}
	return out
}

// Compute matches left against right and returns a single-channel 8-bit
// disparity map. It aggregates the cost volume ([StereoSGM.Aggregate]), decodes
// the winner at each pixel, rejects ambiguous pixels by the uniqueness ratio and
// occluded pixels by the Disp12MaxDiff left-right check. Unmatched pixels hold
// [InvalidDisparity].
//
// It panics on empty input, a size or channel mismatch, or an even BlockSize.
func (s StereoSGM) Compute(left, right *cv.Mat) *cv.Mat {
	return s.ComputeFloat(left, right).ToMat()
}

// ComputeFloat is like [StereoSGM.Compute] but returns a sub-pixel [DisparityF].
// After winner-take-all decoding it fits an equiangular parabola to the
// aggregated costs at d-1, d and d+1 and shifts the disparity by the parabola
// vertex, recovering fractional disparities. Rejected pixels hold
// [InvalidDisparityF].
func (s StereoSGM) ComputeFloat(left, right *cv.Mat) *DisparityF {
	minD, numD, _, _, _, uniq := s.resolved()
	agg := s.Aggregate(left, right)
	rows, cols := agg.Rows, agg.Cols

	// Right-view disparities for the left-right consistency check.
	dispRight := rightDisparities(agg)

	out := NewDisparityF(rows, cols)
	borderLimit := minD + numD - 1
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if x < borderLimit {
				continue
			}
			base := (y*cols + x) * numD
			bestIdx, bestC := 0, costSentinel
			for idx := 0; idx < numD; idx++ {
				if c := agg.Data[base+idx]; c < bestC {
					bestC, bestIdx = c, idx
				}
			}
			// Uniqueness: second-best non-adjacent cost must be clearly worse.
			secondC := costSentinel
			for idx := 0; idx < numD; idx++ {
				if idx < bestIdx-1 || idx > bestIdx+1 {
					if c := agg.Data[base+idx]; c < secondC {
						secondC = c
					}
				}
			}
			if secondC != costSentinel && int64(secondC)*100 <= int64(bestC)*int64(100+uniq) {
				continue
			}
			dL := minD + bestIdx
			// Left-right consistency (disp12MaxDiff).
			if s.Disp12MaxDiff >= 0 {
				xr := x - dL
				if xr < 0 {
					continue
				}
				dR := dispRight[y*cols+xr]
				if dR < 0 || absInt(dL-dR) > s.Disp12MaxDiff {
					continue
				}
			}
			// Sub-pixel parabola refinement.
			off := 0.0
			if bestIdx > 0 && bestIdx < numD-1 {
				off = SubpixelParabola(int(agg.Data[base+bestIdx-1]), int(bestC), int(agg.Data[base+bestIdx+1]))
			}
			out.Data[y*cols+x] = float32(float64(dL) + off)
		}
	}
	return out
}

// rightDisparities decodes a right-view integer disparity for every column from
// a left-referenced aggregated volume, using the standard relation that right
// pixel xr competes for left pixel xr+d at disparity d. Columns with no
// candidate hold -1.
func rightDisparities(agg *CostVolume) []int {
	rows, cols := agg.Rows, agg.Cols
	numD := agg.NumDisparities
	minD := agg.MinDisparity
	out := make([]int, rows*cols)
	for y := 0; y < rows; y++ {
		for xr := 0; xr < cols; xr++ {
			bestD, bestC := -1, costSentinel
			for idx := 0; idx < numD; idx++ {
				d := minD + idx
				xl := xr + d
				if xl >= cols {
					break
				}
				if xl < 0 {
					continue
				}
				c := agg.Data[(y*cols+xl)*numD+idx]
				if c < bestC {
					bestC, bestD = c, d
				}
			}
			out[y*cols+xr] = bestD
		}
	}
	return out
}

// absInt returns the absolute value of v.
func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
