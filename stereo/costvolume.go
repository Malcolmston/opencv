package stereo

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// costInfinity is the sentinel matching cost assigned to disparities whose
// right-image sample falls outside the picture. It is large enough to never win
// a minimisation yet small enough that path-aggregation sums stay well within
// int32.
const costInfinity = int32(1 << 24)

// costSentinel is a "larger than any real (possibly aggregated) cost" value used
// to seed best/second-best searches. It stays well within int32 even after
// eight-path summation.
const costSentinel = int32(1 << 30)

// CostType selects the per-pixel dissimilarity measure used when building a
// [CostVolume] with [MatchingCostVolume].
type CostType int

// CostSAD, CostSSD, CostAD and CostCensus are the supported dissimilarity
// measures.
const (
	// CostSAD is the sum of absolute intensity differences over the block window.
	CostSAD CostType = iota
	// CostSSD is the sum of squared intensity differences over the block window.
	CostSSD
	// CostAD is the pixelwise absolute difference of the block-window means, a
	// cheap smoothed absolute difference.
	CostAD
	// CostCensus is the block-summed Hamming distance of 5×5 census codes; it is
	// robust to gain and bias changes between the two views.
	CostCensus
)

// CostVolume holds the dense matching cost of a rectified stereo pair: for every
// pixel (y, x) it stores the cost of assigning each candidate disparity in the
// range [MinDisparity, MinDisparity+NumDisparities). It is the shared substrate
// for winner-take-all decoding ([CostVolume.WinnerTakeAll]), semi-global
// aggregation ([StereoSGM]), confidence estimation ([ComputeConfidence]) and
// sub-pixel refinement ([RefineSubpixel]).
type CostVolume struct {
	// Rows and Cols are the image dimensions.
	Rows, Cols int
	// MinDisparity is the disparity of index 0.
	MinDisparity int
	// NumDisparities is the number of disparity hypotheses per pixel.
	NumDisparities int
	// Data holds Rows*Cols*NumDisparities costs, indexed as
	// (y*Cols+x)*NumDisparities + idx where idx is d-MinDisparity.
	Data []int32
}

// newCostVolume allocates a zeroed cost volume.
func newCostVolume(rows, cols, minDisparity, numDisparities int) *CostVolume {
	return &CostVolume{
		Rows:           rows,
		Cols:           cols,
		MinDisparity:   minDisparity,
		NumDisparities: numDisparities,
		Data:           make([]int32, rows*cols*numDisparities),
	}
}

// At returns the cost of disparity d at pixel (y, x). It panics if d is outside
// the volume's disparity range or the coordinates are out of bounds.
func (v *CostVolume) At(y, x, d int) int32 {
	if y < 0 || y >= v.Rows || x < 0 || x >= v.Cols {
		panic(fmt.Sprintf("stereo: CostVolume.At(%d,%d) out of range %dx%d", y, x, v.Rows, v.Cols))
	}
	idx := d - v.MinDisparity
	if idx < 0 || idx >= v.NumDisparities {
		panic(fmt.Sprintf("stereo: CostVolume.At disparity %d outside [%d,%d)", d, v.MinDisparity, v.MinDisparity+v.NumDisparities))
	}
	return v.Data[(y*v.Cols+x)*v.NumDisparities+idx]
}

// WinnerTakeAll decodes the volume to an 8-bit disparity map by choosing, at
// each pixel, the disparity with the smallest cost. Pixels in the left border
// band of width MinDisparity+NumDisparities-1 (where the full search range is
// unavailable) are set to [InvalidDisparity], matching the convention of the
// other matchers. Decoded disparities are clamped to [0, 255].
func (v *CostVolume) WinnerTakeAll() *cv.Mat {
	out := cv.NewMat(v.Rows, v.Cols, 1)
	borderLimit := v.MinDisparity + v.NumDisparities - 1
	for y := 0; y < v.Rows; y++ {
		for x := 0; x < v.Cols; x++ {
			if x < borderLimit {
				continue
			}
			base := (y*v.Cols + x) * v.NumDisparities
			bestIdx, bestC := 0, costSentinel
			for idx := 0; idx < v.NumDisparities; idx++ {
				if c := v.Data[base+idx]; c < bestC {
					bestC, bestIdx = c, idx
				}
			}
			out.Data[y*v.Cols+x] = uint8(clampInt(v.MinDisparity+bestIdx, 0, 255))
		}
	}
	return out
}

// MatchingCostVolume builds a [CostVolume] for a rectified pair using the chosen
// [CostType]. For each left pixel (y, x) and disparity d = minDisparity+idx it
// scores the left block window against the right window centred at (y, x-d);
// right samples outside the picture use [costInfinity] so out-of-range
// disparities are never selected. Inputs may be single- or three-channel and
// must share dimensions.
//
// It panics on empty input, a size or channel mismatch, an even blockSize, or a
// non-positive numDisparities.
func MatchingCostVolume(left, right *cv.Mat, minDisparity, numDisparities, blockSize int, cost CostType) *CostVolume {
	if numDisparities <= 0 {
		panic("stereo: MatchingCostVolume requires numDisparities > 0")
	}
	requireOdd(blockSize, "MatchingCostVolume.blockSize")
	if cost == CostCensus {
		return CensusCostVolume(left, right, minDisparity, numDisparities, 5, 5, blockSize)
	}
	rows, cols, gl := toGrayGrid(left)
	rrows, rcols, gr := toGrayGrid(right)
	if rows != rrows || cols != rcols {
		panic(fmt.Sprintf("stereo: MatchingCostVolume size mismatch left %dx%d right %dx%d", rows, cols, rrows, rcols))
	}
	half := blockSize / 2
	vol := newCostVolume(rows, cols, minDisparity, numDisparities)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			base := (y*cols + x) * numDisparities
			for idx := 0; idx < numDisparities; idx++ {
				d := minDisparity + idx
				if x-d < 0 {
					vol.Data[base+idx] = costInfinity
					continue
				}
				vol.Data[base+idx] = int32(pixelCost(gl, gr, rows, cols, y, x, d, half, cost))
			}
		}
	}
	return vol
}

// pixelCost evaluates one dissimilarity measure for the left window at (y,x)
// against the right window at (y,x-d).
func pixelCost(gl, gr []int, rows, cols, y, x, d, half int, cost CostType) int {
	switch cost {
	case CostSAD:
		return blockSAD(gl, gr, rows, cols, y, x, d, half)
	case CostSSD:
		return blockSSD(gl, gr, rows, cols, y, x, d, half)
	case CostAD:
		return blockMeanAD(gl, gr, rows, cols, y, x, d, half)
	default:
		return blockSAD(gl, gr, rows, cols, y, x, d, half)
	}
}

// blockSSD returns the sum of squared differences between the left window
// centred at (y,x) and the right window centred at (y,x-d), edges replicated.
func blockSSD(gl, gr []int, rows, cols, y, x, d, half int) int {
	s := 0
	for dy := -half; dy <= half; dy++ {
		yy := clampInt(y+dy, 0, rows-1)
		rowBase := yy * cols
		for dx := -half; dx <= half; dx++ {
			lx := clampInt(x+dx, 0, cols-1)
			rx := clampInt(x-d+dx, 0, cols-1)
			diff := gl[rowBase+lx] - gr[rowBase+rx]
			s += diff * diff
		}
	}
	return s
}

// blockMeanAD returns the absolute difference of the two block-window means, a
// smoothed absolute difference that suppresses pixel noise.
func blockMeanAD(gl, gr []int, rows, cols, y, x, d, half int) int {
	sumL, sumR, n := 0, 0, 0
	for dy := -half; dy <= half; dy++ {
		yy := clampInt(y+dy, 0, rows-1)
		rowBase := yy * cols
		for dx := -half; dx <= half; dx++ {
			lx := clampInt(x+dx, 0, cols-1)
			rx := clampInt(x-d+dx, 0, cols-1)
			sumL += gl[rowBase+lx]
			sumR += gr[rowBase+rx]
			n++
		}
	}
	diff := sumL/n - sumR/n
	if diff < 0 {
		diff = -diff
	}
	return diff
}
