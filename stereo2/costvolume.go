package stereo2

import (
	"fmt"
	"math"
)

// CostVolume is a dense disparity-space image (DSI): for every pixel it holds
// the matching cost of each candidate disparity. It is the intermediate
// representation shared by cost aggregation ([AggregateBoxFilter]), semi-global
// aggregation ([SemiGlobalAggregate]) and disparity extraction
// ([CostVolume.ToDisparity], [RefineSubpixelParabola]).
//
// The candidate at index k corresponds to the actual disparity MinDisparity+k.
type CostVolume struct {
	// Rows is the volume height in pixels.
	Rows int
	// Cols is the volume width in pixels.
	Cols int
	// Disparities is the number of candidate disparities per pixel.
	Disparities int
	// MinDisparity is the disparity represented by candidate index 0.
	MinDisparity int
	// Data holds Rows*Cols*Disparities costs; the cost of candidate k at pixel
	// (y,x) is at index (y*Cols+x)*Disparities + k.
	Data []float32
}

// NewCostVolume allocates a zero-filled cost volume. It panics if any dimension
// is not positive.
func NewCostVolume(rows, cols, disparities, minDisparity int) *CostVolume {
	if rows <= 0 || cols <= 0 || disparities <= 0 {
		panic(fmt.Sprintf("stereo2: NewCostVolume requires positive dimensions, got %dx%dx%d", rows, cols, disparities))
	}
	return &CostVolume{
		Rows:         rows,
		Cols:         cols,
		Disparities:  disparities,
		MinDisparity: minDisparity,
		Data:         make([]float32, rows*cols*disparities),
	}
}

// Dims returns the volume dimensions as (rows, cols, disparities).
func (v *CostVolume) Dims() (rows, cols, disparities int) {
	return v.Rows, v.Cols, v.Disparities
}

// At returns the cost of candidate disparity index k at pixel (y, x). It panics
// if any index is out of range.
func (v *CostVolume) At(y, x, k int) float32 {
	if y < 0 || y >= v.Rows || x < 0 || x >= v.Cols || k < 0 || k >= v.Disparities {
		panic(fmt.Sprintf("stereo2: CostVolume.At(%d,%d,%d) out of range %dx%dx%d", y, x, k, v.Rows, v.Cols, v.Disparities))
	}
	return v.Data[(y*v.Cols+x)*v.Disparities+k]
}

// Set stores the cost c for candidate disparity index k at pixel (y, x). It
// panics if any index is out of range.
func (v *CostVolume) Set(y, x, k int, c float32) {
	if y < 0 || y >= v.Rows || x < 0 || x >= v.Cols || k < 0 || k >= v.Disparities {
		panic(fmt.Sprintf("stereo2: CostVolume.Set(%d,%d,%d) out of range %dx%dx%d", y, x, k, v.Rows, v.Cols, v.Disparities))
	}
	v.Data[(y*v.Cols+x)*v.Disparities+k] = c
}

// ArgMin returns the candidate disparity index with the smallest cost at pixel
// (y, x) together with that cost. Ties resolve to the lowest index. It panics
// if the coordinates are out of range.
func (v *CostVolume) ArgMin(y, x int) (k int, cost float32) {
	if y < 0 || y >= v.Rows || x < 0 || x >= v.Cols {
		panic(fmt.Sprintf("stereo2: CostVolume.ArgMin(%d,%d) out of range %dx%d", y, x, v.Rows, v.Cols))
	}
	base := (y*v.Cols + x) * v.Disparities
	best := 0
	bestC := v.Data[base]
	for d := 1; d < v.Disparities; d++ {
		if v.Data[base+d] < bestC {
			bestC = v.Data[base+d]
			best = d
		}
	}
	return best, bestC
}

// ToDisparity extracts an integer winner-take-all disparity map from the volume.
// For each pixel the candidate with the lowest cost is selected; pixels whose
// best cost still carries an out-of-image penalty are marked [InvalidDisparity].
//
// uniquenessRatio, when positive, enforces the standard uniqueness test: the
// best cost must beat the best cost among non-neighbouring candidates by at
// least uniquenessRatio percent, otherwise the pixel is rejected as ambiguous.
func (v *CostVolume) ToDisparity(uniquenessRatio int) *DisparityMap {
	out := NewDisparityMap(v.Rows, v.Cols)
	for y := 0; y < v.Rows; y++ {
		for x := 0; x < v.Cols; x++ {
			best, bestC := v.ArgMin(y, x)
			if bestC >= stereo2invalidThreshold {
				continue
			}
			if uniquenessRatio > 0 && !stereo2unique(v, y, x, best, bestC, uniquenessRatio) {
				continue
			}
			out.Data[y*v.Cols+x] = float32(v.MinDisparity + best)
		}
	}
	return out
}

// stereo2unique reports whether candidate best at (y,x) wins the uniqueness test
// against every candidate that is not its immediate neighbour.
func stereo2unique(v *CostVolume, y, x, best int, bestC float32, ratio int) bool {
	base := (y*v.Cols + x) * v.Disparities
	secondBest := float32(math.MaxFloat32)
	for d := 0; d < v.Disparities; d++ {
		if d == best || d == best-1 || d == best+1 {
			continue
		}
		if v.Data[base+d] < secondBest {
			secondBest = v.Data[base+d]
		}
	}
	if secondBest == float32(math.MaxFloat32) {
		return true
	}
	// best must be better than secondBest by ratio percent:
	// bestC * (100 + ratio) <= secondBest * 100.
	return bestC*float32(100+ratio) <= secondBest*100
}
