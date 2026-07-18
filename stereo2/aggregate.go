package stereo2

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// BuildCostVolume constructs the raw, un-aggregated per-pixel matching cost
// volume for a rectified stereo pair. For reference pixel (y,x) and candidate
// disparity d = minDisparity+k the cost is the pointwise dissimilarity between
// left(y,x) and right(y,x-d): absolute intensity difference when squared is
// false, squared difference otherwise. Hypotheses that reference a column
// outside the right image are given an out-of-image penalty so they are never
// selected.
//
// The result is the input to [AggregateBoxFilter] (for block matching) or
// [SemiGlobalAggregate] (for SGM). It panics if the images differ in size or if
// numDisparities is not positive.
func BuildCostVolume(left, right *cv.Mat, minDisparity, numDisparities int, squared bool) *CostVolume {
	stereo2checkPair(left, right)
	if numDisparities <= 0 {
		panic(fmt.Sprintf("stereo2: BuildCostVolume requires numDisparities > 0, got %d", numDisparities))
	}
	rows, cols := left.Rows, left.Cols
	vol := NewCostVolume(rows, cols, numDisparities, minDisparity)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			lv := stereo2intensity(left, y, x)
			base := (y*cols + x) * numDisparities
			for k := 0; k < numDisparities; k++ {
				xr := x - (minDisparity + k)
				if xr < 0 || xr >= cols {
					vol.Data[base+k] = stereo2invalidCost
					continue
				}
				diff := lv - stereo2intensity(right, y, xr)
				if diff < 0 {
					diff = -diff
				}
				if squared {
					vol.Data[base+k] = float32(diff * diff)
				} else {
					vol.Data[base+k] = float32(diff)
				}
			}
		}
	}
	return vol
}

// AggregateBoxFilter sums the matching cost of each disparity candidate over a
// square (2*radius+1) window, turning a per-pixel cost volume into an
// SAD/SSD-style block cost volume. Aggregation is performed separably (a
// horizontal pass followed by a vertical pass) with edge replication at the
// borders. It panics if radius is negative.
//
// The out-of-image penalties placed by [BuildCostVolume] survive aggregation,
// so [CostVolume.ToDisparity] still marks border pixels invalid.
func AggregateBoxFilter(vol *CostVolume, radius int) *CostVolume {
	if radius < 0 {
		panic(fmt.Sprintf("stereo2: AggregateBoxFilter requires radius >= 0, got %d", radius))
	}
	rows, cols, disp := vol.Rows, vol.Cols, vol.Disparities
	out := NewCostVolume(rows, cols, disp, vol.MinDisparity)
	if radius == 0 {
		copy(out.Data, vol.Data)
		return out
	}
	tmp := make([]float32, len(vol.Data))
	// Horizontal pass: tmp = box sum over columns of vol.
	for y := 0; y < rows; y++ {
		row := y * cols
		for k := 0; k < disp; k++ {
			for x := 0; x < cols; x++ {
				var s float32
				for i := -radius; i <= radius; i++ {
					xx := clampInt(x+i, 0, cols-1)
					s += vol.Data[(row+xx)*disp+k]
				}
				tmp[(row+x)*disp+k] = s
			}
		}
	}
	// Vertical pass: out = box sum over rows of tmp.
	for x := 0; x < cols; x++ {
		for k := 0; k < disp; k++ {
			for y := 0; y < rows; y++ {
				var s float32
				for j := -radius; j <= radius; j++ {
					yy := clampInt(y+j, 0, rows-1)
					s += tmp[(yy*cols+x)*disp+k]
				}
				out.Data[(y*cols+x)*disp+k] = s
			}
		}
	}
	return out
}
