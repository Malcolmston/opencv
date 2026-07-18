package stereo2

import (
	"fmt"
	"math/bits"

	cv "github.com/malcolmston/opencv"
)

// CensusMap holds the census-transform signature of every pixel of an image.
// The census transform encodes, for each pixel, the sign of its intensity
// relative to each neighbour in a rectangular window as one bit of a bitstring;
// two signatures are compared with the Hamming distance ([HammingDistance]),
// giving a matching cost that is invariant to monotonic intensity changes such
// as differing camera gain between the two views.
//
// Border pixels whose window falls partly outside the image hold a zero
// signature.
type CensusMap struct {
	// Rows is the map height in pixels.
	Rows int
	// Cols is the map width in pixels.
	Cols int
	// WindowW is the census window width (odd).
	WindowW int
	// WindowH is the census window height (odd).
	WindowH int
	// Data holds Rows*Cols signatures in row-major order.
	Data []uint64
}

// CensusTransform computes the census signature of every pixel using a
// windowW-by-windowH neighbourhood (both rounded up to odd). It panics if the
// window has more than 64 comparison neighbours, since a signature is stored in
// a single uint64.
func CensusTransform(img *cv.Mat, windowW, windowH int) *CensusMap {
	if img == nil {
		panic("stereo2: CensusTransform on nil image")
	}
	if windowW <= 0 || windowH <= 0 {
		panic(fmt.Sprintf("stereo2: CensusTransform requires positive window, got %dx%d", windowW, windowH))
	}
	if windowW%2 == 0 {
		windowW++
	}
	if windowH%2 == 0 {
		windowH++
	}
	if windowW*windowH-1 > 64 {
		panic(fmt.Sprintf("stereo2: census window %dx%d has more than 64 neighbours", windowW, windowH))
	}
	rw, rh := windowW/2, windowH/2
	rows, cols := img.Rows, img.Cols
	cm := &CensusMap{Rows: rows, Cols: cols, WindowW: windowW, WindowH: windowH, Data: make([]uint64, rows*cols)}
	for y := rh; y < rows-rh; y++ {
		for x := rw; x < cols-rw; x++ {
			center := stereo2intensity(img, y, x)
			var sig uint64
			var bit uint
			for j := -rh; j <= rh; j++ {
				for i := -rw; i <= rw; i++ {
					if i == 0 && j == 0 {
						continue
					}
					if stereo2intensity(img, y+j, x+i) < center {
						sig |= 1 << bit
					}
					bit++
				}
			}
			cm.Data[y*cols+x] = sig
		}
	}
	return cm
}

// At returns the census signature at row y, column x. It panics if the
// coordinates are out of range.
func (c *CensusMap) At(y, x int) uint64 {
	if y < 0 || y >= c.Rows || x < 0 || x >= c.Cols {
		panic(fmt.Sprintf("stereo2: CensusMap.At(%d,%d) out of range %dx%d", y, x, c.Rows, c.Cols))
	}
	return c.Data[y*c.Cols+x]
}

// Size returns the map dimensions as (rows, cols).
func (c *CensusMap) Size() (rows, cols int) {
	return c.Rows, c.Cols
}

// HammingDistance returns the number of differing bits between two census
// signatures, i.e. the population count of a XOR b.
func HammingDistance(a, b uint64) int {
	return bits.OnesCount64(a ^ b)
}

// BuildCensusCostVolume constructs a per-pixel matching cost volume whose cost
// is the Hamming distance between census signatures. For reference pixel (y,x)
// and candidate disparity d = minDisparity+k the cost compares left(y,x) with
// right(y,x-d). Hypotheses referencing a column outside the right map receive an
// out-of-image penalty. The two maps must have identical dimensions and window.
//
// Aggregate the result with [AggregateBoxFilter] (local) or
// [SemiGlobalAggregate] (SGM) before extracting disparities.
func BuildCensusCostVolume(left, right *CensusMap, minDisparity, numDisparities int) *CostVolume {
	if left == nil || right == nil {
		panic("stereo2: nil census map")
	}
	if left.Rows != right.Rows || left.Cols != right.Cols {
		panic(fmt.Sprintf("stereo2: census size mismatch %dx%d vs %dx%d", left.Rows, left.Cols, right.Rows, right.Cols))
	}
	if numDisparities <= 0 {
		panic(fmt.Sprintf("stereo2: BuildCensusCostVolume requires numDisparities > 0, got %d", numDisparities))
	}
	rows, cols := left.Rows, left.Cols
	vol := NewCostVolume(rows, cols, numDisparities, minDisparity)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			ls := left.Data[y*cols+x]
			base := (y*cols + x) * numDisparities
			for k := 0; k < numDisparities; k++ {
				xr := x - (minDisparity + k)
				if xr < 0 || xr >= cols {
					vol.Data[base+k] = stereo2invalidCost
					continue
				}
				vol.Data[base+k] = float32(HammingDistance(ls, right.Data[y*cols+xr]))
			}
		}
	}
	return vol
}

// CensusMatcher is a stereo matcher that scores disparities by aggregated
// Hamming distance between census signatures. It is markedly more robust to
// radiometric differences between the two cameras than raw intensity block
// matching.
//
// The zero value is not useful; construct one with [NewCensusMatcher].
type CensusMatcher struct {
	// MinDisparity is the smallest disparity searched.
	MinDisparity int
	// NumDisparities is the width of the disparity search range in pixels.
	NumDisparities int
	// CensusW is the census window width (odd).
	CensusW int
	// CensusH is the census window height (odd).
	CensusH int
	// AggRadius is the radius of the square cost-aggregation window; 0 aggregates
	// nothing (pure per-pixel census).
	AggRadius int
	// UniquenessRatio, when positive, applies the uniqueness test during
	// disparity extraction.
	UniquenessRatio int
}

// NewCensusMatcher returns a CensusMatcher with the given search range, a
// censusW-by-censusH census window and the given aggregation radius. It panics
// if numDisparities or the window sizes are not positive.
func NewCensusMatcher(minDisparity, numDisparities, censusW, censusH, aggRadius int) *CensusMatcher {
	if numDisparities <= 0 || censusW <= 0 || censusH <= 0 {
		panic("stereo2: NewCensusMatcher requires positive numDisparities and window")
	}
	return &CensusMatcher{
		MinDisparity:   minDisparity,
		NumDisparities: numDisparities,
		CensusW:        censusW,
		CensusH:        censusH,
		AggRadius:      aggRadius,
	}
}

// Compute matches left against right and returns the left-referenced disparity
// map. It panics if the images differ in size.
func (cm *CensusMatcher) Compute(left, right *cv.Mat) *DisparityMap {
	stereo2checkPair(left, right)
	lc := CensusTransform(left, cm.CensusW, cm.CensusH)
	rc := CensusTransform(right, cm.CensusW, cm.CensusH)
	vol := BuildCensusCostVolume(lc, rc, cm.MinDisparity, cm.NumDisparities)
	if cm.AggRadius > 0 {
		vol = AggregateBoxFilter(vol, cm.AggRadius)
	}
	return cm.finish(vol, lc, rc)
}

// ComputeRight matches right against left and returns the right-referenced
// disparity map for use with [LeftRightCheck]. It panics if the images differ
// in size.
func (cm *CensusMatcher) ComputeRight(left, right *cv.Mat) *DisparityMap {
	stereo2checkPair(left, right)
	lc := CensusTransform(left, cm.CensusW, cm.CensusH)
	rc := CensusTransform(right, cm.CensusW, cm.CensusH)
	// Right-referenced volume: for right pixel (y,x), candidate d matches left
	// column x+d.
	rows, cols := left.Rows, left.Cols
	vol := NewCostVolume(rows, cols, cm.NumDisparities, cm.MinDisparity)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			rs := rc.Data[y*cols+x]
			base := (y*cols + x) * cm.NumDisparities
			for k := 0; k < cm.NumDisparities; k++ {
				xl := x + (cm.MinDisparity + k)
				if xl < 0 || xl >= cols {
					vol.Data[base+k] = stereo2invalidCost
					continue
				}
				vol.Data[base+k] = float32(HammingDistance(rs, lc.Data[y*cols+xl]))
			}
		}
	}
	if cm.AggRadius > 0 {
		vol = AggregateBoxFilter(vol, cm.AggRadius)
	}
	return vol.ToDisparity(cm.UniquenessRatio)
}

// finish extracts the disparity map from an aggregated cost volume, also
// invalidating the census border where signatures are undefined.
func (cm *CensusMatcher) finish(vol *CostVolume, lc, rc *CensusMap) *DisparityMap {
	disp := vol.ToDisparity(cm.UniquenessRatio)
	rw, rh := lc.WindowW/2, lc.WindowH/2
	ar := cm.AggRadius
	for y := 0; y < disp.Rows; y++ {
		for x := 0; x < disp.Cols; x++ {
			if y < rh+ar || y >= disp.Rows-rh-ar || x < rw+ar || x >= disp.Cols-rw-ar {
				disp.Data[y*disp.Cols+x] = InvalidDisparity
			}
		}
	}
	return disp
}
