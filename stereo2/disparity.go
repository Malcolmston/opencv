package stereo2

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// InvalidDisparity is the sentinel stored in a [DisparityMap] for pixels that
// have no reliable match. It is negative so it can never be confused with a
// genuine disparity, which is always non-negative.
const InvalidDisparity float32 = -1

// InvalidDepth is the sentinel stored in a [DepthMap] for pixels whose depth is
// unknown. It is negative so it can never be confused with a genuine depth,
// which is always positive.
const InvalidDepth float32 = -1

// stereo2invalidCost is the per-pixel penalty assigned in a [CostVolume] to a
// disparity hypothesis that references an out-of-image pixel. It is large
// enough that any aggregated window containing it dominates a genuine cost.
const stereo2invalidCost float32 = 1e9

// stereo2invalidThreshold is the aggregated-cost level above which a winning
// hypothesis is treated as invalid (it must have summed at least one
// [stereo2invalidCost] penalty).
const stereo2invalidThreshold float32 = 1e8

// DisparityMap is a single-channel, row-major map of sub-pixel disparities held
// as float32 values. Pixels without a valid match hold [InvalidDisparity]. It
// is the primary result type of every matcher in this package and complements
// the uint8 [github.com/malcolmston/opencv.Mat] used for raw imagery.
type DisparityMap struct {
	// Rows is the map height in pixels.
	Rows int
	// Cols is the map width in pixels.
	Cols int
	// Data holds Rows*Cols samples in row-major order.
	Data []float32
}

// NewDisparityMap allocates a DisparityMap whose samples are all
// [InvalidDisparity]. It panics if either dimension is not positive.
func NewDisparityMap(rows, cols int) *DisparityMap {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("stereo2: NewDisparityMap requires positive dimensions, got %dx%d", rows, cols))
	}
	d := &DisparityMap{Rows: rows, Cols: cols, Data: make([]float32, rows*cols)}
	for i := range d.Data {
		d.Data[i] = InvalidDisparity
	}
	return d
}

// At returns the disparity at row y, column x. It panics if the coordinates are
// out of range.
func (d *DisparityMap) At(y, x int) float32 {
	if y < 0 || y >= d.Rows || x < 0 || x >= d.Cols {
		panic(fmt.Sprintf("stereo2: DisparityMap.At(%d,%d) out of range %dx%d", y, x, d.Rows, d.Cols))
	}
	return d.Data[y*d.Cols+x]
}

// Set stores the disparity v at row y, column x. It panics if the coordinates
// are out of range.
func (d *DisparityMap) Set(y, x int, v float32) {
	if y < 0 || y >= d.Rows || x < 0 || x >= d.Cols {
		panic(fmt.Sprintf("stereo2: DisparityMap.Set(%d,%d) out of range %dx%d", y, x, d.Rows, d.Cols))
	}
	d.Data[y*d.Cols+x] = v
}

// Size returns the map dimensions as (rows, cols).
func (d *DisparityMap) Size() (rows, cols int) {
	return d.Rows, d.Cols
}

// Valid reports whether the pixel at row y, column x holds a genuine (non-sentinel)
// disparity. It panics if the coordinates are out of range.
func (d *DisparityMap) Valid(y, x int) bool {
	v := d.At(y, x)
	return v >= 0 && !math.IsNaN(float64(v))
}

// Clone returns a deep copy of the map.
func (d *DisparityMap) Clone() *DisparityMap {
	out := &DisparityMap{Rows: d.Rows, Cols: d.Cols, Data: make([]float32, len(d.Data))}
	copy(out.Data, d.Data)
	return out
}

// Fill sets every sample to v.
func (d *DisparityMap) Fill(v float32) {
	for i := range d.Data {
		d.Data[i] = v
	}
}

// CountValid returns the number of pixels holding a genuine disparity.
func (d *DisparityMap) CountValid() int {
	n := 0
	for _, v := range d.Data {
		if v >= 0 && !math.IsNaN(float64(v)) {
			n++
		}
	}
	return n
}

// MinMax returns the smallest and largest genuine disparities in the map and
// whether any valid pixel was found. If none are valid it returns (0, 0, false).
func (d *DisparityMap) MinMax() (min, max float32, ok bool) {
	for _, v := range d.Data {
		if v < 0 || math.IsNaN(float64(v)) {
			continue
		}
		if !ok {
			min, max, ok = v, v, true
			continue
		}
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return min, max, ok
}

// ToMat renders the map to a viewable single-channel 8-bit
// [github.com/malcolmston/opencv.Mat]. Valid disparities are linearly scaled so
// that the map minimum maps to 0 and the map maximum maps to 255; invalid
// pixels become 0. A flat map (min == max) renders every valid pixel as 255.
func (d *DisparityMap) ToMat() *cv.Mat {
	out := cv.NewMat(d.Rows, d.Cols, 1)
	min, max, ok := d.MinMax()
	if !ok {
		return out
	}
	span := max - min
	for y := 0; y < d.Rows; y++ {
		for x := 0; x < d.Cols; x++ {
			v := d.Data[y*d.Cols+x]
			if v < 0 || math.IsNaN(float64(v)) {
				continue
			}
			var u uint8 = 255
			if span > 0 {
				u = uint8(clampInt(int(math.Round(float64((v-min)/span*255))), 0, 255))
			}
			out.Set(y, x, 0, u)
		}
	}
	return out
}

// DepthMap is a single-channel, row-major map of metric depths (typically in
// the same length unit as the stereo baseline) held as float32 values. Pixels
// whose depth is unknown hold [InvalidDepth].
type DepthMap struct {
	// Rows is the map height in pixels.
	Rows int
	// Cols is the map width in pixels.
	Cols int
	// Data holds Rows*Cols depth samples in row-major order.
	Data []float32
}

// NewDepthMap allocates a DepthMap whose samples are all [InvalidDepth]. It
// panics if either dimension is not positive.
func NewDepthMap(rows, cols int) *DepthMap {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("stereo2: NewDepthMap requires positive dimensions, got %dx%d", rows, cols))
	}
	m := &DepthMap{Rows: rows, Cols: cols, Data: make([]float32, rows*cols)}
	for i := range m.Data {
		m.Data[i] = InvalidDepth
	}
	return m
}

// At returns the depth at row y, column x. It panics if the coordinates are out
// of range.
func (m *DepthMap) At(y, x int) float32 {
	if y < 0 || y >= m.Rows || x < 0 || x >= m.Cols {
		panic(fmt.Sprintf("stereo2: DepthMap.At(%d,%d) out of range %dx%d", y, x, m.Rows, m.Cols))
	}
	return m.Data[y*m.Cols+x]
}

// Set stores the depth v at row y, column x. It panics if the coordinates are
// out of range.
func (m *DepthMap) Set(y, x int, v float32) {
	if y < 0 || y >= m.Rows || x < 0 || x >= m.Cols {
		panic(fmt.Sprintf("stereo2: DepthMap.Set(%d,%d) out of range %dx%d", y, x, m.Rows, m.Cols))
	}
	m.Data[y*m.Cols+x] = v
}

// Size returns the map dimensions as (rows, cols).
func (m *DepthMap) Size() (rows, cols int) {
	return m.Rows, m.Cols
}

// Valid reports whether the pixel at row y, column x holds a genuine (positive)
// depth. It panics if the coordinates are out of range.
func (m *DepthMap) Valid(y, x int) bool {
	v := m.At(y, x)
	return v > 0 && !math.IsNaN(float64(v))
}

// Clone returns a deep copy of the map.
func (m *DepthMap) Clone() *DepthMap {
	out := &DepthMap{Rows: m.Rows, Cols: m.Cols, Data: make([]float32, len(m.Data))}
	copy(out.Data, m.Data)
	return out
}

// MinMax returns the smallest and largest genuine depths and whether any valid
// pixel was found. If none are valid it returns (0, 0, false).
func (m *DepthMap) MinMax() (min, max float32, ok bool) {
	for _, v := range m.Data {
		if v <= 0 || math.IsNaN(float64(v)) {
			continue
		}
		if !ok {
			min, max, ok = v, v, true
			continue
		}
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return min, max, ok
}

// ToMat renders the depth map to a viewable single-channel 8-bit
// [github.com/malcolmston/opencv.Mat]. Depths are linearly scaled so that near
// maps to 255 and far maps to 0 (near objects brighter); values outside
// [near, far] are clamped and invalid pixels become 0. It panics if far <= near.
func (m *DepthMap) ToMat(near, far float32) *cv.Mat {
	if far <= near {
		panic(fmt.Sprintf("stereo2: DepthMap.ToMat requires far > near, got near=%g far=%g", near, far))
	}
	out := cv.NewMat(m.Rows, m.Cols, 1)
	span := far - near
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			v := m.Data[y*m.Cols+x]
			if v <= 0 || math.IsNaN(float64(v)) {
				continue
			}
			t := (v - near) / span
			if t < 0 {
				t = 0
			}
			if t > 1 {
				t = 1
			}
			u := uint8(clampInt(int(math.Round(float64((1-t)*255))), 0, 255))
			out.Set(y, x, 0, u)
		}
	}
	return out
}

// stereo2intensity reads pixel (y, x) of m as a luminance value in [0,255].
// Single-channel images return the sample directly; multi-channel images return
// the average of up to the first three channels.
func stereo2intensity(m *cv.Mat, y, x int) float64 {
	if m.Channels == 1 {
		return float64(m.Data[(y*m.Cols+x)*m.Channels])
	}
	n := m.Channels
	if n > 3 {
		n = 3
	}
	base := (y*m.Cols + x) * m.Channels
	var s float64
	for c := 0; c < n; c++ {
		s += float64(m.Data[base+c])
	}
	return s / float64(n)
}

// stereo2checkPair panics if left and right do not have identical dimensions.
func stereo2checkPair(left, right *cv.Mat) {
	if left == nil || right == nil {
		panic("stereo2: nil image")
	}
	if left.Rows != right.Rows || left.Cols != right.Cols {
		panic(fmt.Sprintf("stereo2: image size mismatch %dx%d vs %dx%d",
			left.Rows, left.Cols, right.Rows, right.Cols))
	}
}

// clampInt clamps v to the inclusive range [lo, hi].
func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
