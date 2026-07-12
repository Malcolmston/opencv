package stereo

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// InvalidDisparityF is the sentinel stored in a [DisparityF] for pixels with no
// reliable match. It is negative so it is distinguishable from every genuine
// disparity (which are non-negative), unlike the uint8 maps where 0 doubles as
// the invalid marker.
const InvalidDisparityF float32 = -1

// DisparityF is a single-channel, row-major map of sub-pixel disparities held as
// float32 values. It complements the uint8 [github.com/malcolmston/opencv.Mat]
// disparity maps returned by the integer matchers: sub-pixel refinement
// ([RefineSubpixel], [StereoSGM.ComputeFloat]) needs the extra precision that a
// float carries. Pixels without a match hold [InvalidDisparityF].
type DisparityF struct {
	// Rows is the map height.
	Rows int
	// Cols is the map width.
	Cols int
	// Data holds Rows*Cols samples in row-major order.
	Data []float32
}

// NewDisparityF allocates a DisparityF whose samples are all
// [InvalidDisparityF]. It panics if either dimension is not positive.
func NewDisparityF(rows, cols int) *DisparityF {
	if rows <= 0 || cols <= 0 {
		panic(fmt.Sprintf("stereo: NewDisparityF requires positive dimensions, got %dx%d", rows, cols))
	}
	d := &DisparityF{Rows: rows, Cols: cols, Data: make([]float32, rows*cols)}
	for i := range d.Data {
		d.Data[i] = InvalidDisparityF
	}
	return d
}

// At returns the disparity at row y, column x. It panics if the coordinates are
// out of range.
func (d *DisparityF) At(y, x int) float32 {
	if y < 0 || y >= d.Rows || x < 0 || x >= d.Cols {
		panic(fmt.Sprintf("stereo: DisparityF.At(%d,%d) out of range %dx%d", y, x, d.Rows, d.Cols))
	}
	return d.Data[y*d.Cols+x]
}

// ToMat quantises the sub-pixel map to an 8-bit single-channel
// [github.com/malcolmston/opencv.Mat]. Each valid sample is rounded to the
// nearest integer and clamped to [0, 255]; [InvalidDisparityF] samples become
// [InvalidDisparity] (0).
func (d *DisparityF) ToMat() *cv.Mat {
	out := cv.NewMat(d.Rows, d.Cols, 1)
	for i, v := range d.Data {
		if v < 0 || math.IsNaN(float64(v)) {
			out.Data[i] = InvalidDisparity
			continue
		}
		r := int(math.Round(float64(v)))
		out.Data[i] = uint8(clampInt(r, 0, 255))
	}
	return out
}

// Rect is an axis-aligned integer rectangle, analogous to cv::Rect. A rectangle
// with non-positive Width or Height is empty.
type Rect struct {
	// X is the left edge (column) of the rectangle.
	X int
	// Y is the top edge (row) of the rectangle.
	Y int
	// Width is the horizontal extent in pixels.
	Width int
	// Height is the vertical extent in pixels.
	Height int
}

// Empty reports whether the rectangle covers no pixels.
func (r Rect) Empty() bool { return r.Width <= 0 || r.Height <= 0 }

// Area returns the number of pixels the rectangle covers (0 when empty).
func (r Rect) Area() int {
	if r.Empty() {
		return 0
	}
	return r.Width * r.Height
}

// grayMat returns a single-channel 8-bit copy of m, converting three-channel
// input with the root package's RGB->Gray. It panics on empty or unsupported
// input. The result never aliases m.
func grayMat(m *cv.Mat) *cv.Mat {
	if m == nil || m.Empty() {
		panic("stereo: nil or empty input Mat")
	}
	switch m.Channels {
	case 1:
		return m.Clone()
	case 3:
		return cv.CvtColor(m, cv.ColorRGB2Gray)
	default:
		panic(fmt.Sprintf("stereo: input must be 1- or 3-channel, got %d", m.Channels))
	}
}

// matToIntGrid copies a single-channel Mat into a flat []int in row-major order.
func matToIntGrid(m *cv.Mat) (rows, cols int, g []int) {
	rows, cols = m.Rows, m.Cols
	g = make([]int, rows*cols)
	for i := range g {
		g[i] = int(m.Data[i])
	}
	return rows, cols, g
}
