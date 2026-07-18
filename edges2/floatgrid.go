package edges2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// FloatGrid is a dense, single-channel raster of float64 samples in row-major
// order. It is the carrier for the signed, unbounded responses produced by the
// gradient operators and analysis filters in this package (Sobel components,
// Laplacian of Gaussian, difference of Gaussians and gradient magnitude) and
// converts to and from [cv.Mat] rather than duplicating that type.
//
// The value at row y, column x lives at Data[y*Cols+x]. The zero value is not
// usable; build instances with [NewFloatGrid], [MatToFloatGrid] or the
// functions that return one.
type FloatGrid struct {
	// Rows is the grid height (number of rows).
	Rows int
	// Cols is the grid width (number of columns).
	Cols int
	// Data holds Rows*Cols samples in row-major order.
	Data []float64
}

// NewFloatGrid allocates a zero-filled FloatGrid with the given dimensions. It
// panics if either dimension is not positive.
func NewFloatGrid(rows, cols int) *FloatGrid {
	if rows <= 0 || cols <= 0 {
		panic("edges2: NewFloatGrid requires positive dimensions")
	}
	return &FloatGrid{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

// MatToFloatGrid converts a single-channel [cv.Mat] to a FloatGrid, copying its
// samples as float64 values. It panics on multi-channel input.
func MatToFloatGrid(m *cv.Mat) *FloatGrid {
	edges2RequireGray(m, "MatToFloatGrid")
	g := NewFloatGrid(m.Rows, m.Cols)
	for i, v := range m.Data {
		g.Data[i] = float64(v)
	}
	return g
}

// At returns the sample at row y, column x. It panics if the coordinates are
// out of range.
func (g *FloatGrid) At(y, x int) float64 {
	if y < 0 || y >= g.Rows || x < 0 || x >= g.Cols {
		panic("edges2: FloatGrid.At out of range")
	}
	return g.Data[y*g.Cols+x]
}

// Set stores value at row y, column x. It panics if the coordinates are out of
// range.
func (g *FloatGrid) Set(y, x int, value float64) {
	if y < 0 || y >= g.Rows || x < 0 || x >= g.Cols {
		panic("edges2: FloatGrid.Set out of range")
	}
	g.Data[y*g.Cols+x] = value
}

// Clone returns an independent deep copy of the grid.
func (g *FloatGrid) Clone() *FloatGrid {
	out := NewFloatGrid(g.Rows, g.Cols)
	copy(out.Data, g.Data)
	return out
}

// Abs returns a new grid holding the absolute value of every sample.
func (g *FloatGrid) Abs() *FloatGrid {
	out := NewFloatGrid(g.Rows, g.Cols)
	for i, v := range g.Data {
		out.Data[i] = math.Abs(v)
	}
	return out
}

// MinMax returns the smallest and largest samples in the grid. It panics on an
// empty grid.
func (g *FloatGrid) MinMax() (min, max float64) {
	if len(g.Data) == 0 {
		panic("edges2: FloatGrid.MinMax on empty grid")
	}
	min = math.Inf(1)
	max = math.Inf(-1)
	for _, v := range g.Data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return min, max
}

// ToMat converts the grid to an 8-bit [cv.Mat] by rounding each sample and
// clamping it to the [0,255] range (values are not rescaled).
func (g *FloatGrid) ToMat() *cv.Mat {
	m := cv.NewMat(g.Rows, g.Cols, 1)
	for i, v := range g.Data {
		m.Data[i] = edges2ClampU8(v)
	}
	return m
}

// ToMatNormalized converts the grid to an 8-bit [cv.Mat], linearly rescaling
// the full value range [min,max] onto [0,255]. A constant grid maps to all
// zeros.
func (g *FloatGrid) ToMatNormalized() *cv.Mat {
	min, max := g.MinMax()
	m := cv.NewMat(g.Rows, g.Cols, 1)
	span := max - min
	if span == 0 {
		return m
	}
	for i, v := range g.Data {
		m.Data[i] = edges2ClampU8((v - min) / span * 255)
	}
	return m
}

// Threshold returns a binary [cv.Mat] in which every sample strictly greater
// than t becomes foreground (255) and every other sample background (0).
func (g *FloatGrid) Threshold(t float64) *cv.Mat {
	m := cv.NewMat(g.Rows, g.Cols, 1)
	for i, v := range g.Data {
		if v > t {
			m.Data[i] = 255
		}
	}
	return m
}
