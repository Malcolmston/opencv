package optflow

import "math"

// FlowField is a dense, two-channel float64 motion field: for every pixel it
// stores a horizontal (X / u) and vertical (Y / v) displacement measured from a
// previous frame to a next frame. It is the two-channel float analogue of the
// root package's single-channel matrices, which cannot hold an interleaved
// (u, v) pair.
//
// Samples are stored interleaved in Data with length Rows*Cols*2: the flow for
// row y, column x is at Data[(y*Cols+x)*2] (u, horizontal) and
// Data[(y*Cols+x)*2+1] (v, vertical). Positive u points right (increasing
// column), positive v points down (increasing row), matching the image
// coordinate convention used throughout the root cv package.
type FlowField struct {
	// Rows is the field height.
	Rows int
	// Cols is the field width.
	Cols int
	// Data holds the interleaved (u, v) displacements, row-major, length
	// Rows*Cols*2.
	Data []float64
}

// NewFlowField allocates a zero-filled FlowField. It panics if a dimension is
// not positive.
func NewFlowField(rows, cols int) *FlowField {
	if rows <= 0 || cols <= 0 {
		panic("optflow: NewFlowField requires positive dimensions")
	}
	return &FlowField{Rows: rows, Cols: cols, Data: make([]float64, rows*cols*2)}
}

// At returns the (u, v) displacement stored at row y, column x. It panics if the
// coordinates are out of range.
func (f *FlowField) At(y, x int) (u, v float64) {
	if y < 0 || y >= f.Rows || x < 0 || x >= f.Cols {
		panic("optflow: FlowField.At out of range")
	}
	i := (y*f.Cols + x) * 2
	return f.Data[i], f.Data[i+1]
}

// Set stores the (u, v) displacement at row y, column x. It panics if the
// coordinates are out of range.
func (f *FlowField) Set(y, x int, u, v float64) {
	if y < 0 || y >= f.Rows || x < 0 || x >= f.Cols {
		panic("optflow: FlowField.Set out of range")
	}
	i := (y*f.Cols + x) * 2
	f.Data[i] = u
	f.Data[i+1] = v
}

// set is the unchecked internal setter used on hot paths.
func (f *FlowField) set(y, x int, u, v float64) {
	i := (y*f.Cols + x) * 2
	f.Data[i] = u
	f.Data[i+1] = v
}

// Clone returns a deep copy of the field with its own backing storage.
func (f *FlowField) Clone() *FlowField {
	out := NewFlowField(f.Rows, f.Cols)
	copy(out.Data, f.Data)
	return out
}

// MeanFlow returns the average (u, v) displacement over the interior of the
// field, excluding a border of the given width where estimates are least
// reliable. border is clamped to a non-negative value that leaves a non-empty
// interior; if none remains the mean over the whole field is returned.
func (f *FlowField) MeanFlow(border int) (u, v float64) {
	if border < 0 {
		border = 0
	}
	if 2*border >= f.Rows || 2*border >= f.Cols {
		border = 0
	}
	var su, sv float64
	var n int
	for y := border; y < f.Rows-border; y++ {
		for x := border; x < f.Cols-border; x++ {
			i := (y*f.Cols + x) * 2
			su += f.Data[i]
			sv += f.Data[i+1]
			n++
		}
	}
	if n == 0 {
		return 0, 0
	}
	return su / float64(n), sv / float64(n)
}

// Magnitude returns the per-pixel flow magnitude sqrt(u²+v²) as a row-major
// slice of length Rows*Cols. It is convenient for thresholding, normalisation
// or visualisation.
func (f *FlowField) Magnitude() []float64 {
	out := make([]float64, f.Rows*f.Cols)
	for p := 0; p < len(out); p++ {
		u := f.Data[p*2]
		v := f.Data[p*2+1]
		out[p] = math.Hypot(u, v)
	}
	return out
}

// MaxMagnitude returns the largest flow magnitude in the field, or zero for an
// empty field. It is used internally by [FlowToColor] to normalise the colour
// wheel, and is exported because callers often want the same scale.
func (f *FlowField) MaxMagnitude() float64 {
	var maxv float64
	for p := 0; p < f.Rows*f.Cols; p++ {
		u := f.Data[p*2]
		v := f.Data[p*2+1]
		m := math.Hypot(u, v)
		if m > maxv {
			maxv = m
		}
	}
	return maxv
}
