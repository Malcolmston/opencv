package cudalegacy

import (
	cv "github.com/malcolmston/opencv"
)

// Flow is a dense, per-pixel motion field: for every pixel it stores a
// horizontal (U) and vertical (V) displacement, both as [cv.FloatMat] planes the
// same size as the frames they describe. Positive U points right (increasing
// column) and positive V points down (increasing row), matching the image
// coordinate convention of the root package.
//
// OpenCV's cudalegacy optical-flow entry points write two CV_32FC1 GpuMats
// (velx, vely); the 8-bit root Mat cannot hold signed sub-pixel displacements,
// so this package returns them as float planes. The zero value is not usable —
// construct with [NewFlow].
type Flow struct {
	// U holds horizontal displacements; V holds vertical displacements. Both
	// have identical dimensions.
	U, V *cv.FloatMat
}

// NewFlow allocates a zero-filled Flow of the given size. It panics on
// non-positive dimensions.
func NewFlow(rows, cols int) *Flow {
	if rows <= 0 || cols <= 0 {
		panic("cudalegacy: NewFlow requires positive dimensions")
	}
	return &Flow{U: cv.NewFloatMat(rows, cols), V: cv.NewFloatMat(rows, cols)}
}

// Rows returns the field height.
func (f *Flow) Rows() int { return f.U.Rows }

// Cols returns the field width.
func (f *Flow) Cols() int { return f.U.Cols }

// At returns the (u, v) displacement stored at row y, column x.
func (f *Flow) At(y, x int) (u, v float64) {
	i := y*f.U.Cols + x
	return f.U.Data[i], f.V.Data[i]
}

// set stores the (u, v) displacement at row y, column x.
func (f *Flow) set(y, x int, u, v float64) {
	i := y*f.U.Cols + x
	f.U.Data[i] = u
	f.V.Data[i] = v
}
