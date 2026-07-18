package textdet

import (
	cv "github.com/malcolmston/opencv"
)

// newGray returns a rows x cols single-channel Mat filled with fill.
func newGray(rows, cols int, fill uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	m.SetTo(fill)
	return m
}

// paintRect fills the rectangle [x0,x0+w) x [y0,y0+h) of m with value v.
func paintRect(m *cv.Mat, x0, y0, w, h int, v uint8) {
	for y := y0; y < y0+h; y++ {
		for x := x0; x < x0+w; x++ {
			m.Data[y*m.Cols+x] = v
		}
	}
}
