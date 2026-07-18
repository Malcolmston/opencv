package inpaint

import (
	cv "github.com/malcolmston/opencv"
)

// uniformMat returns a rows x cols x channels Mat with every sample set to val.
func uniformMat(rows, cols, channels int, val uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, channels)
	m.SetTo(val)
	return m
}

// rampMat returns a single-channel Mat whose pixel (y, x) holds a*x+b*y+base,
// clamped to [0,255]. Used for linear known-answer tests.
func rampMat(rows, cols, a, b, base int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := a*x + b*y + base
			m.Set(y, x, 0, clampInt255(v))
		}
	}
	return m
}

func clampInt255(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// centerHoleMask returns a rows x cols mask with the rectangle [x0,x0+w) x
// [y0,y0+h) selected.
func centerHoleMask(rows, cols, y0, x0, h, w int) *Mask {
	m := NewMask(rows, cols)
	for y := y0; y < y0+h; y++ {
		for x := x0; x < x0+w; x++ {
			m.Set(y, x, true)
		}
	}
	return m
}

// absU8 returns |int(a)-int(b)|.
func absU8(a, b uint8) int {
	if a > b {
		return int(a) - int(b)
	}
	return int(b) - int(a)
}
