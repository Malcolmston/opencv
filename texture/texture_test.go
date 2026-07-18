package texture_test

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// makeGray builds a single-channel cv.Mat from a 2-D slice of gray values.
func makeGray(vals [][]uint8) *cv.Mat {
	rows := len(vals)
	cols := len(vals[0])
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Data[y*cols+x] = vals[y][x]
		}
	}
	return m
}

// fill builds a rows x cols single-channel Mat with every pixel set to v.
func fill(rows, cols int, v uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for i := range m.Data {
		m.Data[i] = v
	}
	return m
}

// approx reports whether a and b are within tol.
func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }
