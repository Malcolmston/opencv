package stitch

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// clampByte rounds v to the nearest integer and clamps it to the [0,255] range
// of a uint8 sample.
func clampByte(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// solveLinear solves the dense linear system a·x = b for x using Gaussian
// elimination with partial pivoting. a is n×n given row-major as a[][]; b has
// length n. It returns the solution and true, or nil and false if the matrix is
// singular. The inputs are copied and left unmodified.
func solveLinear(a [][]float64, b []float64) ([]float64, bool) {
	n := len(b)
	m := make([][]float64, n)
	for i := 0; i < n; i++ {
		m[i] = make([]float64, n+1)
		copy(m[i], a[i])
		m[i][n] = b[i]
	}
	for col := 0; col < n; col++ {
		pivot := col
		best := math.Abs(m[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(m[r][col]); v > best {
				best = v
				pivot = r
			}
		}
		if best < 1e-15 {
			return nil, false
		}
		m[col], m[pivot] = m[pivot], m[col]
		inv := 1 / m[col][col]
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := m[r][col] * inv
			if f == 0 {
				continue
			}
			for c := col; c <= n; c++ {
				m[r][c] -= f * m[col][c]
			}
		}
	}
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = m[i][n] / m[i][i]
	}
	return x, true
}

// sampleBilinear returns the bilinearly-interpolated value of channel c at the
// continuous source location (fx, fy) and reports whether that location lies
// within the image. Neighbour indices used for interpolation are clamped to the
// image edge, but the covered flag reflects the true position so warping can
// leave uncovered canvas pixels transparent.
func sampleBilinear(img *cv.Mat, fx, fy float64, c int) (value float64, covered bool) {
	if fx < 0 || fy < 0 || fx > float64(img.Cols-1) || fy > float64(img.Rows-1) {
		return 0, false
	}
	x0 := int(math.Floor(fx))
	y0 := int(math.Floor(fy))
	x1 := x0 + 1
	y1 := y0 + 1
	ax := fx - float64(x0)
	ay := fy - float64(y0)
	if x1 > img.Cols-1 {
		x1 = img.Cols - 1
	}
	if y1 > img.Rows-1 {
		y1 = img.Rows - 1
	}
	ch := img.Channels
	i00 := (y0*img.Cols + x0) * ch
	i01 := (y0*img.Cols + x1) * ch
	i10 := (y1*img.Cols + x0) * ch
	i11 := (y1*img.Cols + x1) * ch
	v00 := float64(img.Data[i00+c])
	v01 := float64(img.Data[i01+c])
	v10 := float64(img.Data[i10+c])
	v11 := float64(img.Data[i11+c])
	top := v00*(1-ax) + v01*ax
	bot := v10*(1-ax) + v11*ax
	return top*(1-ay) + bot*ay, true
}
