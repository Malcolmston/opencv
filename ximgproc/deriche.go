package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// GradientDericheX computes the horizontal (x-direction) image gradient of img
// with Deriche's recursive filter (Deriche, "Using Canny's criteria to derive a
// recursively implemented optimal edge detector", 1987) and returns it as a
// signed [cv.FloatMat].
//
// The Deriche operator is the infinite-support, recursively implementable
// optimum of Canny's edge criteria: a first-derivative-of-smoothing kernel whose
// sharpness is set by alpha (larger alpha ⇒ narrower kernel, less smoothing;
// alpha ≈ 1 is a good default). GradientDericheX differentiates along x with the
// recursive derivative filter and smooths along y with the matching recursive
// blur, so a positive value marks an intensity increase from left to right.
//
// img may be 1- or 3-channel; colour is reduced to luma. It panics if alpha is
// non-positive. The result is deterministic and independent of image scanning
// order.
func GradientDericheX(img *cv.Mat, alpha float64) *cv.FloatMat {
	if alpha <= 0 {
		panic("ximgproc: GradientDericheX requires alpha > 0")
	}
	rows, cols := img.Rows, img.Cols
	g := channelPlane(toGray(img), 0)
	// Derivative along x (rows), then smooth along y (columns).
	applyAlongRows(g, rows, cols, func(line []float64) { dericheDeriv(line, alpha) })
	applyAlongCols(g, rows, cols, func(line []float64) { dericheSmooth(line, alpha) })
	return floatMatFrom(g, rows, cols)
}

// GradientDericheY computes the vertical (y-direction) image gradient of img
// with Deriche's recursive filter and returns it as a signed [cv.FloatMat]. It
// smooths along x and differentiates along y, so a positive value marks an
// intensity increase from top to bottom. See [GradientDericheX] for the meaning
// of alpha. It panics if alpha is non-positive.
func GradientDericheY(img *cv.Mat, alpha float64) *cv.FloatMat {
	if alpha <= 0 {
		panic("ximgproc: GradientDericheY requires alpha > 0")
	}
	rows, cols := img.Rows, img.Cols
	g := channelPlane(toGray(img), 0)
	applyAlongRows(g, rows, cols, func(line []float64) { dericheSmooth(line, alpha) })
	applyAlongCols(g, rows, cols, func(line []float64) { dericheDeriv(line, alpha) })
	return floatMatFrom(g, rows, cols)
}

// dericheSmooth applies the Deriche recursive smoothing (blur) filter to a line
// in place. It sums a causal and an anti-causal 2nd-order pass sharing the poles
// b1 = 2e^{-α}, b2 = −e^{-2α}; the numerator constant k normalises the DC gain to
// 1, so a flat line is returned unchanged.
func dericheSmooth(line []float64, alpha float64) {
	e := math.Exp(-alpha)
	e2 := math.Exp(-2 * alpha)
	k := (1 - e) * (1 - e) / (1 + 2*alpha*e - e2)
	b1 := 2 * e
	b2 := -e2
	y1 := causalPass(line, k, k*e*(alpha-1), 0, b1, b2)
	y2 := anticausalPass(line, 0, k*e*(alpha+1), -k*e2, b1, b2)
	for i := range line {
		line[i] = y1[i] + y2[i]
	}
}

// dericheDeriv applies the Deriche recursive first-derivative filter to a line
// in place. A causal pass responds to the past sample and an anti-causal pass to
// the future sample with opposite sign, so the antisymmetric result is positive
// where the signal rises and zero on flat regions.
func dericheDeriv(line []float64, alpha float64) {
	e := math.Exp(-alpha)
	e2 := math.Exp(-2 * alpha)
	b1 := 2 * e
	b2 := -e2
	kd := -(1 - e) * (1 - e) // steady-state derivative gain of −1 per side
	y1 := causalPass(line, 0, kd, 0, b1, b2)
	y2 := anticausalPass(line, 0, -kd, 0, b1, b2)
	for i := range line {
		line[i] = y1[i] + y2[i]
	}
}

// applyAlongRows runs f on each row of a row-major rows×cols plane in place.
func applyAlongRows(p []float64, rows, cols int, f func([]float64)) {
	line := make([]float64, cols)
	for y := 0; y < rows; y++ {
		base := y * cols
		copy(line, p[base:base+cols])
		f(line)
		copy(p[base:base+cols], line)
	}
}

// applyAlongCols runs f on each column of a row-major rows×cols plane in place.
func applyAlongCols(p []float64, rows, cols int, f func([]float64)) {
	line := make([]float64, rows)
	for x := 0; x < cols; x++ {
		for y := 0; y < rows; y++ {
			line[y] = p[y*cols+x]
		}
		f(line)
		for y := 0; y < rows; y++ {
			p[y*cols+x] = line[y]
		}
	}
}

// floatMatFrom wraps a row-major plane in a cv.FloatMat.
func floatMatFrom(p []float64, rows, cols int) *cv.FloatMat {
	out := cv.NewFloatMat(rows, cols)
	copy(out.Data, p)
	return out
}
