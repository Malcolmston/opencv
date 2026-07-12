package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// GradientPaillouX computes the horizontal (x-direction) image gradient of img
// with the Paillou recursive filter (Philippe Paillou, "Detecting step edges in
// noisy SAR images: a new linear operator", 1997) and returns it as a signed
// [cv.FloatMat].
//
// The Paillou operator generalises Deriche's by giving its recursive kernel a
// damped-oscillatory impulse response exp(−α|x|)·cos(ω x): alpha sets the decay
// (edge sharpness / noise robustness) and omega the oscillation, letting the
// filter be tuned to a preferred edge frequency. The two share the second-order
// poles b1 = 2·e^{-α}·cos ω, b2 = −e^{-2α}. GradientPaillouX differentiates along
// x and smooths along y, so a positive value marks an intensity increase from
// left to right.
//
// img may be 1- or 3-channel; colour is reduced to luma. It panics if alpha is
// non-positive. The result is deterministic. With omega = 0 the poles reduce to
// Deriche's, though the numerator normalisation differs.
func GradientPaillouX(img *cv.Mat, alpha, omega float64) *cv.FloatMat {
	if alpha <= 0 {
		panic("ximgproc: GradientPaillouX requires alpha > 0")
	}
	rows, cols := img.Rows, img.Cols
	g := channelPlane(toGray(img), 0)
	b1, b2 := paillouPoles(alpha, omega)
	applyAlongRows(g, rows, cols, func(line []float64) { paillouDeriv(line, b1, b2) })
	applyAlongCols(g, rows, cols, func(line []float64) { paillouSmooth(line, b1, b2) })
	return floatMatFrom(g, rows, cols)
}

// GradientPaillouY computes the vertical (y-direction) image gradient of img
// with the Paillou recursive filter and returns it as a signed [cv.FloatMat]. It
// smooths along x and differentiates along y, so a positive value marks an
// intensity increase from top to bottom. See [GradientPaillouX] for the meaning
// of alpha and omega. It panics if alpha is non-positive.
func GradientPaillouY(img *cv.Mat, alpha, omega float64) *cv.FloatMat {
	if alpha <= 0 {
		panic("ximgproc: GradientPaillouY requires alpha > 0")
	}
	rows, cols := img.Rows, img.Cols
	g := channelPlane(toGray(img), 0)
	b1, b2 := paillouPoles(alpha, omega)
	applyAlongRows(g, rows, cols, func(line []float64) { paillouSmooth(line, b1, b2) })
	applyAlongCols(g, rows, cols, func(line []float64) { paillouDeriv(line, b1, b2) })
	return floatMatFrom(g, rows, cols)
}

// paillouPoles returns the shared second-order feedback coefficients for the
// damped-oscillator recursion.
func paillouPoles(alpha, omega float64) (b1, b2 float64) {
	a := math.Exp(-alpha)
	b1 = 2 * a * math.Cos(omega)
	b2 = -a * a
	return b1, b2
}

// paillouSmooth applies a DC-normalised bidirectional (cascaded causal then
// anti-causal) 2nd-order recursive smoother with the given poles to a line in
// place. Each pass has unit DC gain, so a flat line is returned unchanged.
func paillouSmooth(line []float64, b1, b2 float64) {
	g := 1 - b1 - b2 // per-pass numerator gives unit DC gain
	y := causalPass(line, g, 0, 0, b1, b2)
	z := anticausalPass(y, g, 0, 0, b1, b2)
	copy(line, z)
}

// paillouDeriv applies the antisymmetric recursive derivative with the given
// poles to a line in place: the difference of a forward-looking and a
// backward-looking unit-gain smoother, which is zero on flat regions and
// positive where the signal rises.
func paillouDeriv(line []float64, b1, b2 float64) {
	g := 1 - b1 - b2
	yc := causalPass(line, g, 0, 0, b1, b2)     // backward-looking
	ya := anticausalPass(line, g, 0, 0, b1, b2) // forward-looking
	for i := range line {
		line[i] = ya[i] - yc[i]
	}
}
