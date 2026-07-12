package intensity

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// CurvePoint is a single control point of a [ToneCurve], mapping input intensity
// In to output intensity Out. Both coordinates are expressed on the 0..255
// intensity scale.
type CurvePoint struct {
	// In is the input intensity of the control point, in [0,255].
	In float64
	// Out is the desired output intensity at In, in [0,255].
	Out float64
}

// validateCurve panics unless points forms a valid strictly-increasing set of at
// least two control points within [0,255].
func validateCurve(points []CurvePoint, name string) {
	if len(points) < 2 {
		panic("intensity: " + name + " requires at least two control points")
	}
	for i, p := range points {
		if p.In < 0 || p.In > 255 || p.Out < 0 || p.Out > 255 {
			panic(fmt.Sprintf("intensity: %s control point %d out of range: %+v", name, i, p))
		}
		if i > 0 && !(p.In > points[i-1].In) {
			panic(fmt.Sprintf("intensity: %s control points must have strictly increasing In, violated at %d", name, i))
		}
	}
}

// ToneCurveLUT builds a 256-entry lookup table by fitting a natural cubic spline
// through points (which must be sorted by In, strictly increasing, each within
// [0,255]) and sampling it at every integer input 0..255. Because a natural
// cubic spline interpolates its knots exactly, the table reproduces each control
// point: table[round(In)] == round(Out) whenever In is integral. Inputs below
// the first control point hold the first Out, and inputs above the last hold the
// last Out, so the curve is defined across the whole range. It panics if points
// is invalid.
//
// This is the classic "curves" adjustment of photo editors: a gentle S through
// (0,0),(64,48),(192,208),(255,255) boosts contrast, while lifting the lower
// control points raises shadows.
func ToneCurveLUT(points []CurvePoint) []uint8 {
	validateCurve(points, "ToneCurveLUT")
	m := len(points) - 1

	x := make([]float64, len(points))
	y := make([]float64, len(points))
	for i, p := range points {
		x[i] = p.In
		y[i] = p.Out
	}

	// Solve for the spline second derivatives (M) with natural boundaries
	// M[0] = M[m] = 0 using the standard tridiagonal (Thomas) elimination.
	h := make([]float64, m)
	for i := 0; i < m; i++ {
		h[i] = x[i+1] - x[i]
	}
	mm := make([]float64, len(points)) // second derivatives
	if m >= 2 {
		lower := make([]float64, m) // sub-diagonal
		diag := make([]float64, m)  // main diagonal (indices 1..m-1 used)
		upper := make([]float64, m) // super-diagonal
		rhs := make([]float64, m)
		for i := 1; i < m; i++ {
			lower[i] = h[i-1]
			diag[i] = 2 * (h[i-1] + h[i])
			upper[i] = h[i]
			rhs[i] = 6 * ((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])
		}
		// Forward elimination.
		for i := 2; i < m; i++ {
			w := lower[i] / diag[i-1]
			diag[i] -= w * upper[i-1]
			rhs[i] -= w * rhs[i-1]
		}
		// Back substitution into mm[1..m-1]; mm[0] = mm[m] = 0.
		mm[m-1] = rhs[m-1] / diag[m-1]
		for i := m - 2; i >= 1; i-- {
			mm[i] = (rhs[i] - upper[i]*mm[i+1]) / diag[i]
		}
	}

	lut := make([]uint8, 256)
	seg := 0
	for i := 0; i < 256; i++ {
		v := float64(i)
		switch {
		case v <= x[0]:
			lut[i] = clampToUint8(y[0] + 0.5)
			continue
		case v >= x[m]:
			lut[i] = clampToUint8(y[m] + 0.5)
			continue
		}
		for seg < m-1 && v > x[seg+1] {
			seg++
		}
		for seg > 0 && v < x[seg] {
			seg--
		}
		hi := h[seg]
		a := x[seg+1] - v
		b := v - x[seg]
		val := mm[seg]*a*a*a/(6*hi) + mm[seg+1]*b*b*b/(6*hi) +
			(y[seg]/hi-mm[seg]*hi/6)*a + (y[seg+1]/hi-mm[seg+1]*hi/6)*b
		lut[i] = clampToUint8(val + 0.5)
	}
	return lut
}

// ToneCurve remaps every channel of img through the natural cubic spline defined
// by points and returns a new [cv.Mat]. It is the pixel-level companion to
// [ToneCurveLUT]; the same table is applied to every channel so a colour image
// is retoned without a hue shift. It panics on an empty image or an invalid
// control set (fewer than two points, out-of-range coordinates, or an In
// sequence that is not strictly increasing).
func ToneCurve(img *cv.Mat, points []CurvePoint) *cv.Mat {
	requireImage(img, "ToneCurve")
	return applyLUT(img, ToneCurveLUT(points))
}
