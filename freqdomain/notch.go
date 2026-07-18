package freqdomain

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Notch describes a single conjugate notch centred at frequency offset (U, V)
// relative to the spectrum centre. D0 is the notch radius (the cutoff of the
// underlying Butterworth response) and Order controls the sharpness of the
// transition. Because a real image has a conjugate-symmetric spectrum, each
// Notch suppresses both (U, V) and its mirror (−U, −V).
type Notch struct {
	// U is the vertical (row) frequency offset from the centre.
	U float64
	// V is the horizontal (column) frequency offset from the centre.
	V float64
	// D0 is the notch radius (Butterworth cutoff), which must be positive.
	D0 float64
	// Order is the Butterworth order, which must be positive.
	Order int
}

// NotchReject returns a centred Butterworth notch-reject transfer function of
// size rows×cols that suppresses each listed [Notch] and its conjugate mirror.
// The overall response is the product, over all notches, of a high-pass
// Butterworth response centred at (U, V) and one centred at (−U, −V); values
// near a notch centre approach 0 while frequencies far from every notch pass
// with gain near 1. It panics if any notch has D0 <= 0 or Order <= 0.
func NotchReject(rows, cols int, notches []Notch) *cv.FloatMat {
	for _, nk := range notches {
		requirePositiveCutoff(nk.D0, "NotchReject")
		requireOrder(nk.Order, "NotchReject")
	}
	cy := float64(rows / 2)
	cx := float64(cols / 2)
	out := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			h := 1.0
			for _, nk := range notches {
				n2 := 2.0 * float64(nk.Order)
				dyP := float64(y) - cy - nk.U
				dxP := float64(x) - cx - nk.V
				dk := math.Hypot(dyP, dxP)
				dyM := float64(y) - cy + nk.U
				dxM := float64(x) - cx + nk.V
				dmk := math.Hypot(dyM, dxM)
				h *= butterHighResponse(dk, nk.D0, n2)
				h *= butterHighResponse(dmk, nk.D0, n2)
			}
			out.Data[y*cols+x] = h
		}
	}
	return out
}

// NotchPass returns a centred Butterworth notch-pass transfer function, the
// complement of [NotchReject], which keeps only the frequencies near the listed
// notches. It panics if any notch has D0 <= 0 or Order <= 0.
func NotchPass(rows, cols int, notches []Notch) *cv.FloatMat {
	return complementFilter(NotchReject(rows, cols, notches))
}

// butterHighResponse evaluates a Butterworth high-pass response of cutoff d0 and
// exponent n2 at distance d, returning 0 at d==0.
func butterHighResponse(d, d0, n2 float64) float64 {
	if d == 0 {
		return 0
	}
	return 1 / (1 + math.Pow(d0/d, n2))
}
