package freqdomain

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// HomomorphicEmphasis returns a centred high-frequency-emphasis transfer
// function of the form
//
//	H(D) = (gammaHigh − gammaLow)·(1 − exp(−c·D²/cutoff²)) + gammaLow
//
// used for homomorphic filtering. With gammaLow < 1 < gammaHigh it attenuates
// low frequencies (illumination) while boosting high frequencies (reflectance
// and detail); c controls the sharpness of the transition. It panics unless
// cutoff > 0.
func HomomorphicEmphasis(rows, cols int, gammaLow, gammaHigh, cutoff, c float64) *cv.FloatMat {
	requirePositiveCutoff(cutoff, "HomomorphicEmphasis")
	d02 := cutoff * cutoff
	return buildFilter(rows, cols, func(d float64) float64 {
		return (gammaHigh-gammaLow)*(1-math.Exp(-c*(d*d)/d02)) + gammaLow
	})
}

// HomomorphicFilterFloat applies homomorphic filtering to a real image: it takes
// the natural log of (1 + pixel), filters it in the frequency domain with the
// emphasis function from [HomomorphicEmphasis], exponentiates the result and
// subtracts 1. This compresses the dynamic range of illumination while
// enhancing local contrast. The returned float image has the same size as f. It
// panics unless cutoff > 0.
func HomomorphicFilterFloat(f *cv.FloatMat, gammaLow, gammaHigh, cutoff, c float64) *cv.FloatMat {
	ln := cv.NewFloatMat(f.Rows, f.Cols)
	for i, v := range f.Data {
		ln.Data[i] = math.Log(1 + v)
	}
	filter := HomomorphicEmphasis(f.Rows, f.Cols, gammaLow, gammaHigh, cutoff, c)
	filtered := ApplyFilter(ln, filter)
	out := cv.NewFloatMat(f.Rows, f.Cols)
	for i, v := range filtered.Data {
		out.Data[i] = math.Expm1(v)
	}
	return out
}

// HomomorphicFilter applies homomorphic filtering to an 8-bit image and returns
// an 8-bit image. It converts the input to float with [MatToFloat], runs
// [HomomorphicFilterFloat] and rescales the result to [0,255]. Typical
// parameters are gammaLow ≈ 0.5, gammaHigh ≈ 2.0, a small cutoff and c ≈ 1. It
// panics unless cutoff > 0.
func HomomorphicFilter(m *cv.Mat, gammaLow, gammaHigh, cutoff, c float64) *cv.Mat {
	f := MatToFloat(m)
	return FloatToMatScaled(HomomorphicFilterFloat(f, gammaLow, gammaHigh, cutoff, c))
}
