package freqdomain

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// CrossPowerSpectrum returns the normalised cross-power spectrum of two spectra,
//
//	R = (A · conj(B)) / |A · conj(B)|,
//
// whose inverse transform is the phase-correlation surface. Elements with
// near-zero magnitude are left at zero. It panics on a size mismatch.
func CrossPowerSpectrum(a, b *Spectrum) *Spectrum {
	requireSameSpectrum(a, b, "CrossPowerSpectrum")
	out := NewSpectrum(a.Rows, a.Cols)
	for i := range a.Re {
		ar, ai := a.Re[i], a.Im[i]
		br, bi := b.Re[i], -b.Im[i] // conj(B)
		cr := ar*br - ai*bi
		ci := ar*bi + ai*br
		mag := math.Hypot(cr, ci)
		if mag > 1e-12 {
			out.Re[i] = cr / mag
			out.Im[i] = ci / mag
		}
	}
	return out
}

// PhaseCorrelation estimates the circular translational shift between two real
// images of equal size using the phase-correlation method: it forms the
// normalised cross-power spectrum, inverse-transforms it and locates the peak of
// the resulting correlation surface. It returns the integer shift (dx, dy) that
// maps image a onto image b together with the peak response, the peak height
// divided by the total absolute response, in (0,1]. Shifts larger than half a
// dimension are reported as negative. It panics on a size mismatch.
func PhaseCorrelation(a, b *cv.FloatMat) (dx, dy int, response float64) {
	fx, fy, r := phaseCorrPeak(a, b)
	return fx, fy, r
}

// RegisterTranslation returns only the integer (dx, dy) shift that best aligns
// image a onto image b, discarding the response value. It is a thin convenience
// wrapper over [PhaseCorrelation]. It panics on a size mismatch.
func RegisterTranslation(a, b *cv.FloatMat) (dx, dy int) {
	x, y, _ := phaseCorrPeak(a, b)
	return x, y
}

// phaseCorrPeak performs the core phase-correlation computation, returning the
// wrapped integer peak location and the normalised response.
func phaseCorrPeak(a, b *cv.FloatMat) (dx, dy int, response float64) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("freqdomain: PhaseCorrelation shape mismatch")
	}
	rows, cols := a.Rows, a.Cols
	A := FFT2D(a)
	B := FFT2D(b)
	R := CrossPowerSpectrum(B, A) // peaks at the shift mapping a onto b
	corr := IFFT2D(R)
	peak := math.Inf(-1)
	var total float64
	px, py := 0, 0
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := corr.Data[y*cols+x]
			total += math.Abs(v)
			if v > peak {
				peak = v
				px, py = x, y
			}
		}
	}
	if px > cols/2 {
		px -= cols
	}
	if py > rows/2 {
		py -= rows
	}
	if total > 0 {
		response = peak / total
		if response > 1 {
			response = 1
		}
	}
	return px, py, response
}

// ApplyWindow multiplies a real image element-wise by a window of matching size
// (such as the Hann window from cv.CreateHanningWindow) and returns a new image.
// Windowing before [PhaseCorrelation] suppresses the spurious spectral leakage
// caused by the non-periodic image borders. It panics on a size mismatch.
func ApplyWindow(f *cv.FloatMat, window *cv.FloatMat) *cv.FloatMat {
	if f.Rows != window.Rows || f.Cols != window.Cols {
		panic("freqdomain: ApplyWindow shape mismatch")
	}
	out := cv.NewFloatMat(f.Rows, f.Cols)
	for i := range f.Data {
		out.Data[i] = f.Data[i] * window.Data[i]
	}
	return out
}
