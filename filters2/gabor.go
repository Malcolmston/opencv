package filters2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// GaborFilter convolves a single-channel image with the real (even) Gabor
// kernel described by p at the given odd kernel size and returns the signed
// response as a [FloatImage]. It panics on multi-channel or empty input or an
// invalid kernel size or parameters.
func GaborFilter(src *cv.Mat, size int, p GaborParams) *FloatImage {
	requireGray(src, "GaborFilter")
	return ConvolveMat(src, GaborKernel(size, p))
}

// GaborMagnitude convolves a single-channel image with the even (cosine) and
// odd (sine) Gabor kernels described by p and returns the magnitude of the
// resulting complex response, sqrt(even^2+odd^2), as a [FloatImage]. This
// quadrature energy is phase-invariant and is the usual feature extracted from
// a Gabor filter. It panics on multi-channel or empty input or invalid
// parameters.
func GaborMagnitude(src *cv.Mat, size int, p GaborParams) *FloatImage {
	requireGray(src, "GaborMagnitude")
	if p.Sigma <= 0 || p.Lambda <= 0 {
		panic("filters2: GaborMagnitude requires positive Sigma and Lambda")
	}
	requireOddPositive(size, "GaborMagnitude")
	even := gaborKernelPhase(size, p, p.Psi, math.Cos)
	odd := gaborKernelPhase(size, p, p.Psi, math.Sin)
	fi := MatToFloatImage(src)
	re := Convolve(fi, even)
	im := Convolve(fi, odd)
	return Magnitude(re, im)
}

// GaborBank builds a bank of [GaborParams] sweeping numOrientations equally
// spaced orientations over [0,pi) for each supplied wavelength in lambdas,
// sharing the given sigma, gamma and psi. The result is ordered wavelength-major
// then orientation. It panics on a non-positive orientation count or an empty
// wavelength list.
func GaborBank(numOrientations int, lambdas []float64, sigma, gamma, psi float64) []GaborParams {
	if numOrientations < 1 {
		panic("filters2: GaborBank requires numOrientations >= 1")
	}
	if len(lambdas) == 0 {
		panic("filters2: GaborBank requires at least one wavelength")
	}
	bank := make([]GaborParams, 0, numOrientations*len(lambdas))
	for _, lambda := range lambdas {
		for o := 0; o < numOrientations; o++ {
			theta := math.Pi * float64(o) / float64(numOrientations)
			bank = append(bank, GaborParams{
				Sigma:  sigma,
				Theta:  theta,
				Lambda: lambda,
				Gamma:  gamma,
				Psi:    psi,
			})
		}
	}
	return bank
}

// GaborBankResponse applies every filter in a bank to a single-channel image
// and returns the per-filter magnitude responses, in bank order. It panics on
// multi-channel or empty input, an empty bank, or an invalid kernel size.
func GaborBankResponse(src *cv.Mat, size int, bank []GaborParams) []*FloatImage {
	requireGray(src, "GaborBankResponse")
	if len(bank) == 0 {
		panic("filters2: GaborBankResponse requires a non-empty bank")
	}
	out := make([]*FloatImage, len(bank))
	for i, p := range bank {
		out[i] = GaborMagnitude(src, size, p)
	}
	return out
}

// GaborEnergy applies every filter in a bank to a single-channel image and
// returns, per pixel, the maximum magnitude response across the bank, rescaled
// to a viewable 8-bit [cv.Mat]. This is a simple rotation- and scale-tolerant
// texture-energy feature. It panics on multi-channel or empty input, an empty
// bank, or an invalid kernel size.
func GaborEnergy(src *cv.Mat, size int, bank []GaborParams) *cv.Mat {
	responses := GaborBankResponse(src, size, bank)
	rows, cols := src.Rows, src.Cols
	energy := NewFloatImage(rows, cols)
	for _, r := range responses {
		for i, v := range r.Data {
			if v > energy.Data[i] {
				energy.Data[i] = v
			}
		}
	}
	return energy.Normalize()
}
