package freqdomain

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// GaussianPSF returns a normalised size×size point-spread function sampled from
// an isotropic Gaussian of standard deviation sigma, centred in the kernel. The
// samples sum to 1, so convolving with it preserves the mean brightness. It
// panics unless size > 0 and sigma > 0.
func GaussianPSF(size int, sigma float64) *cv.FloatMat {
	if size <= 0 {
		panic("freqdomain: GaussianPSF requires positive size")
	}
	if sigma <= 0 {
		panic("freqdomain: GaussianPSF requires positive sigma")
	}
	out := cv.NewFloatMat(size, size)
	c := float64(size-1) / 2
	denom := 2 * sigma * sigma
	var sum float64
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			dy := float64(y) - c
			dx := float64(x) - c
			v := math.Exp(-(dy*dy + dx*dx) / denom)
			out.Data[y*size+x] = v
			sum += v
		}
	}
	for i := range out.Data {
		out.Data[i] /= sum
	}
	return out
}

// MotionBlurPSF returns a normalised size×size point-spread function modelling
// linear motion blur of the given angle in degrees, measured counter-clockwise
// from the horizontal axis. A one-pixel-wide line of unit weight is drawn
// through the kernel centre and normalised to sum to 1. It panics unless
// size > 0.
func MotionBlurPSF(size int, angle float64) *cv.FloatMat {
	if size <= 0 {
		panic("freqdomain: MotionBlurPSF requires positive size")
	}
	out := cv.NewFloatMat(size, size)
	c := float64(size-1) / 2
	rad := angle * math.Pi / 180
	dx := math.Cos(rad)
	dy := -math.Sin(rad)
	half := float64(size-1) / 2
	steps := size * 2
	for i := 0; i <= steps; i++ {
		t := (float64(i)/float64(steps))*2*half - half
		x := int(math.Round(c + t*dx))
		y := int(math.Round(c + t*dy))
		if x >= 0 && x < size && y >= 0 && y < size {
			out.Data[y*size+x] = 1
		}
	}
	var sum float64
	for _, v := range out.Data {
		sum += v
	}
	if sum == 0 {
		out.Data[int(c)*size+int(c)] = 1
		sum = 1
	}
	for i := range out.Data {
		out.Data[i] /= sum
	}
	return out
}

// PSFToOTF converts a spatial point-spread function into an optical transfer
// function (its FFT) of the requested size. The PSF is zero-padded to rows×cols
// and circularly shifted so its centre lands at the origin, so the resulting
// OTF has no linear-phase component and models circular convolution. It panics
// unless the PSF fits within the requested size.
func PSFToOTF(psf *cv.FloatMat, rows, cols int) *Spectrum {
	if psf.Rows > rows || psf.Cols > cols {
		panic(fmt.Sprintf("freqdomain: PSFToOTF psf %dx%d larger than %dx%d", psf.Rows, psf.Cols, rows, cols))
	}
	padded := make([]float64, rows*cols)
	for y := 0; y < psf.Rows; y++ {
		for x := 0; x < psf.Cols; x++ {
			padded[y*cols+x] = psf.Data[y*psf.Cols+x]
		}
	}
	padded = circularShift2D(rows, cols, padded, -(psf.Rows / 2), -(psf.Cols / 2))
	s := NewSpectrum(rows, cols)
	copy(s.Re, padded)
	return FFT2DComplex(s)
}

// ConvolveFFT returns the circular convolution of a real image with a kernel,
// computed by multiplying their spectra. The kernel is treated as a
// point-spread function centred in its own support. The result has the same
// size as f.
func ConvolveFFT(f *cv.FloatMat, kernel *cv.FloatMat) *cv.FloatMat {
	otf := PSFToOTF(kernel, f.Rows, f.Cols)
	return IFFT2D(FFT2D(f).Mul(otf))
}

// InverseFilter restores a real degraded image by dividing its spectrum by the
// optical transfer function of the point-spread function psf. Frequencies where
// the OTF magnitude is at or below threshold are left unchanged (their input
// value is kept) to avoid amplifying noise by dividing by near-zero. It panics
// on a size mismatch between g and the OTF support. Use a small positive
// threshold, e.g. 0.01.
func InverseFilter(g *cv.FloatMat, psf *cv.FloatMat, threshold float64) *cv.FloatMat {
	G := FFT2D(g)
	H := PSFToOTF(psf, g.Rows, g.Cols)
	F := NewSpectrum(g.Rows, g.Cols)
	for i := range G.Re {
		hr, hi := H.Re[i], H.Im[i]
		mag2 := hr*hr + hi*hi
		if math.Sqrt(mag2) <= threshold {
			F.Re[i] = G.Re[i]
			F.Im[i] = G.Im[i]
			continue
		}
		gr, gi := G.Re[i], G.Im[i]
		// (gr+gi·i)/(hr+hi·i) = (G·conj(H))/|H|²
		F.Re[i] = (gr*hr + gi*hi) / mag2
		F.Im[i] = (gi*hr - gr*hi) / mag2
	}
	return IFFT2D(F)
}

// WienerDeconvolution restores a real degraded image using the Wiener filter
//
//	F̂ = conj(H) / (|H|² + nsr) · G,
//
// where H is the optical transfer function of the point-spread function psf, G
// is the spectrum of the degraded image and nsr is the (constant) noise-to-
// signal power ratio. A larger nsr suppresses noise amplification at the cost of
// sharpness; nsr = 0 reduces to a pseudo-inverse filter. It panics unless
// nsr >= 0. The result has the same size as g.
func WienerDeconvolution(g *cv.FloatMat, psf *cv.FloatMat, nsr float64) *cv.FloatMat {
	if nsr < 0 {
		panic("freqdomain: WienerDeconvolution requires nsr >= 0")
	}
	G := FFT2D(g)
	H := PSFToOTF(psf, g.Rows, g.Cols)
	F := NewSpectrum(g.Rows, g.Cols)
	for i := range G.Re {
		hr, hi := H.Re[i], H.Im[i]
		mag2 := hr*hr + hi*hi
		wr := hr / (mag2 + nsr)
		wi := -hi / (mag2 + nsr)
		gr, gi := G.Re[i], G.Im[i]
		F.Re[i] = gr*wr - gi*wi
		F.Im[i] = gr*wi + gi*wr
	}
	return IFFT2D(F)
}
