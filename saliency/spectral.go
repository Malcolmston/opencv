package saliency

import (
	"math"
	"math/cmplx"

	cv "github.com/malcolmston/opencv"
)

// StaticSaliency is implemented by the static (single-image) saliency
// detectors. It mirrors OpenCV's cv::saliency::StaticSaliency: a detector
// consumes one image and produces a single-channel saliency map in which
// brighter samples mark more visually salient locations.
type StaticSaliency interface {
	// ComputeSaliency returns a single-channel saliency map, the same size as
	// img, normalised to the 8-bit range.
	ComputeSaliency(img *cv.Mat) *cv.Mat
}

// StaticSaliencySpectralResidual detects saliency with the spectral-residual
// method of Hou & Zhang, "Saliency Detection: A Spectral Residual Approach"
// (CVPR 2007), the same algorithm as OpenCV's
// cv::saliency::StaticSaliencySpectralResidual.
//
// The image is reduced to grayscale and resampled to a small working size
// (ResizedWidth×ResizedHeight). Its 2-D Fourier transform is split into a
// log-amplitude spectrum and a phase spectrum. The spectral residual — the
// log-amplitude minus its own local average — captures the parts of the
// spectrum that deviate from the smooth, statistically-expected 1/f falloff and
// therefore correspond to novel, salient structure. Recombining the residual
// amplitude with the original phase and inverse-transforming yields a saliency
// map, which is squared, blurred and resized back to the input dimensions.
//
// Construct one with [NewStaticSaliencySpectralResidual]. The zero value is not
// usable; use the constructor so the working size is set.
type StaticSaliencySpectralResidual struct {
	// ResizedWidth and ResizedHeight are the working dimensions the image is
	// resampled to before the transform. They are rounded up to a power of two
	// internally so the radix-2 FFT applies. Smaller sizes emphasise coarse,
	// large-scale saliency; the OpenCV default is 64×64.
	ResizedWidth  int
	ResizedHeight int
}

// NewStaticSaliencySpectralResidual returns a detector configured with the
// OpenCV default 64×64 working resolution.
func NewStaticSaliencySpectralResidual() *StaticSaliencySpectralResidual {
	return &StaticSaliencySpectralResidual{ResizedWidth: 64, ResizedHeight: 64}
}

// spectralEpsilon guards the logarithm of the amplitude spectrum against zeros
// at frequencies with no energy.
const spectralEpsilon = 1e-12

// ComputeSaliency returns the spectral-residual saliency map of img, a
// single-channel [cv.Mat] the same size as img and normalised to [0,255]. It
// panics if img is nil or empty.
func (s *StaticSaliencySpectralResidual) ComputeSaliency(img *cv.Mat) *cv.Mat {
	gray := grayPlane(img)

	w := nextPow2(s.ResizedWidth)
	h := nextPow2(s.ResizedHeight)
	if w < 2 {
		w = 64
	}
	if h < 2 {
		h = 64
	}
	small := resizePlane(gray, h, w)

	// Forward 2-D FFT of the real image.
	field := make([]complex128, h*w)
	for i, v := range small.data {
		field[i] = complex(v, 0)
	}
	fft2D(field, h, w, false)

	// Separate the log-amplitude and phase spectra.
	logAmp := newPlane(h, w)
	phase := make([]float64, h*w)
	for i, c := range field {
		mag := cmplx.Abs(c)
		if mag < spectralEpsilon {
			mag = spectralEpsilon
		}
		logAmp.data[i] = math.Log(mag)
		phase[i] = cmplx.Phase(c)
	}

	// The spectral residual is the log-amplitude minus its 3×3 local average.
	smooth := meanBlur(logAmp, 1)
	for i := range field {
		residual := logAmp.data[i] - smooth.data[i]
		field[i] = cmplx.Rect(math.Exp(residual), phase[i])
	}

	// Inverse transform and take the squared magnitude as raw saliency.
	fft2D(field, h, w, true)
	sal := newPlane(h, w)
	for i, c := range field {
		re, im := real(c), imag(c)
		sal.data[i] = re*re + im*im
	}

	// Smooth, resize back to the input size and normalise.
	blurred := gaussianBlurPlane(sal, 5, 2.5)
	full := resizePlane(blurred, gray.rows, gray.cols)
	return full.normalizedMat()
}
