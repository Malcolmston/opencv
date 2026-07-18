package saliency2

import (
	"math"
	"math/cmplx"

	cv "github.com/malcolmston/opencv"
)

// StaticSaliencySpectralResidual detects saliency with the spectral-residual
// method of Hou & Zhang, "Saliency Detection: A Spectral Residual Approach"
// (CVPR 2007), the same algorithm as OpenCV's
// cv::saliency::StaticSaliencySpectralResidual.
//
// The image is reduced to grayscale and resampled to a small working size. Its
// 2-D Fourier transform is split into a log-amplitude spectrum and a phase
// spectrum. The spectral residual — the log-amplitude minus its own 3x3 local
// average — isolates the parts of the spectrum that deviate from the smooth,
// statistically expected 1/f falloff and therefore correspond to novel, salient
// structure. Recombining the residual amplitude with the original phase and
// inverse-transforming yields the saliency map, which is squared, blurred and
// resized back to the input dimensions.
//
// Construct one with [NewStaticSaliencySpectralResidual]; the zero value is not
// usable.
type StaticSaliencySpectralResidual struct {
	// ResizedWidth and ResizedHeight are the working dimensions the image is
	// resampled to before the transform. They are rounded up to a power of two
	// internally so the radix-2 FFT applies. Smaller sizes emphasise coarse,
	// large-scale saliency; the OpenCV default is 64x64.
	ResizedWidth, ResizedHeight int
}

// NewStaticSaliencySpectralResidual returns a detector configured with the
// OpenCV default 64x64 working resolution.
func NewStaticSaliencySpectralResidual() *StaticSaliencySpectralResidual {
	return &StaticSaliencySpectralResidual{ResizedWidth: 64, ResizedHeight: 64}
}

// spectralEpsilon guards the logarithm of the amplitude spectrum against zeros
// at frequencies with no energy.
const spectralEpsilon = 1e-12

// ComputeSaliencyMap returns the spectral-residual saliency of img as a
// [SaliencyMap] the same size as img. It panics if img is nil or empty.
func (s *StaticSaliencySpectralResidual) ComputeSaliencyMap(img *cv.Mat) *SaliencyMap {
	gray := saliency2GrayFloat(img)

	w := saliency2NextPow2(s.ResizedWidth)
	h := saliency2NextPow2(s.ResizedHeight)
	if w < 2 {
		w = 64
	}
	if h < 2 {
		h = 64
	}
	small := saliency2ResizeMap(gray, h, w)

	field := make([]complex128, h*w)
	for i, v := range small.Data {
		field[i] = complex(v, 0)
	}
	saliency2FFT2D(field, h, w, false)

	logAmp := NewSaliencyMap(h, w)
	phase := make([]float64, h*w)
	for i, c := range field {
		mag := cmplx.Abs(c)
		if mag < spectralEpsilon {
			mag = spectralEpsilon
		}
		logAmp.Data[i] = math.Log(mag)
		phase[i] = cmplx.Phase(c)
	}

	// Spectral residual: log-amplitude minus its 3x3 local mean.
	smooth := saliency2BoxBlurMap(logAmp, 1)
	for i := range field {
		residual := logAmp.Data[i] - smooth.Data[i]
		field[i] = cmplx.Rect(math.Exp(residual), phase[i])
	}

	saliency2FFT2D(field, h, w, true)
	sal := NewSaliencyMap(h, w)
	for i, c := range field {
		re, im := real(c), imag(c)
		sal.Data[i] = re*re + im*im
	}

	blurred := saliency2GaussianBlurMap(sal, 5, 2.5)
	return saliency2ResizeMap(blurred, gray.Rows, gray.Cols)
}

// ComputeSaliency returns the spectral-residual saliency map of img as an 8-bit
// single-channel [cv.Mat]. It satisfies the [StaticSaliency] interface.
func (s *StaticSaliencySpectralResidual) ComputeSaliency(img *cv.Mat) *cv.Mat {
	return s.ComputeSaliencyMap(img).ToMat()
}

// SpectralResidualSaliency is a convenience wrapper that computes the
// spectral-residual saliency map of img with the default detector settings.
func SpectralResidualSaliency(img *cv.Mat) *cv.Mat {
	return NewStaticSaliencySpectralResidual().ComputeSaliency(img)
}
