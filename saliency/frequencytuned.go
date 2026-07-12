package saliency

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// StaticSaliencyFrequencyTuned implements the frequency-tuned salient region
// detector of Achanta, Hemami, Estrada & Süsstrunk, "Frequency-tuned Salient
// Region Detection" (CVPR 2009).
//
// The image is converted to CIE L*a*b* colour. Saliency at a pixel is the
// squared Euclidean distance, in Lab, between that pixel's slightly Gaussian-
// smoothed colour and the arithmetic mean Lab colour of the whole image:
//
//	S(x) = ‖ I_μ − I_ωhc(x) ‖²
//
// The whole-image mean removes the very lowest spatial frequencies (large flat
// regions) while the small blur removes the very highest (pixel noise and fine
// texture), leaving well-defined, uniformly highlighted salient objects with
// crisp boundaries. For a single distinct object the mean colour tracks the
// dominant background, so the object stands out strongly.
//
// Construct one with [NewStaticSaliencyFrequencyTuned]. It satisfies
// [StaticSaliency].
type StaticSaliencyFrequencyTuned struct {
	// BlurKSize is the size of the small separable Gaussian applied before the
	// distance is measured. The default is 5.
	BlurKSize int
	// BlurSigma is that Gaussian's standard deviation (<=0 derives it from the
	// kernel size). The default is 1.5.
	BlurSigma float64
}

// NewStaticSaliencyFrequencyTuned returns a detector with a 5×5 pre-blur.
func NewStaticSaliencyFrequencyTuned() *StaticSaliencyFrequencyTuned {
	return &StaticSaliencyFrequencyTuned{BlurKSize: 5, BlurSigma: 1.5}
}

// ComputeSaliency returns the frequency-tuned saliency map of img: a
// single-channel [cv.Mat] the same size as img, normalised to [0,255]. It
// panics if img is nil or empty.
func (s *StaticSaliencyFrequencyTuned) ComputeSaliency(img *cv.Mat) *cv.Mat {
	l, a, b := labPlanes(img)

	ks := s.BlurKSize
	if ks < 1 {
		ks = 5
	}
	lb := gaussianBlurPlane(l, ks, s.BlurSigma)
	ab := gaussianBlurPlane(a, ks, s.BlurSigma)
	bb := gaussianBlurPlane(b, ks, s.BlurSigma)

	lm, am, bm := l.mean(), a.mean(), b.mean()

	out := newPlane(l.rows, l.cols)
	for i := range out.data {
		dl := lb.data[i] - lm
		da := ab.data[i] - am
		db := bb.data[i] - bm
		out.data[i] = math.Sqrt(dl*dl + da*da + db*db)
	}
	return out.normalizedMat()
}
