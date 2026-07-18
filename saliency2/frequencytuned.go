package saliency2

import cv "github.com/malcolmston/opencv"

// StaticSaliencyFrequencyTuned detects saliency with the frequency-tuned model
// of Achanta, Hemami, Estrada & Susstrunk, "Frequency-tuned Salient Region
// Detection" (CVPR 2009).
//
// The image is converted to CIE L*a*b* colour. Each channel is smoothed with a
// small Gaussian to remove high-frequency texture, and the saliency of a pixel
// is the squared Euclidean distance in L*a*b* between its smoothed colour and
// the whole-image mean colour. Regions whose colour departs from the global
// average — the defining property of a salient object against its background —
// score highly, at full input resolution and with well-defined boundaries.
//
// Construct one with [NewStaticSaliencyFrequencyTuned]; the zero value uses a
// 1-pixel-sigma smoothing.
type StaticSaliencyFrequencyTuned struct {
	// Ksize is the Gaussian pre-smoothing kernel size (default 3).
	Ksize int
	// Sigma is the Gaussian pre-smoothing standard deviation (default 1),
	// applied to each L*a*b* channel.
	Sigma float64
}

// NewStaticSaliencyFrequencyTuned returns a detector with a 3x3, sigma-1
// smoothing kernel.
func NewStaticSaliencyFrequencyTuned() *StaticSaliencyFrequencyTuned {
	return &StaticSaliencyFrequencyTuned{Ksize: 3, Sigma: 1.0}
}

// ComputeSaliencyMap returns the frequency-tuned saliency of img as a
// [SaliencyMap] the same size as img. It panics if img is nil or empty.
func (s *StaticSaliencyFrequencyTuned) ComputeSaliencyMap(img *cv.Mat) *SaliencyMap {
	l, a, b := saliency2LabFloat(img)

	ksize := s.Ksize
	if ksize < 1 {
		ksize = 3
	}
	sigma := s.Sigma
	if sigma <= 0 {
		sigma = 1.0
	}
	lb := saliency2GaussianBlurMap(l, ksize, sigma)
	ab := saliency2GaussianBlurMap(a, ksize, sigma)
	bb := saliency2GaussianBlurMap(b, ksize, sigma)

	mL := l.Mean()
	mA := a.Mean()
	mB := b.Mean()

	out := NewSaliencyMap(l.Rows, l.Cols)
	for i := range out.Data {
		dl := lb.Data[i] - mL
		da := ab.Data[i] - mA
		db := bb.Data[i] - mB
		out.Data[i] = dl*dl + da*da + db*db
	}
	return out
}

// ComputeSaliency returns the frequency-tuned saliency map of img as an 8-bit
// single-channel [cv.Mat]. It satisfies the [StaticSaliency] interface.
func (s *StaticSaliencyFrequencyTuned) ComputeSaliency(img *cv.Mat) *cv.Mat {
	return s.ComputeSaliencyMap(img).ToMat()
}

// FrequencyTunedSaliency is a convenience wrapper that computes the
// frequency-tuned saliency map of img with the default detector settings.
func FrequencyTunedSaliency(img *cv.Mat) *cv.Mat {
	return NewStaticSaliencyFrequencyTuned().ComputeSaliency(img)
}
