package saliency

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// StaticSaliencyFineGrained detects saliency with a multi-scale center-surround
// scheme in the spirit of Montabone & Soto, "Human detection using a mobile
// platform and novel features derived from a visual saliency mechanism" (Image
// and Vision Computing 2010), the algorithm behind OpenCV's
// cv::saliency::StaticSaliencyFineGrained.
//
// The image is reduced to grayscale, and at each of several octave scales the
// absolute difference between every pixel and the mean of the surrounding
// window (an on/off center-surround response) is measured with the help of a
// summed-area table. Small scales react to fine detail while large scales fill
// in the interior of sizeable salient regions; the per-scale responses are
// individually normalised and averaged, so a bright object on a flat background
// lights up as a whole rather than only along its edges.
//
// Construct one with [NewStaticSaliencyFineGrained].
type StaticSaliencyFineGrained struct {
	// Scales is the number of center-surround octaves. The surround window
	// radius doubles each octave (1, 2, 4, … pixels). The OpenCV default is 6.
	Scales int
}

// NewStaticSaliencyFineGrained returns a detector configured with the OpenCV
// default of six scales.
func NewStaticSaliencyFineGrained() *StaticSaliencyFineGrained {
	return &StaticSaliencyFineGrained{Scales: 6}
}

// ComputeSaliency returns the fine-grained saliency map of img, a
// single-channel [cv.Mat] the same size as img and normalised to [0,255]. It
// panics if img is nil or empty.
func (f *StaticSaliencyFineGrained) ComputeSaliency(img *cv.Mat) *cv.Mat {
	gray := grayPlane(img)
	n := f.Scales
	if n <= 0 {
		n = 6
	}

	sat := gray.integral()
	acc := newPlane(gray.rows, gray.cols)
	maxRadius := gray.rows
	if gray.cols > maxRadius {
		maxRadius = gray.cols
	}

	for s := 0; s < n; s++ {
		r := 1 << uint(s)
		if r > maxRadius {
			r = maxRadius
		}
		resp := newPlane(gray.rows, gray.cols)
		for y := 0; y < gray.rows; y++ {
			for x := 0; x < gray.cols; x++ {
				surround := boxMean(sat, gray.rows, gray.cols, y, x, r)
				resp.data[y*gray.cols+x] = math.Abs(gray.at(y, x) - surround)
			}
		}
		// Give every scale equal weight regardless of its raw magnitude.
		resp.normalizeUnit()
		for i := range acc.data {
			acc.data[i] += resp.data[i]
		}
	}
	return acc.normalizedMat()
}
