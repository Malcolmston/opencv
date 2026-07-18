package saliency2

import cv "github.com/malcolmston/opencv"

// StaticSaliencyFineGrained detects saliency with the fine-grained
// center-surround model of Montabone & Soto, "Human detection using a mobile
// platform and novel features derived from a visual saliency mechanism" (Image
// and Vision Computing 2010), the algorithm behind OpenCV's
// cv::saliency::StaticSaliencyFineGrained.
//
// The grayscale image is compared against local averages taken over a range of
// window radii. At each scale the absolute difference between a pixel and the
// mean of its surrounding window measures on-centre/off-centre contrast; the
// per-scale differences are averaged so that both small, crisp features and
// large, smooth objects contribute. The local means are computed in linear time
// with an integral image, so the cost is independent of window size.
//
// Construct one with [NewStaticSaliencyFineGrained]; the zero value computes a
// single default scale.
type StaticSaliencyFineGrained struct {
	// Scales holds the window radii used for the center-surround differences.
	// Radii that do not fit the image (>= half the smaller dimension) are
	// skipped at compute time. When empty, a default geometric set is used.
	Scales []int
}

// NewStaticSaliencyFineGrained returns a detector with the default multi-scale
// radii {1, 2, 4, 8, 16}.
func NewStaticSaliencyFineGrained() *StaticSaliencyFineGrained {
	return &StaticSaliencyFineGrained{Scales: []int{1, 2, 4, 8, 16}}
}

// ComputeSaliencyMap returns the fine-grained saliency of img as a
// [SaliencyMap] the same size as img. It panics if img is nil or empty.
func (s *StaticSaliencyFineGrained) ComputeSaliencyMap(img *cv.Mat) *SaliencyMap {
	gray := saliency2GrayFloat(img)
	rows, cols := gray.Rows, gray.Cols
	limit := rows
	if cols < limit {
		limit = cols
	}
	limit /= 2

	scales := s.Scales
	if len(scales) == 0 {
		scales = []int{1, 2, 4, 8, 16}
	}
	used := make([]int, 0, len(scales))
	for _, r := range scales {
		if r >= 1 && r < limit {
			used = append(used, r)
		}
	}
	if len(used) == 0 {
		used = []int{1}
	}

	ii := saliency2Integral(gray)
	out := NewSaliencyMap(rows, cols)
	for _, r := range used {
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				mean := saliency2BoxMean(ii, rows, cols, y, x, r)
				d := gray.Data[y*cols+x] - mean
				if d < 0 {
					d = -d
				}
				out.Data[y*cols+x] += d
			}
		}
	}
	inv := 1 / float64(len(used))
	for i := range out.Data {
		out.Data[i] *= inv
	}
	return saliency2GaussianBlurMap(out, 3, 1.0)
}

// ComputeSaliency returns the fine-grained saliency map of img as an 8-bit
// single-channel [cv.Mat]. It satisfies the [StaticSaliency] interface.
func (s *StaticSaliencyFineGrained) ComputeSaliency(img *cv.Mat) *cv.Mat {
	return s.ComputeSaliencyMap(img).ToMat()
}

// FineGrainedSaliency is a convenience wrapper that computes the fine-grained
// saliency map of img with the default detector settings.
func FineGrainedSaliency(img *cv.Mat) *cv.Mat {
	return NewStaticSaliencyFineGrained().ComputeSaliency(img)
}
