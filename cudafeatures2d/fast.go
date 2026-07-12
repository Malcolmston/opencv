package cudafeatures2d

import (
	"github.com/malcolmston/opencv/features2d"
)

// FastFeatureDetector is a CPU-backed mirror of cv::cuda::FastFeatureDetector.
// It detects FAST-9 corners by delegating to features2d.FastFeatureDetector.
//
// Construct one with [CreateFastFeatureDetector].
type FastFeatureDetector struct {
	impl *features2d.FastFeatureDetector
}

// CreateFastFeatureDetector returns a FAST detector with the given intensity
// threshold and non-maximum suppression setting. It mirrors
// cv::cuda::FastFeatureDetector::create; the GPU max_npoints and type parameters
// have no effect here and are omitted.
func CreateFastFeatureDetector(threshold int, nonmaxSuppression bool) *FastFeatureDetector {
	return &FastFeatureDetector{
		impl: &features2d.FastFeatureDetector{
			Threshold:         threshold,
			NonmaxSuppression: nonmaxSuppression,
		},
	}
}

// Detect returns the FAST corners of the device image as keypoints, optionally
// filtered by mask (pass a nil or empty GpuMat for no mask). It mirrors
// cv::cuda::FastFeatureDetector::detect. It panics if image is empty.
func (d *FastFeatureDetector) Detect(image, mask *GpuMat) []KeyPoint {
	if image.Empty() {
		panic("cudafeatures2d: FastFeatureDetector.Detect on empty image")
	}
	kps := d.impl.Detect(image.host())
	return applyMask(kps, mask)
}

// DetectAsync is the streamed form of [FastFeatureDetector.Detect]. The stream
// is inert in this synchronous port; a nil stream is accepted.
func (d *FastFeatureDetector) DetectAsync(image, mask *GpuMat, _ *Stream) []KeyPoint {
	return d.Detect(image, mask)
}
