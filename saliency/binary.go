package saliency

import cv "github.com/malcolmston/opencv"

// ComputeBinaryMap converts a single-channel saliency map into a binary
// foreground mask (samples are 0 or 255) by thresholding it with Otsu's method,
// mirroring OpenCV's cv::saliency::StaticSaliency::computeBinaryMap. The
// threshold is chosen automatically from the map's histogram, so a saliency map
// with one distinct salient region yields a mask that isolates that region. It
// panics if saliency is nil or not single-channel.
func ComputeBinaryMap(saliency *cv.Mat) *cv.Mat {
	if saliency == nil || saliency.Empty() {
		panic("saliency: ComputeBinaryMap given an empty map")
	}
	if saliency.Channels != 1 {
		panic("saliency: ComputeBinaryMap requires a single-channel saliency map")
	}
	mask, _ := cv.Threshold(saliency, 0, 255, cv.ThreshBinary|cv.ThreshOtsu)
	return mask
}
