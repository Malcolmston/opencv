package xfeatures2d

import cv "github.com/malcolmston/opencv"

// TEBLID computes the Triplet-based Efficient Binary Local Image Descriptor, a
// weight-free port of OpenCV's cv::xfeatures2d::TEBLID.
//
// TEBLID is the successor of [BEBLID]: it uses the same average-box weak-learner
// form (each bit compares the mean intensity of two small boxes) but was trained
// on a much larger, more varied patch set and, in OpenCV, ships in larger sizes.
// This port mirrors that relationship — it reuses the same integral-image box
// mechanism as BEBLID with a wider default sampling window and a distinct
// pseudo-random arrangement, and, like the rest of this package, embeds no
// learned tables (documented as the untrained, weight-free approximation). The
// pattern is rotated to the keypoint orientation for rotation invariance.
//
// Two descriptors are compared with the [HammingDistance].
type TEBLID struct {
	pattern *boxPairPattern
	bytes   int
}

// NewTEBLID returns a TEBLID extractor producing descriptors of the given byte
// length (typically 32 or 64). It panics if bytes is not positive.
func NewTEBLID(bytes int) *TEBLID {
	if bytes <= 0 {
		panic("xfeatures2d: NewTEBLID requires a positive byte length")
	}
	return &TEBLID{
		pattern: newBoxPairPattern(bytes*8, 20, 4, 0x7eb11d),
		bytes:   bytes,
	}
}

// DescriptorSizeBytes returns the number of bytes in each descriptor.
func (t *TEBLID) DescriptorSizeBytes() int { return t.bytes }

// Compute describes each keypoint of img and returns the keypoints (with Angle
// set to the estimated orientation) and their bit-packed descriptors. img may
// be single- or three-channel; a colour image is converted to gray.
func (t *TEBLID) Compute(img *cv.Mat, keypoints []KeyPoint) ([]KeyPoint, [][]byte) {
	return t.pattern.compute(toGray(img), keypoints, t.bytes, true)
}
