package xfeatures2d

import (
	"math/bits"

	cv "github.com/malcolmston/opencv"
)

// KeyPoint is a salient image point produced by a detector in this package. It
// is a self-contained mirror of OpenCV's cv::KeyPoint (declared here so the
// package does not depend on any sibling cv/* subpackage).
type KeyPoint struct {
	// Pt is the keypoint location in pixel coordinates (x is the column, y is
	// the row).
	Pt cv.Point
	// Size is the diameter of the meaningful keypoint neighbourhood in pixels.
	Size float64
	// Angle is the keypoint orientation in degrees in the range [0, 360). A
	// value of -1 means no orientation was computed.
	Angle float64
	// Response is the detector response used to rank keypoints; larger is
	// stronger.
	Response float64
}

// HammingDistance returns the number of differing bits between two bit-packed
// binary descriptors, i.e. the population count of a XOR b. It panics if the
// descriptors have different lengths.
func HammingDistance(a, b []byte) int {
	if len(a) != len(b) {
		panic("xfeatures2d: HammingDistance on descriptors of different length")
	}
	d := 0
	for i := range a {
		d += bits.OnesCount8(a[i] ^ b[i])
	}
	return d
}
