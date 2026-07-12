package features2d

import cv "github.com/malcolmston/opencv"

// KeyPoint is a salient image point produced by a detector. It mirrors OpenCV's
// cv::KeyPoint.
type KeyPoint struct {
	// Pt is the keypoint location in pixel coordinates (x is the column, y the
	// row).
	Pt cv.Point
	// Size is the diameter of the meaningful keypoint neighbourhood in pixels.
	Size float64
	// Angle is the keypoint orientation in degrees in the range [0, 360). A
	// value of -1 means no orientation was computed.
	Angle float64
	// Response is the detector response used to rank keypoints; larger is
	// stronger.
	Response float64
	// Octave is the pyramid layer the keypoint was detected in. This package
	// works at a single scale, so it is always 0.
	Octave int
}

// DMatch records a correspondence between a query descriptor and a train
// descriptor, as produced by a [BFMatcher]. It mirrors OpenCV's cv::DMatch.
type DMatch struct {
	// QueryIdx is the index of the descriptor in the query set.
	QueryIdx int
	// TrainIdx is the index of the matched descriptor in the train set.
	TrainIdx int
	// Distance is the descriptor distance; smaller means more similar.
	Distance float64
}

// Descriptors holds a set of feature descriptors, one per keypoint. Exactly one
// of Binary or Float is populated: binary descriptors (as produced by [BRIEF]
// and [ORB]) pack their bits into bytes and are compared with the Hamming
// distance, while float descriptors are compared with the Euclidean distance.
// Build one with [NewBinaryDescriptors] or [NewFloatDescriptors].
type Descriptors struct {
	// Binary holds one bit-packed descriptor per keypoint; nil for float
	// descriptors. Every row has the same length.
	Binary [][]byte
	// Float holds one float descriptor per keypoint; nil for binary
	// descriptors. Every row has the same length.
	Float [][]float64
}

// NewBinaryDescriptors wraps bit-packed descriptor rows (such as the [][]byte
// returned by [ORB.DetectAndCompute]) in a [Descriptors].
func NewBinaryDescriptors(rows [][]byte) Descriptors {
	return Descriptors{Binary: rows}
}

// NewFloatDescriptors wraps float descriptor rows in a [Descriptors].
func NewFloatDescriptors(rows [][]float64) Descriptors {
	return Descriptors{Float: rows}
}

// IsBinary reports whether the descriptors are binary (Hamming) rather than
// float (Euclidean).
func (d Descriptors) IsBinary() bool {
	return d.Binary != nil
}

// Len returns the number of descriptor rows.
func (d Descriptors) Len() int {
	if d.Binary != nil {
		return len(d.Binary)
	}
	return len(d.Float)
}
