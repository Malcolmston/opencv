package matching2

import (
	"math"
	"math/bits"
)

// NormType selects the distance used to compare floating-point descriptors.
type NormType int

const (
	// NormL2 is the Euclidean distance, the correct choice for gradient-based
	// descriptors such as SIFT.
	NormL2 NormType = iota
	// NormL2Sqr is the squared Euclidean distance. It ranks matches identically
	// to NormL2 but avoids the square root, so it is faster when only the
	// ordering matters.
	NormL2Sqr
	// NormL1 is the Manhattan (sum of absolute differences) distance.
	NormL1
)

// DMatch records a correspondence between a query descriptor and a train
// descriptor, mirroring OpenCV's cv::DMatch and the type of the same name in
// the features2d subpackage.
type DMatch struct {
	// QueryIdx is the index of the descriptor in the query set.
	QueryIdx int
	// TrainIdx is the index of the matched descriptor in the train set.
	TrainIdx int
	// ImgIdx is the index of the train image the descriptor came from; it is 0
	// unless the caller matches against multiple images.
	ImgIdx int
	// Distance is the descriptor distance; smaller means more similar.
	Distance float64
}

// L2Distance returns the Euclidean distance between two equal-length float
// descriptors. It panics if the lengths differ.
func L2Distance(a, b []float64) float64 {
	return math.Sqrt(L2SquaredDistance(a, b))
}

// L2SquaredDistance returns the squared Euclidean distance between two
// equal-length float descriptors. It panics if the lengths differ.
func L2SquaredDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("matching2: L2SquaredDistance length mismatch")
	}
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// L1Distance returns the Manhattan distance between two equal-length float
// descriptors. It panics if the lengths differ.
func L1Distance(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("matching2: L1Distance length mismatch")
	}
	var s float64
	for i := range a {
		s += math.Abs(a[i] - b[i])
	}
	return s
}

// HammingDistance returns the number of differing bits between two equal-length
// bit-packed binary descriptors. It is the correct distance for binary
// descriptors such as ORB and BRIEF. It panics if the lengths differ.
func HammingDistance(a, b []byte) int {
	if len(a) != len(b) {
		panic("matching2: HammingDistance length mismatch")
	}
	d := 0
	for i := range a {
		d += bits.OnesCount8(a[i] ^ b[i])
	}
	return d
}

// FloatDistance computes the distance between two float descriptors under the
// given norm. It panics if the lengths differ or norm is unknown.
func FloatDistance(norm NormType, a, b []float64) float64 {
	switch norm {
	case NormL2:
		return L2Distance(a, b)
	case NormL2Sqr:
		return L2SquaredDistance(a, b)
	case NormL1:
		return L1Distance(a, b)
	default:
		panic("matching2: unknown NormType")
	}
}

// NormalizeL2 returns a copy of the descriptor scaled to unit Euclidean length.
// A zero vector is returned unchanged. This is the standard pre-processing for
// RootSIFT-style matching.
func NormalizeL2(d []float64) []float64 {
	var s float64
	for _, x := range d {
		s += x * x
	}
	out := make([]float64, len(d))
	if s == 0 {
		copy(out, d)
		return out
	}
	inv := 1 / math.Sqrt(s)
	for i, x := range d {
		out[i] = x * inv
	}
	return out
}
