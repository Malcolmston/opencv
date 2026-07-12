package flann

import (
	"fmt"
	"math"
	"math/bits"
)

// positiveInf is the pruning bound used before a k-NN set is full.
var positiveInf = math.Inf(1)

// DistanceFunc reports the distance between two elements of type T. It must be
// symmetric and non-negative, returning 0 for identical inputs, so that the
// ordering of results is well defined. [DistL2] and [DistHamming] are the two
// distances used by the built-in indices; a custom one can be supplied to
// [NewLinearIndexFunc].
type DistanceFunc[T any] func(a, b T) float64

// DistL2 returns the Euclidean (L2) distance between two real-valued vectors,
// sqrt(sum((a[i]-b[i])^2)). It is the distance used by [KDTreeIndex],
// [KMeansIndex] and the float [LinearIndex], and the unit in which their radius
// searches are measured. It panics if the vectors differ in length.
func DistL2(a, b []float64) float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("flann: DistL2 length mismatch %d vs %d", len(a), len(b)))
	}
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// distL2Sq returns the squared Euclidean distance, avoiding the square root
// where only relative ordering matters. It assumes equal lengths.
func distL2Sq(a, b []float64) float64 {
	var sum float64
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return sum
}

// DistHamming returns the Hamming distance between two binary descriptors: the
// number of bit positions at which they differ. Each byte holds eight bits, so
// the result lies in [0, 8*len(a)]. It is the distance used by [LSHIndex] and
// the binary [LinearIndex]. It panics if the descriptors differ in length.
func DistHamming(a, b []byte) float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("flann: DistHamming length mismatch %d vs %d", len(a), len(b)))
	}
	var d int
	for i := range a {
		d += bits.OnesCount8(a[i] ^ b[i])
	}
	return float64(d)
}
