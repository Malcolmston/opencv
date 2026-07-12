package flann

import (
	"fmt"
	"math"
)

// DistL1 returns the Manhattan (L1, "city block") distance between two
// real-valued vectors, sum(|a[i]-b[i]|). It is a genuine metric and, like every
// distance here, is symmetric, non-negative and zero for identical inputs. It
// panics if the vectors differ in length.
func DistL1(a, b []float64) float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("flann: DistL1 length mismatch %d vs %d", len(a), len(b)))
	}
	var sum float64
	for i := range a {
		sum += math.Abs(a[i] - b[i])
	}
	return sum
}

// DistChiSquare returns the (symmetric) chi-square distance between two
// non-negative vectors, sum((a[i]-b[i])^2 / (a[i]+b[i])), a standard measure of
// dissimilarity between histograms. Coordinate pairs whose sum is zero
// contribute nothing (their squared difference is also zero). It panics if the
// vectors differ in length.
func DistChiSquare(a, b []float64) float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("flann: DistChiSquare length mismatch %d vs %d", len(a), len(b)))
	}
	var sum float64
	for i := range a {
		denom := a[i] + b[i]
		if denom == 0 {
			continue
		}
		d := a[i] - b[i]
		sum += d * d / denom
	}
	return sum
}

// DistMinkowski returns the order-p Minkowski distance between two real-valued
// vectors, (sum(|a[i]-b[i]|^p))^(1/p). p == 1 recovers [DistL1] and p == 2
// recovers [DistL2]; as p grows the distance approaches the Chebyshev (L-inf)
// maximum-coordinate distance. It panics if the vectors differ in length or if
// p < 1 (values below 1 do not yield a metric). Use [MinkowskiDist] to obtain a
// [DistanceFunc] bound to a fixed p for [NewLinearIndexFunc].
func DistMinkowski(a, b []float64, p float64) float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("flann: DistMinkowski length mismatch %d vs %d", len(a), len(b)))
	}
	if p < 1 {
		panic(fmt.Sprintf("flann: DistMinkowski order p=%v must be >= 1", p))
	}
	var sum float64
	for i := range a {
		sum += math.Pow(math.Abs(a[i]-b[i]), p)
	}
	return math.Pow(sum, 1/p)
}

// MinkowskiDist returns a [DistanceFunc] that computes the order-p Minkowski
// distance, suitable for [NewLinearIndexFunc]. It panics immediately if p < 1.
func MinkowskiDist(p float64) DistanceFunc[[]float64] {
	if p < 1 {
		panic(fmt.Sprintf("flann: MinkowskiDist order p=%v must be >= 1", p))
	}
	return func(a, b []float64) float64 { return DistMinkowski(a, b, p) }
}

// DistHellinger returns the Hellinger distance between two non-negative vectors
// (typically probability distributions or histograms),
// sqrt(0.5 * sum((sqrt(a[i]) - sqrt(b[i]))^2)). It is bounded in [0, 1] for
// vectors that each sum to one and is a true metric. Negative coordinates are
// treated as zero under the square root. It panics if the vectors differ in
// length.
func DistHellinger(a, b []float64) float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("flann: DistHellinger length mismatch %d vs %d", len(a), len(b)))
	}
	var sum float64
	for i := range a {
		d := math.Sqrt(nonNeg(a[i])) - math.Sqrt(nonNeg(b[i]))
		sum += d * d
	}
	return math.Sqrt(0.5 * sum)
}

// DistCosine returns the cosine distance between two real-valued vectors,
// 1 - (a·b)/(|a||b|), which lies in [0, 2]: 0 for vectors pointing the same
// way, 1 for orthogonal vectors, 2 for opposite ones. If either vector is the
// zero vector its direction is undefined and the distance is defined as 1. It
// panics if the vectors differ in length.
func DistCosine(a, b []float64) float64 {
	if len(a) != len(b) {
		panic(fmt.Sprintf("flann: DistCosine length mismatch %d vs %d", len(a), len(b)))
	}
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 1
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	// Guard against tiny floating-point excursions outside [-1, 1].
	if cos > 1 {
		cos = 1
	} else if cos < -1 {
		cos = -1
	}
	return 1 - cos
}

// nonNeg clamps negative values to zero for the square-root distances.
func nonNeg(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}
