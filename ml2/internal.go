package ml2

import (
	"math"
	"sort"
)

// ml2squaredEuclidean returns the squared L2 distance between two equal-length
// vectors. Callers that only compare distances avoid the square root.
func ml2squaredEuclidean(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// ml2euclidean returns the L2 distance between two equal-length vectors.
func ml2euclidean(a, b []float64) float64 {
	return math.Sqrt(ml2squaredEuclidean(a, b))
}

// ml2dot returns the dot product of two equal-length vectors.
func ml2dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// ml2argmax returns the index of the largest element of v, breaking ties toward
// the lowest index. It returns -1 for an empty slice.
func ml2argmax(v []float64) int {
	if len(v) == 0 {
		return -1
	}
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

// ml2maxInt returns the largest of the provided integers, or 0 for empty input.
func ml2maxInt(v []int) int {
	if len(v) == 0 {
		return 0
	}
	m := v[0]
	for _, x := range v[1:] {
		if x > m {
			m = x
		}
	}
	return m
}

// ml2numClasses returns one plus the largest label, i.e. the number of distinct
// class indices assuming labels are dense in [0, k). It returns 0 for empty
// input.
func ml2numClasses(labels []int) int {
	if len(labels) == 0 {
		return 0
	}
	return ml2maxInt(labels) + 1
}

// ml2majority returns the most frequent label among the given labels, using
// numClasses buckets and breaking ties toward the lowest label index.
func ml2majority(labels []int, numClasses int) int {
	counts := make([]int, numClasses)
	for _, l := range labels {
		counts[l]++
	}
	best, bestC := 0, -1
	for c, n := range counts {
		if n > bestC {
			bestC, best = n, c
		}
	}
	return best
}

// ml2softmaxInPlace converts a vector of scores into a probability distribution
// in place, using the numerically stable max-shift formulation.
func ml2softmaxInPlace(z []float64) {
	if len(z) == 0 {
		return
	}
	mx := z[0]
	for _, v := range z[1:] {
		if v > mx {
			mx = v
		}
	}
	var sum float64
	for i := range z {
		z[i] = math.Exp(z[i] - mx)
		sum += z[i]
	}
	if sum == 0 {
		return
	}
	for i := range z {
		z[i] /= sum
	}
}

// ml2columnMean returns the per-column mean of a rectangular data matrix.
func ml2columnMean(x [][]float64) []float64 {
	if len(x) == 0 {
		return nil
	}
	d := len(x[0])
	mean := make([]float64, d)
	for _, row := range x {
		for j := 0; j < d; j++ {
			mean[j] += row[j]
		}
	}
	n := float64(len(x))
	for j := range mean {
		mean[j] /= n
	}
	return mean
}

// ml2sortedUnique returns the sorted distinct values of v.
func ml2sortedUnique(v []float64) []float64 {
	if len(v) == 0 {
		return nil
	}
	cp := make([]float64, len(v))
	copy(cp, v)
	sort.Float64s(cp)
	out := cp[:1]
	for _, x := range cp[1:] {
		if x != out[len(out)-1] {
			out = append(out, x)
		}
	}
	return out
}
