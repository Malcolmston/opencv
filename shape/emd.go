package shape

import "math"

// EMDL1 returns the Earth Mover's Distance between two one-dimensional
// histograms (or signatures) under the L1 ground metric, mirroring the intent of
// OpenCV's cv::EMDL1 for the 1-D case. The two histograms must have the same
// length; each is treated as a distribution over equally spaced bins.
//
// For distributions on a line with the L1 (absolute-difference) ground distance,
// the earth mover's distance has the classical closed form: the total flow
// needed equals the L1 distance between the two cumulative distribution
// functions, that is the sum over bins of the absolute difference of the running
// partial sums. This is exact — no linear program is solved — and is the value
// returned here. Both inputs are normalised to unit mass first, so the result is
// scale-free; a histogram of all zeros is treated as empty and yields 0 against
// another empty histogram (and the full mass of the other otherwise).
//
// It panics when the two histograms have different lengths.
func EMDL1(h1, h2 []float64) float64 {
	if len(h1) != len(h2) {
		panic("shape: EMDL1 histograms of unequal length")
	}
	if len(h1) == 0 {
		return 0
	}
	var s1, s2 float64
	for i := range h1 {
		s1 += math.Abs(h1[i])
		s2 += math.Abs(h2[i])
	}
	// Normalise to unit mass so the distance is independent of overall counts.
	norm := func(v, s float64) float64 {
		if s == 0 {
			return 0
		}
		return math.Abs(v) / s
	}
	var cdf1, cdf2, total float64
	for i := range h1 {
		cdf1 += norm(h1[i], s1)
		cdf2 += norm(h2[i], s2)
		total += math.Abs(cdf1 - cdf2)
	}
	return total
}
