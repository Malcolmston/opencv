package ml

import (
	"errors"
	"math"
	"sort"
)

// Common training errors returned by the Train methods.
var (
	// ErrNoSamples reports that the sample matrix was empty.
	ErrNoSamples = errors.New("ml: no samples provided")
	// ErrLabelMismatch reports that the number of labels (or responses) did
	// not equal the number of samples.
	ErrLabelMismatch = errors.New("ml: number of labels must equal number of samples")
	// ErrRaggedSamples reports that not every sample had the same feature
	// count.
	ErrRaggedSamples = errors.New("ml: all samples must have the same number of features")
	// ErrNotTrained reports that a prediction method was called before Train
	// completed successfully.
	ErrNotTrained = errors.New("ml: model has not been trained")
)

// validateSamples checks that samples is non-empty and rectangular and that
// labelCount matches the sample count. It returns the feature dimensionality.
func validateSamples(samples [][]float64, labelCount int) (dim int, err error) {
	if len(samples) == 0 {
		return 0, ErrNoSamples
	}
	if labelCount != len(samples) {
		return 0, ErrLabelMismatch
	}
	dim = len(samples[0])
	if dim == 0 {
		return 0, ErrRaggedSamples
	}
	for _, s := range samples {
		if len(s) != dim {
			return 0, ErrRaggedSamples
		}
	}
	return dim, nil
}

// classInfo returns the sorted set of distinct labels together with a map from
// a label value to its index in that sorted slice. The stable ordering makes
// multiclass models deterministic regardless of label ordering in the input.
func classInfo(labels []int) (classes []int, index map[int]int) {
	set := make(map[int]struct{}, len(labels))
	for _, l := range labels {
		set[l] = struct{}{}
	}
	classes = make([]int, 0, len(set))
	for c := range set {
		classes = append(classes, c)
	}
	sort.Ints(classes)
	index = make(map[int]int, len(classes))
	for i, c := range classes {
		index[c] = i
	}
	return classes, index
}

// squaredEuclidean returns the squared L2 distance between a and b, which must
// have the same length.
func squaredEuclidean(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// dot returns the inner product of a and b, which must have the same length.
func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// argmax returns the index of the largest element in v. Ties resolve to the
// lowest index. It panics on an empty slice.
func argmax(v []float64) int {
	if len(v) == 0 {
		panic("ml: argmax of empty slice")
	}
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

// cloneSample returns an independent copy of x.
func cloneSample(x []float64) []float64 {
	out := make([]float64, len(x))
	copy(out, x)
	return out
}

// scaler standardises features to zero mean and unit variance. Features with
// zero variance are left unscaled (their std is treated as 1) so that constant
// columns do not produce NaNs.
type scaler struct {
	mean []float64
	std  []float64
}

// fitScaler computes per-feature mean and standard deviation over samples.
func fitScaler(samples [][]float64) *scaler {
	dim := len(samples[0])
	n := float64(len(samples))
	mean := make([]float64, dim)
	for _, s := range samples {
		for j, v := range s {
			mean[j] += v
		}
	}
	for j := range mean {
		mean[j] /= n
	}
	std := make([]float64, dim)
	for _, s := range samples {
		for j, v := range s {
			d := v - mean[j]
			std[j] += d * d
		}
	}
	for j := range std {
		std[j] = math.Sqrt(std[j] / n)
		if std[j] == 0 {
			std[j] = 1
		}
	}
	return &scaler{mean: mean, std: std}
}

// transform standardises x in place into a fresh slice using the fitted
// statistics.
func (s *scaler) transform(x []float64) []float64 {
	out := make([]float64, len(x))
	for j, v := range x {
		out[j] = (v - s.mean[j]) / s.std[j]
	}
	return out
}
