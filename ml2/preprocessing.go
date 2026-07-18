package ml2

import "math"

// StandardScaler standardises features to zero mean and unit variance,
// column-wise. Fit learns the per-feature mean and standard deviation from a
// training set; Transform applies the same shift and scale to any data, which
// is the correct way to avoid leaking test statistics into training.
type StandardScaler struct {
	// Mean holds the learned per-feature mean.
	Mean []float64
	// Std holds the learned per-feature standard deviation (population).
	Std []float64
}

// NewStandardScaler returns an unfitted StandardScaler.
func NewStandardScaler() *StandardScaler { return &StandardScaler{} }

// Fit learns the per-feature mean and standard deviation from x. Features with
// zero variance are given a standard deviation of one so Transform leaves them
// unchanged rather than dividing by zero. It panics on empty input.
func (s *StandardScaler) Fit(x [][]float64) {
	if len(x) == 0 {
		panic("ml2: StandardScaler.Fit given no samples")
	}
	d := len(x[0])
	mean := ml2columnMean(x)
	std := make([]float64, d)
	for _, row := range x {
		for j := 0; j < d; j++ {
			diff := row[j] - mean[j]
			std[j] += diff * diff
		}
	}
	n := float64(len(x))
	for j := 0; j < d; j++ {
		std[j] = math.Sqrt(std[j] / n)
		if std[j] == 0 {
			std[j] = 1
		}
	}
	s.Mean, s.Std = mean, std
}

// Transform standardises x with the learned statistics and returns a new
// matrix. It panics if the scaler is unfitted.
func (s *StandardScaler) Transform(x [][]float64) [][]float64 {
	if s.Mean == nil {
		panic("ml2: StandardScaler.Transform before Fit")
	}
	out := make([][]float64, len(x))
	for i, row := range x {
		nr := make([]float64, len(row))
		for j := range row {
			nr[j] = (row[j] - s.Mean[j]) / s.Std[j]
		}
		out[i] = nr
	}
	return out
}

// FitTransform is Fit followed by Transform on the same data.
func (s *StandardScaler) FitTransform(x [][]float64) [][]float64 {
	s.Fit(x)
	return s.Transform(x)
}

// MinMaxScaler linearly rescales each feature into the closed interval [0, 1]
// based on the minimum and maximum observed during Fit. Features that are
// constant across the training set map to zero.
type MinMaxScaler struct {
	// Min holds the learned per-feature minimum.
	Min []float64
	// Max holds the learned per-feature maximum.
	Max []float64
}

// NewMinMaxScaler returns an unfitted MinMaxScaler.
func NewMinMaxScaler() *MinMaxScaler { return &MinMaxScaler{} }

// Fit learns the per-feature minimum and maximum from x. It panics on empty
// input.
func (s *MinMaxScaler) Fit(x [][]float64) {
	if len(x) == 0 {
		panic("ml2: MinMaxScaler.Fit given no samples")
	}
	d := len(x[0])
	mn := make([]float64, d)
	mx := make([]float64, d)
	copy(mn, x[0])
	copy(mx, x[0])
	for _, row := range x[1:] {
		for j := 0; j < d; j++ {
			if row[j] < mn[j] {
				mn[j] = row[j]
			}
			if row[j] > mx[j] {
				mx[j] = row[j]
			}
		}
	}
	s.Min, s.Max = mn, mx
}

// Transform rescales x into [0, 1] with the learned bounds and returns a new
// matrix. It panics if the scaler is unfitted.
func (s *MinMaxScaler) Transform(x [][]float64) [][]float64 {
	if s.Min == nil {
		panic("ml2: MinMaxScaler.Transform before Fit")
	}
	out := make([][]float64, len(x))
	for i, row := range x {
		nr := make([]float64, len(row))
		for j := range row {
			rng := s.Max[j] - s.Min[j]
			if rng == 0 {
				nr[j] = 0
			} else {
				nr[j] = (row[j] - s.Min[j]) / rng
			}
		}
		out[i] = nr
	}
	return out
}

// FitTransform is Fit followed by Transform on the same data.
func (s *MinMaxScaler) FitTransform(x [][]float64) [][]float64 {
	s.Fit(x)
	return s.Transform(x)
}

// Normalize returns a copy of v scaled to unit L2 norm. A zero vector is
// returned unchanged.
func Normalize(v []float64) []float64 {
	var s float64
	for _, x := range v {
		s += x * x
	}
	norm := math.Sqrt(s)
	out := make([]float64, len(v))
	if norm == 0 {
		copy(out, v)
		return out
	}
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}

// NormalizeRows applies [Normalize] to every row of x, returning a new matrix
// in which each row has unit L2 norm.
func NormalizeRows(x [][]float64) [][]float64 {
	out := make([][]float64, len(x))
	for i, row := range x {
		out[i] = Normalize(row)
	}
	return out
}
