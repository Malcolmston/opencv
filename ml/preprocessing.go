package ml

import "math"

// StandardScaler standardises features to zero mean and unit variance, the
// exported counterpart of the standardisation the models apply internally. Fit
// it on the training set, then Transform both training and test data with the
// same statistics. Constant features (zero variance) are left unscaled to avoid
// division by zero.
type StandardScaler struct {
	// Mean holds the per-feature mean learned by Fit.
	Mean []float64
	// Std holds the per-feature standard deviation learned by Fit (a zero std is
	// stored as 1).
	Std []float64
}

// Fit learns the per-feature mean and standard deviation from samples and
// returns the receiver for chaining. It panics if samples is empty or ragged.
func (s *StandardScaler) Fit(samples [][]float64) *StandardScaler {
	if _, err := validateSamples(samples, len(samples)); err != nil {
		panic("ml: StandardScaler.Fit given invalid samples: " + err.Error())
	}
	sc := fitScaler(samples)
	s.Mean = sc.mean
	s.Std = sc.std
	return s
}

// Transform standardises a single feature vector into a fresh slice. It panics
// if the scaler has not been fitted or x has the wrong length.
func (s *StandardScaler) Transform(x []float64) []float64 {
	if s.Mean == nil {
		panic(ErrNotTrained)
	}
	if len(x) != len(s.Mean) {
		panic("ml: StandardScaler.Transform feature-length mismatch")
	}
	out := make([]float64, len(x))
	for j, v := range x {
		out[j] = (v - s.Mean[j]) / s.Std[j]
	}
	return out
}

// TransformAll standardises every sample and returns a new matrix.
func (s *StandardScaler) TransformAll(samples [][]float64) [][]float64 {
	out := make([][]float64, len(samples))
	for i, x := range samples {
		out[i] = s.Transform(x)
	}
	return out
}

// FitTransform fits the scaler on samples and returns the standardised samples
// in one call.
func (s *StandardScaler) FitTransform(samples [][]float64) [][]float64 {
	return s.Fit(samples).TransformAll(samples)
}

// MinMaxScaler rescales each feature linearly to the range [0, 1] based on the
// minimum and maximum seen during Fit. Constant features map to 0.
type MinMaxScaler struct {
	// Min holds the per-feature minimum learned by Fit.
	Min []float64
	// Max holds the per-feature maximum learned by Fit.
	Max []float64
}

// Fit learns the per-feature minimum and maximum from samples and returns the
// receiver for chaining. It panics if samples is empty or ragged.
func (s *MinMaxScaler) Fit(samples [][]float64) *MinMaxScaler {
	dim, err := validateSamples(samples, len(samples))
	if err != nil {
		panic("ml: MinMaxScaler.Fit given invalid samples: " + err.Error())
	}
	s.Min = make([]float64, dim)
	s.Max = make([]float64, dim)
	for j := 0; j < dim; j++ {
		s.Min[j] = math.Inf(1)
		s.Max[j] = math.Inf(-1)
	}
	for _, x := range samples {
		for j, v := range x {
			if v < s.Min[j] {
				s.Min[j] = v
			}
			if v > s.Max[j] {
				s.Max[j] = v
			}
		}
	}
	return s
}

// Transform rescales a single feature vector to [0, 1] into a fresh slice. It
// panics if the scaler has not been fitted or x has the wrong length.
func (s *MinMaxScaler) Transform(x []float64) []float64 {
	if s.Min == nil {
		panic(ErrNotTrained)
	}
	if len(x) != len(s.Min) {
		panic("ml: MinMaxScaler.Transform feature-length mismatch")
	}
	out := make([]float64, len(x))
	for j, v := range x {
		span := s.Max[j] - s.Min[j]
		if span == 0 {
			out[j] = 0
			continue
		}
		out[j] = (v - s.Min[j]) / span
	}
	return out
}

// TransformAll rescales every sample and returns a new matrix.
func (s *MinMaxScaler) TransformAll(samples [][]float64) [][]float64 {
	out := make([][]float64, len(samples))
	for i, x := range samples {
		out[i] = s.Transform(x)
	}
	return out
}

// FitTransform fits the scaler on samples and returns the rescaled samples in
// one call.
func (s *MinMaxScaler) FitTransform(samples [][]float64) [][]float64 {
	return s.Fit(samples).TransformAll(samples)
}
