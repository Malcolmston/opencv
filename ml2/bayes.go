package ml2

import (
	"errors"
	"math"
)

// GaussianNB is a Gaussian naive Bayes classifier. It models each feature,
// conditioned on the class, as an independent one-dimensional Gaussian, and
// predicts the class with the greatest posterior. A small variance floor keeps
// the log-likelihood finite for near-constant features.
type GaussianNB struct {
	classes  int
	features int
	priors   []float64   // log prior per class
	means    [][]float64 // means[c][j]
	vars     [][]float64 // variances[c][j]
	epsilon  float64     // variance floor
}

// NewGaussianNB returns an unfitted Gaussian naive Bayes classifier.
func NewGaussianNB() *GaussianNB { return &GaussianNB{epsilon: 1e-9} }

// Fit estimates per-class priors, feature means and feature variances from the
// training data. It returns an error for empty or mismatched input.
func (m *GaussianNB) Fit(samples [][]float64, labels []int) error {
	if len(samples) == 0 {
		return errors.New("ml2: GaussianNB.Fit given no samples")
	}
	if len(samples) != len(labels) {
		return errors.New("ml2: GaussianNB.Fit requires len(samples) == len(labels)")
	}
	m.classes = ml2numClasses(labels)
	m.features = len(samples[0])
	counts := make([]int, m.classes)
	m.means = make([][]float64, m.classes)
	m.vars = make([][]float64, m.classes)
	for c := 0; c < m.classes; c++ {
		m.means[c] = make([]float64, m.features)
		m.vars[c] = make([]float64, m.features)
	}
	for i, s := range samples {
		c := labels[i]
		counts[c]++
		for j := 0; j < m.features; j++ {
			m.means[c][j] += s[j]
		}
	}
	for c := 0; c < m.classes; c++ {
		if counts[c] == 0 {
			continue
		}
		for j := 0; j < m.features; j++ {
			m.means[c][j] /= float64(counts[c])
		}
	}
	// Global variance to set the floor, following scikit-learn's convention.
	var maxVar float64
	for j := 0; j < m.features; j++ {
		mean := 0.0
		for _, s := range samples {
			mean += s[j]
		}
		mean /= float64(len(samples))
		v := 0.0
		for _, s := range samples {
			d := s[j] - mean
			v += d * d
		}
		v /= float64(len(samples))
		if v > maxVar {
			maxVar = v
		}
	}
	floor := m.epsilon * maxVar
	for i, s := range samples {
		c := labels[i]
		for j := 0; j < m.features; j++ {
			d := s[j] - m.means[c][j]
			m.vars[c][j] += d * d
		}
	}
	m.priors = make([]float64, m.classes)
	for c := 0; c < m.classes; c++ {
		if counts[c] == 0 {
			m.priors[c] = math.Inf(-1)
			continue
		}
		for j := 0; j < m.features; j++ {
			m.vars[c][j] = m.vars[c][j]/float64(counts[c]) + floor
			if m.vars[c][j] <= 0 {
				m.vars[c][j] = floor + 1e-12
			}
		}
		m.priors[c] = math.Log(float64(counts[c]) / float64(len(samples)))
	}
	return nil
}

// jointLogLikelihood returns the unnormalised log posterior of sample under
// each class.
func (m *GaussianNB) jointLogLikelihood(sample []float64) []float64 {
	out := make([]float64, m.classes)
	for c := 0; c < m.classes; c++ {
		ll := m.priors[c]
		if math.IsInf(ll, -1) {
			out[c] = ll
			continue
		}
		for j := 0; j < m.features; j++ {
			v := m.vars[c][j]
			d := sample[j] - m.means[c][j]
			ll += -0.5*math.Log(2*math.Pi*v) - (d*d)/(2*v)
		}
		out[c] = ll
	}
	return out
}

// Predict returns the class with the greatest posterior for sample. It panics
// before Fit.
func (m *GaussianNB) Predict(sample []float64) int {
	if m.means == nil {
		panic("ml2: GaussianNB.Predict before Fit")
	}
	return ml2argmax(m.jointLogLikelihood(sample))
}

// PredictBatch classifies every sample in x.
func (m *GaussianNB) PredictBatch(x [][]float64) []int {
	out := make([]int, len(x))
	for i, s := range x {
		out[i] = m.Predict(s)
	}
	return out
}

// PredictProba returns the posterior class probabilities for sample, normalised
// to sum to one. It panics before Fit.
func (m *GaussianNB) PredictProba(sample []float64) []float64 {
	if m.means == nil {
		panic("ml2: GaussianNB.PredictProba before Fit")
	}
	ll := m.jointLogLikelihood(sample)
	ml2softmaxInPlace(ll)
	return ll
}
