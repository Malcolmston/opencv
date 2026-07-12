package ml

import "math"

// NormalBayesClassifier is a Gaussian naive Bayes classifier. It models each
// feature within each class as an independent normal distribution, estimating a
// per-class, per-feature mean and variance during training. Prediction picks
// the class with the greatest posterior log-probability under those
// assumptions.
//
// Construct with [NewNormalBayesClassifier]. The zero value is not usable.
type NormalBayesClassifier struct {
	classes   []int
	logPrior  []float64
	means     [][]float64 // [class][feature]
	variances [][]float64 // [class][feature]
	dim       int
	trained   bool
}

// NewNormalBayesClassifier returns an untrained Gaussian naive Bayes
// classifier.
func NewNormalBayesClassifier() *NormalBayesClassifier {
	return &NormalBayesClassifier{}
}

// Train estimates the class priors and the per-class Gaussian parameters. A
// small floor is added to every variance to avoid division by zero for
// constant features. It returns an error if the input is empty, ragged, or the
// label count does not match.
func (m *NormalBayesClassifier) Train(samples [][]float64, labels []int) error {
	dim, err := validateSamples(samples, len(labels))
	if err != nil {
		return err
	}
	classes, index := classInfo(labels)
	k := len(classes)
	counts := make([]int, k)
	means := make([][]float64, k)
	variances := make([][]float64, k)
	for c := 0; c < k; c++ {
		means[c] = make([]float64, dim)
		variances[c] = make([]float64, dim)
	}
	for i, s := range samples {
		c := index[labels[i]]
		counts[c]++
		for j, v := range s {
			means[c][j] += v
		}
	}
	for c := 0; c < k; c++ {
		for j := 0; j < dim; j++ {
			means[c][j] /= float64(counts[c])
		}
	}
	for i, s := range samples {
		c := index[labels[i]]
		for j, v := range s {
			d := v - means[c][j]
			variances[c][j] += d * d
		}
	}
	// Global variance floor: a small fraction of the mean feature variance,
	// mirroring scikit-learn's var_smoothing, keeps the Gaussians well-posed.
	var maxVar float64
	for c := 0; c < k; c++ {
		for j := 0; j < dim; j++ {
			variances[c][j] /= float64(counts[c])
			if variances[c][j] > maxVar {
				maxVar = variances[c][j]
			}
		}
	}
	floor := 1e-9 * (maxVar + 1e-12)
	logPrior := make([]float64, k)
	total := float64(len(samples))
	for c := 0; c < k; c++ {
		for j := 0; j < dim; j++ {
			variances[c][j] += floor
		}
		logPrior[c] = math.Log(float64(counts[c]) / total)
	}
	m.classes = classes
	m.logPrior = logPrior
	m.means = means
	m.variances = variances
	m.dim = dim
	m.trained = true
	return nil
}

// logPosteriors returns the unnormalised posterior log-probability of each
// class for sample, in sorted class order.
func (m *NormalBayesClassifier) logPosteriors(sample []float64) []float64 {
	out := make([]float64, len(m.classes))
	for c := range m.classes {
		ll := m.logPrior[c]
		for j, v := range sample {
			variance := m.variances[c][j]
			d := v - m.means[c][j]
			ll += -0.5*math.Log(2*math.Pi*variance) - (d*d)/(2*variance)
		}
		out[c] = ll
	}
	return out
}

// Predict classifies a single sample. It panics if the model is untrained or
// the sample has the wrong length.
func (m *NormalBayesClassifier) Predict(sample []float64) int {
	m.checkQuery(sample)
	return m.classes[argmax(m.logPosteriors(sample))]
}

// PredictBatch classifies every sample and returns the predicted labels in
// order.
func (m *NormalBayesClassifier) PredictBatch(samples [][]float64) []int {
	out := make([]int, len(samples))
	for i, s := range samples {
		out[i] = m.Predict(s)
	}
	return out
}

func (m *NormalBayesClassifier) checkQuery(sample []float64) {
	if !m.trained {
		panic(ErrNotTrained)
	}
	if len(sample) != m.dim {
		panic("ml: NormalBayesClassifier sample has wrong feature count")
	}
}
