package ml2

import (
	"errors"
)

// LogisticRegression is a multinomial (softmax) logistic-regression classifier
// trained by full-batch gradient descent with optional L2 regularisation. It
// handles two or more classes uniformly; the binary case reduces to the usual
// sigmoid model.
type LogisticRegression struct {
	lr       float64
	iters    int
	l2       float64
	classes  int
	features int
	// Weights holds the learned coefficient matrix, Weights[c][j]; it is
	// populated by Fit.
	Weights [][]float64
	// Bias holds the learned per-class intercept; it is populated by Fit.
	Bias []float64
}

// NewLogisticRegression returns a classifier trained with the given learning
// rate, number of gradient-descent iterations and L2 penalty (l2 == 0 disables
// regularisation). It panics if lr <= 0, iters < 1 or l2 < 0.
func NewLogisticRegression(lr float64, iters int, l2 float64) *LogisticRegression {
	if lr <= 0 {
		panic("ml2: NewLogisticRegression requires lr > 0")
	}
	if iters < 1 {
		panic("ml2: NewLogisticRegression requires iters >= 1")
	}
	if l2 < 0 {
		panic("ml2: NewLogisticRegression requires l2 >= 0")
	}
	return &LogisticRegression{lr: lr, iters: iters, l2: l2}
}

// scores returns the pre-softmax class scores for one sample.
func (m *LogisticRegression) scores(sample []float64) []float64 {
	z := make([]float64, m.classes)
	for c := 0; c < m.classes; c++ {
		z[c] = m.Bias[c] + ml2dot(m.Weights[c], sample)
	}
	return z
}

// Fit trains the model with batch gradient descent on the softmax
// cross-entropy loss. It returns an error for empty or mismatched input.
func (m *LogisticRegression) Fit(samples [][]float64, labels []int) error {
	if len(samples) == 0 {
		return errors.New("ml2: LogisticRegression.Fit given no samples")
	}
	if len(samples) != len(labels) {
		return errors.New("ml2: LogisticRegression.Fit requires len(samples) == len(labels)")
	}
	m.classes = ml2numClasses(labels)
	m.features = len(samples[0])
	m.Weights = make([][]float64, m.classes)
	for c := range m.Weights {
		m.Weights[c] = make([]float64, m.features)
	}
	m.Bias = make([]float64, m.classes)
	n := float64(len(samples))
	gradW := make([][]float64, m.classes)
	for c := range gradW {
		gradW[c] = make([]float64, m.features)
	}
	gradB := make([]float64, m.classes)
	for it := 0; it < m.iters; it++ {
		for c := 0; c < m.classes; c++ {
			gradB[c] = 0
			for j := 0; j < m.features; j++ {
				gradW[c][j] = 0
			}
		}
		for i, s := range samples {
			p := m.scores(s)
			ml2softmaxInPlace(p)
			y := labels[i]
			for c := 0; c < m.classes; c++ {
				diff := p[c]
				if c == y {
					diff -= 1
				}
				gradB[c] += diff
				for j := 0; j < m.features; j++ {
					gradW[c][j] += diff * s[j]
				}
			}
		}
		for c := 0; c < m.classes; c++ {
			m.Bias[c] -= m.lr * gradB[c] / n
			for j := 0; j < m.features; j++ {
				g := gradW[c][j]/n + m.l2*m.Weights[c][j]
				m.Weights[c][j] -= m.lr * g
			}
		}
	}
	return nil
}

// PredictProba returns the softmax class probabilities for sample. It panics
// before Fit.
func (m *LogisticRegression) PredictProba(sample []float64) []float64 {
	if m.Weights == nil {
		panic("ml2: LogisticRegression.PredictProba before Fit")
	}
	z := m.scores(sample)
	ml2softmaxInPlace(z)
	return z
}

// Predict returns the most probable class for sample. It panics before Fit.
func (m *LogisticRegression) Predict(sample []float64) int {
	return ml2argmax(m.PredictProba(sample))
}

// PredictBatch classifies every sample in x.
func (m *LogisticRegression) PredictBatch(x [][]float64) []int {
	out := make([]int, len(x))
	for i, s := range x {
		out[i] = m.Predict(s)
	}
	return out
}
