package ml

import "math"

// LogisticRegression is a multinomial (softmax) logistic-regression classifier
// trained by batch gradient descent on the cross-entropy loss with optional L2
// regularisation. It naturally handles two or more classes: with two classes it
// reduces to ordinary binary logistic regression.
//
// Features are standardised internally, so differing feature scales do not
// require caller pre-processing. Construct with [NewLogisticRegression] and
// tune the exported fields before calling Train.
type LogisticRegression struct {
	// LearningRate is the gradient-descent step size; it must be positive.
	LearningRate float64
	// Epochs is the number of full-batch gradient steps.
	Epochs int
	// L2 is the L2 regularisation strength applied to the weights (not the
	// bias). Zero disables regularisation.
	L2 float64

	scaler  *scaler
	weights [][]float64 // [class][feature]
	biases  []float64   // [class]
	classes []int
	dim     int
	trained bool
}

// NewLogisticRegression returns a classifier with sensible defaults (learning
// rate 0.1, 500 epochs, no L2 penalty).
func NewLogisticRegression() *LogisticRegression {
	return &LogisticRegression{LearningRate: 0.1, Epochs: 500}
}

// Train fits the softmax weights by gradient descent. It returns an error if
// the input is empty, ragged, or the label count does not match.
func (m *LogisticRegression) Train(samples [][]float64, labels []int) error {
	dim, err := validateSamples(samples, len(labels))
	if err != nil {
		return err
	}
	if m.LearningRate <= 0 {
		m.LearningRate = 0.1
	}
	if m.Epochs <= 0 {
		m.Epochs = 500
	}
	classes, index := classInfo(labels)
	k := len(classes)
	m.scaler = fitScaler(samples)
	X := make([][]float64, len(samples))
	for i, s := range samples {
		X[i] = m.scaler.transform(s)
	}
	target := make([]int, len(labels))
	for i, l := range labels {
		target[i] = index[l]
	}

	w := make([][]float64, k)
	b := make([]float64, k)
	for c := 0; c < k; c++ {
		w[c] = make([]float64, dim)
	}
	n := float64(len(X))
	logits := make([]float64, k)

	for epoch := 0; epoch < m.Epochs; epoch++ {
		gradW := make([][]float64, k)
		gradB := make([]float64, k)
		for c := 0; c < k; c++ {
			gradW[c] = make([]float64, dim)
		}
		for i, x := range X {
			for c := 0; c < k; c++ {
				logits[c] = dot(w[c], x) + b[c]
			}
			probs := softmax(logits)
			for c := 0; c < k; c++ {
				diff := probs[c]
				if c == target[i] {
					diff -= 1
				}
				for j := range x {
					gradW[c][j] += diff * x[j]
				}
				gradB[c] += diff
			}
		}
		for c := 0; c < k; c++ {
			for j := 0; j < dim; j++ {
				g := gradW[c][j]/n + m.L2*w[c][j]
				w[c][j] -= m.LearningRate * g
			}
			b[c] -= m.LearningRate * gradB[c] / n
		}
	}

	m.weights = w
	m.biases = b
	m.classes = classes
	m.dim = dim
	m.trained = true
	return nil
}

// softmax returns the softmax of logits, shifted by the maximum for numerical
// stability.
func softmax(logits []float64) []float64 {
	maxV := logits[0]
	for _, v := range logits[1:] {
		if v > maxV {
			maxV = v
		}
	}
	out := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		e := math.Exp(v - maxV)
		out[i] = e
		sum += e
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Probabilities returns the predicted class probabilities for a sample, in
// sorted class order (see [LogisticRegression.Classes]). It panics if the model
// is untrained or the sample has the wrong length.
func (m *LogisticRegression) Probabilities(sample []float64) []float64 {
	m.checkQuery(sample)
	scaled := m.scaler.transform(sample)
	logits := make([]float64, len(m.classes))
	for c := range m.classes {
		logits[c] = dot(m.weights[c], scaled) + m.biases[c]
	}
	return softmax(logits)
}

// Classes returns the sorted set of class labels the model was trained on.
func (m *LogisticRegression) Classes() []int {
	return append([]int(nil), m.classes...)
}

// Predict classifies a single sample. It panics if the model is untrained or
// the sample has the wrong length.
func (m *LogisticRegression) Predict(sample []float64) int {
	return m.classes[argmax(m.Probabilities(sample))]
}

// PredictBatch classifies every sample and returns the predicted labels in
// order.
func (m *LogisticRegression) PredictBatch(samples [][]float64) []int {
	out := make([]int, len(samples))
	for i, s := range samples {
		out[i] = m.Predict(s)
	}
	return out
}

func (m *LogisticRegression) checkQuery(sample []float64) {
	if !m.trained {
		panic(ErrNotTrained)
	}
	if len(sample) != m.dim {
		panic("ml: LogisticRegression sample has wrong feature count")
	}
}
