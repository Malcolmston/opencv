package ml

import "math/rand"

// SVM is a linear soft-margin support-vector machine. It minimises the
// regularised hinge loss with the Pegasos stochastic sub-gradient method, and
// handles more than two classes with a one-vs-rest ensemble (one binary machine
// per class, the highest-scoring machine winning at prediction time).
//
// Features are standardised internally, so the model is robust to differing
// feature scales without the caller pre-normalising. Construct with [NewSVM];
// tune the exported fields before calling Train.
type SVM struct {
	// Lambda is the L2 regularisation strength; larger values yield a wider,
	// softer margin. It must be positive.
	Lambda float64
	// Epochs is the number of passes over the training set. More epochs give
	// the sub-gradient descent longer to converge.
	Epochs int
	// Seed makes the internal sample shuffling reproducible.
	Seed int64

	scaler  *scaler
	weights [][]float64 // one weight vector per class
	biases  []float64   // one bias per class
	classes []int
	dim     int
	trained bool
}

// NewSVM returns an SVM with sensible defaults (Lambda 1e-3, 200 epochs).
func NewSVM() *SVM {
	return &SVM{Lambda: 1e-3, Epochs: 200, Seed: 1}
}

// Train fits one Pegasos binary machine per class in a one-vs-rest scheme. It
// returns an error if the input is empty, ragged, or the label count does not
// match.
func (m *SVM) Train(samples [][]float64, labels []int) error {
	dim, err := validateSamples(samples, len(labels))
	if err != nil {
		return err
	}
	if m.Lambda <= 0 {
		m.Lambda = 1e-3
	}
	if m.Epochs <= 0 {
		m.Epochs = 200
	}
	m.scaler = fitScaler(samples)
	scaled := make([][]float64, len(samples))
	for i, s := range samples {
		scaled[i] = m.scaler.transform(s)
	}
	m.classes, _ = classInfo(labels)
	m.dim = dim
	m.weights = make([][]float64, len(m.classes))
	m.biases = make([]float64, len(m.classes))

	for ci, class := range m.classes {
		y := make([]float64, len(labels))
		for i, l := range labels {
			if l == class {
				y[i] = 1
			} else {
				y[i] = -1
			}
		}
		w, b := m.pegasos(scaled, y, int64(ci))
		m.weights[ci] = w
		m.biases[ci] = b
	}
	m.trained = true
	return nil
}

// pegasos trains a single binary hinge-loss machine on standardised samples X
// with ±1 targets y. offset perturbs the seed so each one-vs-rest machine draws
// a distinct but reproducible sample sequence.
func (m *SVM) pegasos(X [][]float64, y []float64, offset int64) (weights []float64, bias float64) {
	n := len(X)
	dim := m.dim
	w := make([]float64, dim)
	var b float64
	rng := rand.New(rand.NewSource(m.Seed + offset))
	lambda := m.Lambda
	t := 0
	for epoch := 0; epoch < m.Epochs; epoch++ {
		for step := 0; step < n; step++ {
			t++
			eta := 1.0 / (lambda * float64(t))
			i := rng.Intn(n)
			margin := y[i] * (dot(w, X[i]) + b)
			scale := 1.0 - eta*lambda
			for j := range w {
				w[j] *= scale
			}
			if margin < 1 {
				for j := range w {
					w[j] += eta * y[i] * X[i][j]
				}
				b += eta * y[i]
			}
		}
	}
	return w, b
}

// score returns the signed decision value of class ci for a standardised
// sample.
func (m *SVM) score(ci int, scaled []float64) float64 {
	return dot(m.weights[ci], scaled) + m.biases[ci]
}

// DecisionScores returns the raw one-vs-rest decision value for every class, in
// the model's sorted class order (see [SVM.Classes]). It panics if the model is
// untrained or the sample has the wrong length.
func (m *SVM) DecisionScores(sample []float64) []float64 {
	m.checkQuery(sample)
	scaled := m.scaler.transform(sample)
	out := make([]float64, len(m.classes))
	for ci := range m.classes {
		out[ci] = m.score(ci, scaled)
	}
	return out
}

// Classes returns the sorted set of class labels the model was trained on.
func (m *SVM) Classes() []int {
	return append([]int(nil), m.classes...)
}

// Predict classifies a single sample, returning the label of the highest-
// scoring one-vs-rest machine. It panics if the model is untrained or the
// sample has the wrong length.
func (m *SVM) Predict(sample []float64) int {
	scores := m.DecisionScores(sample)
	return m.classes[argmax(scores)]
}

// PredictBatch classifies every sample and returns the predicted labels in
// order.
func (m *SVM) PredictBatch(samples [][]float64) []int {
	out := make([]int, len(samples))
	for i, s := range samples {
		out[i] = m.Predict(s)
	}
	return out
}

func (m *SVM) checkQuery(sample []float64) {
	if !m.trained {
		panic(ErrNotTrained)
	}
	if len(sample) != m.dim {
		panic("ml: SVM sample has wrong feature count")
	}
}
