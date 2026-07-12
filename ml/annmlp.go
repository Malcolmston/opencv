package ml

import (
	"math"
	"math/rand"
)

// Activation selects the non-linearity applied to an [ANNMLP]'s hidden layers.
type Activation int

const (
	// Sigmoid is the logistic function 1/(1+e^-x), with output in (0, 1).
	Sigmoid Activation = iota
	// Tanh is the hyperbolic tangent, with output in (-1, 1). It is usually the
	// better default because its zero-centred output speeds up learning.
	Tanh
)

// ANNMLP is a feed-forward artificial neural network (multilayer perceptron)
// trained by error back-propagation, mirroring OpenCV's cv::ml::ANN_MLP. The
// architecture is fully configurable: the input width is inferred from the data,
// HiddenLayers lists the width of each hidden layer, and the output layer has
// one unit per class. Hidden units use the selected [Activation]; the output
// layer is a softmax trained against the cross-entropy loss, so the network is a
// proper probabilistic classifier.
//
// Inputs are standardised internally, and weights are initialised with the
// Xavier/Glorot scheme from a seeded generator, so training is deterministic.
// Construct with [NewANNMLP] and tune the exported fields before calling Train.
type ANNMLP struct {
	// HiddenLayers gives the width of each hidden layer, in order. An empty
	// slice yields a plain softmax (logistic-regression-like) network.
	HiddenLayers []int
	// Activation is the hidden-layer non-linearity.
	Activation Activation
	// LearningRate is the gradient-descent step size; it must be positive.
	LearningRate float64
	// Epochs is the number of full-batch gradient-descent passes.
	Epochs int
	// Seed makes weight initialisation reproducible.
	Seed int64

	layerSizes []int
	weights    [][][]float64 // [layer][out][in]
	biases     [][]float64   // [layer][out]
	scaler     *scaler
	classes    []int
	dim        int
	trained    bool
}

// NewANNMLP returns a network with one hidden layer of the given width, tanh
// activation, and sensible training defaults (learning rate 0.1, 1000 epochs).
// Pass hidden widths to configure a deeper network, e.g. NewANNMLP(16, 8).
func NewANNMLP(hidden ...int) *ANNMLP {
	h := append([]int(nil), hidden...)
	if len(h) == 0 {
		h = []int{16}
	}
	return &ANNMLP{
		HiddenLayers: h,
		Activation:   Tanh,
		LearningRate: 0.1,
		Epochs:       1000,
		Seed:         1,
	}
}

// Train fits the network weights by full-batch back-propagation. It returns an
// error if the input is empty, ragged, or the label count does not match.
func (m *ANNMLP) Train(samples [][]float64, labels []int) error {
	dim, err := validateSamples(samples, len(labels))
	if err != nil {
		return err
	}
	if m.LearningRate <= 0 {
		m.LearningRate = 0.1
	}
	if m.Epochs <= 0 {
		m.Epochs = 1000
	}
	var index map[int]int
	m.classes, index = classInfo(labels)
	k := len(m.classes)
	m.dim = dim

	m.scaler = fitScaler(samples)
	X := make([][]float64, len(samples))
	for i, s := range samples {
		X[i] = m.scaler.transform(s)
	}
	target := make([]int, len(labels))
	for i, l := range labels {
		target[i] = index[l]
	}

	m.layerSizes = make([]int, 0, len(m.HiddenLayers)+2)
	m.layerSizes = append(m.layerSizes, dim)
	m.layerSizes = append(m.layerSizes, m.HiddenLayers...)
	m.layerSizes = append(m.layerSizes, k)

	rng := rand.New(rand.NewSource(m.Seed))
	L := len(m.layerSizes) - 1
	m.weights = make([][][]float64, L)
	m.biases = make([][]float64, L)
	for t := 0; t < L; t++ {
		fanIn := m.layerSizes[t]
		fanOut := m.layerSizes[t+1]
		limit := math.Sqrt(6.0 / float64(fanIn+fanOut))
		m.weights[t] = make([][]float64, fanOut)
		m.biases[t] = make([]float64, fanOut)
		for o := 0; o < fanOut; o++ {
			m.weights[t][o] = make([]float64, fanIn)
			for i := 0; i < fanIn; i++ {
				m.weights[t][o][i] = (rng.Float64()*2 - 1) * limit
			}
		}
	}

	n := float64(len(X))
	for epoch := 0; epoch < m.Epochs; epoch++ {
		gradW := make([][][]float64, L)
		gradB := make([][]float64, L)
		for t := 0; t < L; t++ {
			gradW[t] = make([][]float64, m.layerSizes[t+1])
			gradB[t] = make([]float64, m.layerSizes[t+1])
			for o := range gradW[t] {
				gradW[t][o] = make([]float64, m.layerSizes[t])
			}
		}
		for i, x := range X {
			acts := m.forward(x)
			deltas := make([][]float64, L+1)
			out := acts[L]
			d := make([]float64, len(out))
			for o := range out {
				d[o] = out[o]
				if o == target[i] {
					d[o] -= 1
				}
			}
			deltas[L] = d
			for layer := L - 1; layer >= 1; layer-- {
				dl := make([]float64, m.layerSizes[layer])
				for j := range dl {
					var s float64
					for o := 0; o < m.layerSizes[layer+1]; o++ {
						s += m.weights[layer][o][j] * deltas[layer+1][o]
					}
					dl[j] = s * actDeriv(acts[layer][j], m.Activation)
				}
				deltas[layer] = dl
			}
			for t := 0; t < L; t++ {
				dt := deltas[t+1]
				a := acts[t]
				for o := range dt {
					gradB[t][o] += dt[o]
					row := gradW[t][o]
					for j := range a {
						row[j] += dt[o] * a[j]
					}
				}
			}
		}
		scale := m.LearningRate / n
		for t := 0; t < L; t++ {
			for o := range m.weights[t] {
				for j := range m.weights[t][o] {
					m.weights[t][o][j] -= scale * gradW[t][o][j]
				}
				m.biases[t][o] -= scale * gradB[t][o]
			}
		}
	}
	m.trained = true
	return nil
}

// forward propagates a standardised input through the network and returns the
// activation vector at every layer, with acts[0] the input and acts[len-1] the
// softmax output.
func (m *ANNMLP) forward(x []float64) [][]float64 {
	L := len(m.layerSizes) - 1
	acts := make([][]float64, L+1)
	acts[0] = x
	for t := 0; t < L; t++ {
		z := make([]float64, m.layerSizes[t+1])
		for o := range z {
			s := m.biases[t][o]
			w := m.weights[t][o]
			a := acts[t]
			for i := range a {
				s += w[i] * a[i]
			}
			z[o] = s
		}
		if t == L-1 {
			acts[t+1] = softmax(z)
		} else {
			acts[t+1] = applyActivation(z, m.Activation)
		}
	}
	return acts
}

// applyActivation returns act applied element-wise to z.
func applyActivation(z []float64, act Activation) []float64 {
	out := make([]float64, len(z))
	for i, v := range z {
		switch act {
		case Tanh:
			out[i] = math.Tanh(v)
		default:
			out[i] = 1.0 / (1.0 + math.Exp(-v))
		}
	}
	return out
}

// actDeriv returns the activation derivative expressed in terms of the
// post-activation value a (sigmoid: a(1-a); tanh: 1-a²).
func actDeriv(a float64, act Activation) float64 {
	if act == Tanh {
		return 1 - a*a
	}
	return a * (1 - a)
}

// Probabilities returns the softmax class probabilities for a sample, in sorted
// class order (see [ANNMLP.Classes]). It panics if the model is untrained or the
// sample has the wrong length.
func (m *ANNMLP) Probabilities(sample []float64) []float64 {
	if !m.trained {
		panic(ErrNotTrained)
	}
	if len(sample) != m.dim {
		panic("ml: ANNMLP sample has wrong feature count")
	}
	acts := m.forward(m.scaler.transform(sample))
	return acts[len(acts)-1]
}

// Classes returns the sorted set of class labels the network was trained on.
func (m *ANNMLP) Classes() []int {
	return append([]int(nil), m.classes...)
}

// Predict classifies a single sample. It panics if the model is untrained or the
// sample has the wrong length.
func (m *ANNMLP) Predict(sample []float64) int {
	return m.classes[argmax(m.Probabilities(sample))]
}

// PredictBatch classifies every sample and returns the predicted labels in
// order.
func (m *ANNMLP) PredictBatch(samples [][]float64) []int {
	out := make([]int, len(samples))
	for i, s := range samples {
		out[i] = m.Predict(s)
	}
	return out
}
