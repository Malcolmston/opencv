package ml

import (
	"math"
	"math/rand"
)

// KernelType selects the kernel function of a [KernelSVM].
type KernelType int

const (
	// LinearKernel is K(x,y) = x·y and recovers an ordinary linear SVM.
	LinearKernel KernelType = iota
	// RBFKernel is the Gaussian radial basis function
	// K(x,y) = exp(-gamma·‖x-y‖²); it can separate non-linear data.
	RBFKernel
	// PolyKernel is the polynomial kernel K(x,y) = (gamma·x·y + coef0)^degree.
	PolyKernel
)

// KernelSVM is a kernelised support-vector machine, extending the linear [SVM]
// with non-linear decision boundaries via the kernel trick. It supports the
// linear, Gaussian RBF and polynomial kernels, and handles more than two classes
// with a one-vs-rest ensemble. Training uses the kernel Pegasos algorithm, which
// maintains, for each class, integer coefficients over the training points and
// converges to the same regularised hinge-loss solution as the linear method in
// the kernel's feature space.
//
// Features are standardised internally, which is especially important for the
// RBF kernel. A fixed Seed makes training reproducible. Construct with
// [NewKernelSVM] and tune the exported fields before calling Train.
type KernelSVM struct {
	// Kernel selects the kernel function.
	Kernel KernelType
	// Gamma scales the RBF and polynomial kernels. Zero or negative selects the
	// heuristic 1/numFeatures.
	Gamma float64
	// Degree is the exponent of the polynomial kernel; values below 1 are
	// treated as 2.
	Degree int
	// Coef0 is the additive constant of the polynomial kernel.
	Coef0 float64
	// Lambda is the L2 regularisation strength; it must be positive.
	Lambda float64
	// Epochs is the number of passes over the training set (the Pegasos horizon
	// is Epochs·numSamples steps).
	Epochs int
	// Seed makes the internal sample shuffling reproducible.
	Seed int64

	scaler  *scaler
	sv      [][]float64 // scaled support (all training) vectors
	coef    [][]float64 // [class][i]: signed weight of training point i
	classes []int
	dim     int
	gamma   float64
	degree  int
	trained bool
}

// NewKernelSVM returns a kernel SVM using the given kernel with sensible
// defaults (Lambda 1e-3, 50 epochs, degree 3, coef0 1). The default Coef0 of 1
// makes the polynomial kernel inhomogeneous so it retains lower-order terms.
func NewKernelSVM(kernel KernelType) *KernelSVM {
	return &KernelSVM{
		Kernel: kernel,
		Degree: 3,
		Coef0:  1,
		Lambda: 1e-3,
		Epochs: 50,
		Seed:   1,
	}
}

// Train fits one kernel-Pegasos binary machine per class in a one-vs-rest
// scheme. It returns an error if the input is empty, ragged, or the label count
// does not match.
func (m *KernelSVM) Train(samples [][]float64, labels []int) error {
	dim, err := validateSamples(samples, len(labels))
	if err != nil {
		return err
	}
	if m.Lambda <= 0 {
		m.Lambda = 1e-3
	}
	if m.Epochs <= 0 {
		m.Epochs = 50
	}
	m.degree = m.Degree
	if m.degree < 1 {
		m.degree = 2
	}
	m.gamma = m.Gamma
	if m.gamma <= 0 {
		m.gamma = 1.0 / float64(dim)
	}
	m.dim = dim
	m.scaler = fitScaler(samples)
	m.sv = make([][]float64, len(samples))
	for i, s := range samples {
		m.sv[i] = m.scaler.transform(s)
	}
	m.classes, _ = classInfo(labels)

	// Precompute the kernel Gram matrix once; it is reused by every one-vs-rest
	// machine and by every Pegasos step.
	n := len(m.sv)
	gram := make([][]float64, n)
	for i := 0; i < n; i++ {
		gram[i] = make([]float64, n)
		for j := 0; j <= i; j++ {
			k := m.kernel(m.sv[i], m.sv[j])
			gram[i][j] = k
			gram[j][i] = k
		}
	}

	m.coef = make([][]float64, len(m.classes))
	for ci, class := range m.classes {
		y := make([]float64, len(labels))
		for i, l := range labels {
			if l == class {
				y[i] = 1
			} else {
				y[i] = -1
			}
		}
		m.coef[ci] = m.kernelPegasos(gram, y, int64(ci))
	}
	m.trained = true
	return nil
}

// kernelPegasos trains one binary machine in the dual. alpha[i] accumulates the
// (signed) number of times point i is used as a support vector; the decision
// function is (1/(lambda·t))·Σ alpha_j·y_j·K(x_j, x).
func (m *KernelSVM) kernelPegasos(gram [][]float64, y []float64, offset int64) []float64 {
	n := len(y)
	alpha := make([]float64, n)
	rng := rand.New(rand.NewSource(m.Seed + offset))
	t := 0
	for epoch := 0; epoch < m.Epochs; epoch++ {
		for step := 0; step < n; step++ {
			t++
			i := rng.Intn(n)
			var sum float64
			for j := 0; j < n; j++ {
				if alpha[j] != 0 {
					sum += alpha[j] * y[j] * gram[j][i]
				}
			}
			f := sum / (m.Lambda * float64(t))
			if y[i]*f < 1 {
				alpha[i]++
			}
		}
	}
	// Fold the final 1/(lambda·T) scale into the coefficients so prediction is
	// a plain kernel expansion.
	scale := 1.0 / (m.Lambda * float64(t))
	coef := make([]float64, n)
	for i := range alpha {
		coef[i] = alpha[i] * y[i] * scale
	}
	return coef
}

// kernel evaluates the configured kernel on two standardised vectors.
func (m *KernelSVM) kernel(a, b []float64) float64 {
	switch m.Kernel {
	case RBFKernel:
		return math.Exp(-m.gamma * squaredEuclidean(a, b))
	case PolyKernel:
		return math.Pow(m.gamma*dot(a, b)+m.Coef0, float64(m.degree))
	default:
		return dot(a, b)
	}
}

// DecisionScores returns the one-vs-rest decision value for every class, in
// sorted class order (see [KernelSVM.Classes]). It panics if the model is
// untrained or the sample has the wrong length.
func (m *KernelSVM) DecisionScores(sample []float64) []float64 {
	if !m.trained {
		panic(ErrNotTrained)
	}
	if len(sample) != m.dim {
		panic("ml: KernelSVM sample has wrong feature count")
	}
	scaled := m.scaler.transform(sample)
	kv := make([]float64, len(m.sv))
	for j, sv := range m.sv {
		kv[j] = m.kernel(sv, scaled)
	}
	out := make([]float64, len(m.classes))
	for ci := range m.classes {
		coef := m.coef[ci]
		var s float64
		for j := range coef {
			if coef[j] != 0 {
				s += coef[j] * kv[j]
			}
		}
		out[ci] = s
	}
	return out
}

// Classes returns the sorted set of class labels the model was trained on.
func (m *KernelSVM) Classes() []int {
	return append([]int(nil), m.classes...)
}

// Predict classifies a single sample as the highest-scoring one-vs-rest class.
// It panics if the model is untrained or the sample has the wrong length.
func (m *KernelSVM) Predict(sample []float64) int {
	return m.classes[argmax(m.DecisionScores(sample))]
}

// PredictBatch classifies every sample and returns the predicted labels in
// order.
func (m *KernelSVM) PredictBatch(samples [][]float64) []int {
	out := make([]int, len(samples))
	for i, s := range samples {
		out[i] = m.Predict(s)
	}
	return out
}
