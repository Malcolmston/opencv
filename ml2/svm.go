package ml2

import (
	"errors"
	"math"
	"math/rand"
)

// KernelType selects the kernel function used by an [SVM].
type KernelType int

const (
	// LinearKernel is the dot product k(x, z) = x·z.
	LinearKernel KernelType = iota
	// PolyKernel is the polynomial kernel k(x, z) = (gamma·x·z + coef0)^degree.
	PolyKernel
	// RBFKernel is the Gaussian radial-basis kernel
	// k(x, z) = exp(-gamma·‖x − z‖²).
	RBFKernel
)

// SVMParams configures the training of an [SVM].
type SVMParams struct {
	// Kernel selects the kernel function.
	Kernel KernelType
	// C is the soft-margin regularisation strength (larger C fits harder).
	C float64
	// Gamma scales the polynomial and RBF kernels.
	Gamma float64
	// Degree is the polynomial-kernel degree.
	Degree float64
	// Coef0 is the polynomial-kernel constant term.
	Coef0 float64
	// Tol is the KKT tolerance used by the SMO stopping rule.
	Tol float64
	// MaxPasses is the number of consecutive full passes without an update
	// after which SMO stops.
	MaxPasses int
	// Seed makes the working-set selection deterministic.
	Seed int64
}

// DefaultSVMParams returns reasonable defaults for the given kernel: C = 1,
// Gamma = 0.5, Degree = 3, Coef0 = 1, Tol = 1e-3, MaxPasses = 5.
func DefaultSVMParams(kernel KernelType) SVMParams {
	return SVMParams{
		Kernel:    kernel,
		C:         1,
		Gamma:     0.5,
		Degree:    3,
		Coef0:     1,
		Tol:       1e-3,
		MaxPasses: 5,
		Seed:      1,
	}
}

// SVM is a kernel support-vector-machine classifier. Binary problems are solved
// directly with Platt's Sequential Minimal Optimisation (SMO); problems with
// more than two classes are handled with a one-vs-rest ensemble of binary
// machines, and the class of greatest decision value wins.
type SVM struct {
	params  SVMParams
	classes int
	binary  []*ml2binarySVM
}

// NewSVM returns an untrained SVM using the supplied parameters.
func NewSVM(params SVMParams) *SVM {
	return &SVM{params: params}
}

// kernelFunc evaluates the configured kernel on two vectors.
func (p SVMParams) kernelFunc(a, b []float64) float64 {
	switch p.Kernel {
	case PolyKernel:
		return math.Pow(p.Gamma*ml2dot(a, b)+p.Coef0, p.Degree)
	case RBFKernel:
		return math.Exp(-p.Gamma * ml2squaredEuclidean(a, b))
	default:
		return ml2dot(a, b)
	}
}

// Fit trains the SVM. It returns an error for empty or mismatched input, or for
// a degenerate problem with fewer than two classes.
func (m *SVM) Fit(samples [][]float64, labels []int) error {
	if len(samples) == 0 {
		return errors.New("ml2: SVM.Fit given no samples")
	}
	if len(samples) != len(labels) {
		return errors.New("ml2: SVM.Fit requires len(samples) == len(labels)")
	}
	m.classes = ml2numClasses(labels)
	if m.classes < 2 {
		return errors.New("ml2: SVM.Fit requires at least two classes")
	}
	if m.classes == 2 {
		// Single binary machine: class 1 is positive, class 0 negative.
		y := make([]float64, len(labels))
		for i, l := range labels {
			if l == 1 {
				y[i] = 1
			} else {
				y[i] = -1
			}
		}
		bs := newBinarySVM(m.params)
		if err := bs.fit(samples, y); err != nil {
			return err
		}
		m.binary = []*ml2binarySVM{bs}
		return nil
	}
	// One-vs-rest.
	m.binary = make([]*ml2binarySVM, m.classes)
	for c := 0; c < m.classes; c++ {
		y := make([]float64, len(labels))
		for i, l := range labels {
			if l == c {
				y[i] = 1
			} else {
				y[i] = -1
			}
		}
		bs := newBinarySVM(m.params)
		if err := bs.fit(samples, y); err != nil {
			return err
		}
		m.binary[c] = bs
	}
	return nil
}

// DecisionFunction returns the signed decision value(s) for sample. For a
// binary model it returns a single value whose sign gives the class; for a
// multiclass model it returns one value per class (the one-vs-rest scores). It
// panics before Fit.
func (m *SVM) DecisionFunction(sample []float64) []float64 {
	if m.binary == nil {
		panic("ml2: SVM.DecisionFunction before Fit")
	}
	if m.classes == 2 {
		return []float64{m.binary[0].decision(sample)}
	}
	out := make([]float64, m.classes)
	for c := range m.binary {
		out[c] = m.binary[c].decision(sample)
	}
	return out
}

// Predict returns the predicted class for sample. It panics before Fit.
func (m *SVM) Predict(sample []float64) int {
	d := m.DecisionFunction(sample)
	if m.classes == 2 {
		if d[0] >= 0 {
			return 1
		}
		return 0
	}
	return ml2argmax(d)
}

// PredictBatch classifies every sample in x.
func (m *SVM) PredictBatch(x [][]float64) []int {
	out := make([]int, len(x))
	for i, s := range x {
		out[i] = m.Predict(s)
	}
	return out
}

// NumSupportVectors returns the total number of support vectors retained across
// all internal binary machines. It panics before Fit.
func (m *SVM) NumSupportVectors() int {
	if m.binary == nil {
		panic("ml2: SVM.NumSupportVectors before Fit")
	}
	total := 0
	for _, b := range m.binary {
		total += len(b.svAlphaY)
	}
	return total
}

// ml2binarySVM is a two-class SMO-trained kernel machine with labels in ±1.
type ml2binarySVM struct {
	params SVMParams
	// Retained support vectors and their alpha*y products.
	svX      [][]float64
	svAlphaY []float64
	b        float64
}

func newBinarySVM(p SVMParams) *ml2binarySVM { return &ml2binarySVM{params: p} }

// fit runs the simplified SMO of Platt as presented in the CS229 notes, with a
// seeded random second index for determinism.
func (m *ml2binarySVM) fit(x [][]float64, y []float64) error {
	n := len(x)
	alpha := make([]float64, n)
	var b float64
	C := m.params.C
	tol := m.params.Tol
	rng := rand.New(rand.NewSource(m.params.Seed))

	// Precompute the kernel matrix (n is small for the KAT tests and typical
	// classic-CV feature sets).
	kmat := make([][]float64, n)
	for i := 0; i < n; i++ {
		kmat[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			kmat[i][j] = m.params.kernelFunc(x[i], x[j])
		}
	}

	f := func(i int) float64 {
		var s float64
		for k := 0; k < n; k++ {
			if alpha[k] != 0 {
				s += alpha[k] * y[k] * kmat[k][i]
			}
		}
		return s + b
	}

	passes := 0
	for passes < m.params.MaxPasses {
		numChanged := 0
		for i := 0; i < n; i++ {
			Ei := f(i) - y[i]
			if (y[i]*Ei < -tol && alpha[i] < C) || (y[i]*Ei > tol && alpha[i] > 0) {
				j := rng.Intn(n - 1)
				if j >= i {
					j++
				}
				Ej := f(j) - y[j]
				ai, aj := alpha[i], alpha[j]
				var L, H float64
				if y[i] != y[j] {
					L = math.Max(0, aj-ai)
					H = math.Min(C, C+aj-ai)
				} else {
					L = math.Max(0, ai+aj-C)
					H = math.Min(C, ai+aj)
				}
				if L == H {
					continue
				}
				eta := 2*kmat[i][j] - kmat[i][i] - kmat[j][j]
				if eta >= 0 {
					continue
				}
				ajNew := aj - y[j]*(Ei-Ej)/eta
				if ajNew > H {
					ajNew = H
				} else if ajNew < L {
					ajNew = L
				}
				if math.Abs(ajNew-aj) < 1e-7 {
					continue
				}
				aiNew := ai + y[i]*y[j]*(aj-ajNew)
				b1 := b - Ei - y[i]*(aiNew-ai)*kmat[i][i] - y[j]*(ajNew-aj)*kmat[i][j]
				b2 := b - Ej - y[i]*(aiNew-ai)*kmat[i][j] - y[j]*(ajNew-aj)*kmat[j][j]
				switch {
				case aiNew > 0 && aiNew < C:
					b = b1
				case ajNew > 0 && ajNew < C:
					b = b2
				default:
					b = (b1 + b2) / 2
				}
				alpha[i], alpha[j] = aiNew, ajNew
				numChanged++
			}
		}
		if numChanged == 0 {
			passes++
		} else {
			passes = 0
		}
	}

	// Retain support vectors (alpha > 0).
	for i := 0; i < n; i++ {
		if alpha[i] > 1e-8 {
			m.svX = append(m.svX, x[i])
			m.svAlphaY = append(m.svAlphaY, alpha[i]*y[i])
		}
	}
	m.b = b
	return nil
}

// decision returns the raw decision value for a query point.
func (m *ml2binarySVM) decision(sample []float64) float64 {
	var s float64
	for i := range m.svX {
		s += m.svAlphaY[i] * m.params.kernelFunc(m.svX[i], sample)
	}
	return s + m.b
}
