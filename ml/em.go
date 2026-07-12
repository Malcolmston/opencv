package ml

import "math"

// GaussianMixture is a Gaussian mixture model fitted by the
// Expectation-Maximization (EM) algorithm, mirroring OpenCV's cv::ml::EM. It
// models the data as a weighted sum of K multivariate Gaussians with diagonal
// covariance matrices. Training alternates the E-step (computing each sample's
// posterior responsibility for every component) with the M-step (re-estimating
// the mixture weights, means and variances), monotonically increasing the data
// log-likelihood until it converges or MaxIter is reached.
//
// The components are seeded with a short [KMeans] run, so a fixed Seed makes the
// whole fit reproducible. GaussianMixture is unsupervised: [GaussianMixture.Fit]
// takes only feature vectors, and [GaussianMixture.Predict] returns the index of
// the most probable component (a soft-clustering assignment).
//
// Construct with [NewGaussianMixture].
type GaussianMixture struct {
	// K is the number of mixture components; it must be positive.
	K int
	// MaxIter bounds the number of EM iterations.
	MaxIter int
	// Tol is the minimum per-iteration increase in average log-likelihood
	// required to keep iterating; below it the fit is deemed converged.
	Tol float64
	// Seed makes the KMeans seeding reproducible.
	Seed int64

	weights       []float64
	means         [][]float64
	variances     [][]float64 // diagonal covariance
	dim           int
	logLikelihood float64
	trained       bool
}

// NewGaussianMixture returns an EM model with k components and sensible defaults
// (100 iterations, tolerance 1e-6). It panics if k is not positive.
func NewGaussianMixture(k int) *GaussianMixture {
	if k <= 0 {
		panic("ml: NewGaussianMixture requires k > 0")
	}
	return &GaussianMixture{K: k, MaxIter: 100, Tol: 1e-6, Seed: 1}
}

// Fit estimates the mixture parameters from unlabelled data with EM. It returns
// an error if the data is empty or ragged, or if there are fewer samples than
// components.
func (m *GaussianMixture) Fit(data [][]float64) error {
	dim, err := validateSamples(data, len(data))
	if err != nil {
		return err
	}
	if m.K > len(data) {
		return ErrNoSamples
	}
	if m.MaxIter <= 0 {
		m.MaxIter = 100
	}
	m.dim = dim
	n := len(data)

	// Seed the components with KMeans: cluster means become the initial means,
	// per-cluster spread the initial variances, cluster sizes the weights.
	labels, centers := KMeans(data, m.K, 50, m.Seed)
	m.means = centers
	m.variances = make([][]float64, m.K)
	m.weights = make([]float64, m.K)
	counts := make([]int, m.K)
	for c := 0; c < m.K; c++ {
		m.variances[c] = make([]float64, dim)
	}
	for i, s := range data {
		c := labels[i]
		counts[c]++
		for j := range s {
			d := s[j] - centers[c][j]
			m.variances[c][j] += d * d
		}
	}
	floor := varianceFloor(data)
	for c := 0; c < m.K; c++ {
		m.weights[c] = float64(counts[c]) / float64(n)
		if m.weights[c] == 0 {
			m.weights[c] = 1.0 / float64(n)
		}
		for j := 0; j < dim; j++ {
			if counts[c] > 0 {
				m.variances[c][j] /= float64(counts[c])
			}
			m.variances[c][j] += floor
		}
	}

	resp := make([][]float64, n)
	for i := range resp {
		resp[i] = make([]float64, m.K)
	}

	prevLL := math.Inf(-1)
	for iter := 0; iter < m.MaxIter; iter++ {
		// E-step: responsibilities via log-sum-exp for numerical stability.
		var totalLL float64
		logW := make([]float64, m.K)
		for c := range logW {
			logW[c] = math.Log(m.weights[c])
		}
		for i, s := range data {
			logp := make([]float64, m.K)
			for c := 0; c < m.K; c++ {
				logp[c] = logW[c] + logGaussianDiag(s, m.means[c], m.variances[c])
			}
			maxLp := logp[0]
			for _, v := range logp[1:] {
				if v > maxLp {
					maxLp = v
				}
			}
			var sum float64
			for c := range logp {
				logp[c] = math.Exp(logp[c] - maxLp)
				sum += logp[c]
			}
			for c := range logp {
				resp[i][c] = logp[c] / sum
			}
			totalLL += maxLp + math.Log(sum)
		}
		m.logLikelihood = totalLL

		// M-step.
		for c := 0; c < m.K; c++ {
			var nc float64
			for i := 0; i < n; i++ {
				nc += resp[i][c]
			}
			if nc < 1e-12 {
				nc = 1e-12
			}
			mean := make([]float64, dim)
			for i, s := range data {
				r := resp[i][c]
				for j := range s {
					mean[j] += r * s[j]
				}
			}
			for j := range mean {
				mean[j] /= nc
			}
			variance := make([]float64, dim)
			for i, s := range data {
				r := resp[i][c]
				for j := range s {
					d := s[j] - mean[j]
					variance[j] += r * d * d
				}
			}
			for j := range variance {
				variance[j] = variance[j]/nc + floor
			}
			m.means[c] = mean
			m.variances[c] = variance
			m.weights[c] = nc / float64(n)
		}

		if iter > 0 && totalLL-prevLL < m.Tol*math.Abs(prevLL) {
			break
		}
		prevLL = totalLL
	}
	m.trained = true
	return nil
}

// varianceFloor returns a small floor added to every variance to keep the
// Gaussians well-posed for tightly clustered data.
func varianceFloor(data [][]float64) float64 {
	sc := fitScaler(data)
	var maxVar float64
	for _, s := range sc.std {
		if v := s * s; v > maxVar {
			maxVar = v
		}
	}
	return 1e-6*maxVar + 1e-12
}

// logGaussianDiag returns the log density of x under a diagonal-covariance
// Gaussian with the given mean and per-dimension variance.
func logGaussianDiag(x, mean, variance []float64) float64 {
	var ll float64
	for j := range x {
		v := variance[j]
		d := x[j] - mean[j]
		ll += -0.5 * (math.Log(2*math.Pi*v) + d*d/v)
	}
	return ll
}

// Responsibilities returns the posterior probability that sample belongs to each
// component, in component order. It panics if the model is untrained or the
// sample has the wrong length.
func (m *GaussianMixture) Responsibilities(sample []float64) []float64 {
	m.checkQuery(sample)
	logp := make([]float64, m.K)
	for c := 0; c < m.K; c++ {
		logp[c] = math.Log(m.weights[c]) + logGaussianDiag(sample, m.means[c], m.variances[c])
	}
	maxLp := logp[0]
	for _, v := range logp[1:] {
		if v > maxLp {
			maxLp = v
		}
	}
	var sum float64
	for c := range logp {
		logp[c] = math.Exp(logp[c] - maxLp)
		sum += logp[c]
	}
	for c := range logp {
		logp[c] /= sum
	}
	return logp
}

// LogLikelihood returns the log probability density of sample under the fitted
// mixture. It panics if the model is untrained or the sample has the wrong
// length.
func (m *GaussianMixture) LogLikelihood(sample []float64) float64 {
	m.checkQuery(sample)
	logp := make([]float64, m.K)
	for c := 0; c < m.K; c++ {
		logp[c] = math.Log(m.weights[c]) + logGaussianDiag(sample, m.means[c], m.variances[c])
	}
	maxLp := logp[0]
	for _, v := range logp[1:] {
		if v > maxLp {
			maxLp = v
		}
	}
	var sum float64
	for _, v := range logp {
		sum += math.Exp(v - maxLp)
	}
	return maxLp + math.Log(sum)
}

// TotalLogLikelihood returns the summed data log-likelihood attained at the end
// of training. It panics if the model is untrained.
func (m *GaussianMixture) TotalLogLikelihood() float64 {
	if !m.trained {
		panic(ErrNotTrained)
	}
	return m.logLikelihood
}

// Means returns a copy of the fitted component means.
func (m *GaussianMixture) Means() [][]float64 {
	out := make([][]float64, len(m.means))
	for i, mu := range m.means {
		out[i] = append([]float64(nil), mu...)
	}
	return out
}

// Weights returns a copy of the fitted mixture weights.
func (m *GaussianMixture) Weights() []float64 {
	return append([]float64(nil), m.weights...)
}

// Predict returns the index of the component most responsible for sample, a
// soft-clustering assignment in [0, K). It panics if the model is untrained or
// the sample has the wrong length.
func (m *GaussianMixture) Predict(sample []float64) int {
	return argmax(m.Responsibilities(sample))
}

// PredictBatch assigns every sample to its most probable component.
func (m *GaussianMixture) PredictBatch(samples [][]float64) []int {
	out := make([]int, len(samples))
	for i, s := range samples {
		out[i] = m.Predict(s)
	}
	return out
}

func (m *GaussianMixture) checkQuery(sample []float64) {
	if !m.trained {
		panic(ErrNotTrained)
	}
	if len(sample) != m.dim {
		panic("ml: GaussianMixture sample has wrong feature count")
	}
}
