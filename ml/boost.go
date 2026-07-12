package ml

import (
	"math"
	"sort"
)

// Boost is a boosted ensemble of decision stumps, mirroring OpenCV's
// cv::ml::Boost. It implements the multiclass AdaBoost.SAMME algorithm: it fits a
// sequence of depth-one trees (stumps), each on a reweighted view of the data
// that emphasises the samples its predecessors got wrong, and combines them with
// weights proportional to their accuracy. With two classes SAMME reduces to the
// classic discrete AdaBoost.
//
// Construct with [NewBoost]. Set NEstimators before calling Train.
type Boost struct {
	// NEstimators is the maximum number of stumps to fit. Boosting may stop
	// early if a perfect stump is found or no stump beats random guessing.
	NEstimators int

	stumps  []stump
	alphas  []float64
	classes []int
	index   map[int]int
	dim     int
	trained bool
}

// stump is a single-split decision stump: samples with feature Feature at or
// below Threshold receive LeftClass, the rest RightClass. Fields are exported so
// the ensemble can be serialised with encoding/gob.
type stump struct {
	Feature    int
	Threshold  float64
	LeftClass  int
	RightClass int
}

// predict returns the class label the stump assigns to sample.
func (s stump) predict(sample []float64) int {
	if sample[s.Feature] <= s.Threshold {
		return s.LeftClass
	}
	return s.RightClass
}

// NewBoost returns a boosting ensemble that fits up to nEstimators stumps. It
// panics if nEstimators is not positive.
func NewBoost(nEstimators int) *Boost {
	if nEstimators <= 0 {
		panic("ml: NewBoost requires nEstimators > 0")
	}
	return &Boost{NEstimators: nEstimators}
}

// Train fits the boosted ensemble with AdaBoost.SAMME. It returns an error if
// the input is empty, ragged, or the label count does not match.
func (m *Boost) Train(samples [][]float64, labels []int) error {
	dim, err := validateSamples(samples, len(labels))
	if err != nil {
		return err
	}
	if m.NEstimators <= 0 {
		m.NEstimators = 50
	}
	m.classes, m.index = classInfo(labels)
	m.dim = dim
	k := len(m.classes)
	n := len(samples)

	weights := make([]float64, n)
	for i := range weights {
		weights[i] = 1.0 / float64(n)
	}
	m.stumps = m.stumps[:0]
	m.alphas = m.alphas[:0]

	logK1 := math.Log(float64(k - 1))
	for t := 0; t < m.NEstimators; t++ {
		st, errRate := fitStump(samples, labels, weights, dim, m.classes, m.index)
		// Guard the logarithm against a flawless or hopeless stump.
		if errRate <= 0 {
			errRate = 1e-12
		}
		alpha := math.Log((1-errRate)/errRate) + logK1
		// A stump no better than random (alpha <= 0) adds nothing; stop once we
		// already have at least one learner.
		if alpha <= 0 && len(m.stumps) > 0 {
			break
		}
		m.stumps = append(m.stumps, st)
		m.alphas = append(m.alphas, alpha)

		var norm float64
		for i := range weights {
			if st.predict(samples[i]) != labels[i] {
				weights[i] *= math.Exp(alpha)
			}
			norm += weights[i]
		}
		for i := range weights {
			weights[i] /= norm
		}
		if errRate <= 1e-12 {
			// The stump already classifies every weighted sample correctly.
			break
		}
	}
	m.trained = true
	return nil
}

// fitStump finds the weighted-error-minimising decision stump. For each feature
// it sweeps the sorted sample order, tracking the weighted class distribution on
// either side of the split, and picks the threshold whose majority-class leaves
// leave the least weight misclassified.
func fitStump(samples [][]float64, labels []int, weights []float64, dim int, classes []int, index map[int]int) (best stump, bestErr float64) {
	k := len(classes)
	bestErr = math.Inf(1)
	for f := 0; f < dim; f++ {
		order := make([]int, len(samples))
		for i := range order {
			order[i] = i
		}
		sort.SliceStable(order, func(a, b int) bool {
			return samples[order[a]][f] < samples[order[b]][f]
		})
		left := make([]float64, k)
		right := make([]float64, k)
		for _, i := range order {
			right[index[labels[i]]] += weights[i]
		}
		for p := 0; p < len(order)-1; p++ {
			i := order[p]
			c := index[labels[i]]
			left[c] += weights[i]
			right[c] -= weights[i]
			v1 := samples[order[p]][f]
			v2 := samples[order[p+1]][f]
			if v1 == v2 {
				continue
			}
			li, lw := argmaxWeighted(left)
			ri, rw := argmaxWeighted(right)
			// Total weight is 1, so the misclassified weight is what the two
			// majority leaves fail to capture.
			errRate := 1 - (lw + rw)
			if errRate < bestErr {
				bestErr = errRate
				best = stump{
					Feature:    f,
					Threshold:  (v1 + v2) / 2,
					LeftClass:  classes[li],
					RightClass: classes[ri],
				}
			}
		}
	}
	if math.IsInf(bestErr, 1) {
		// No feature could be split (all samples identical); fall back to the
		// weighted-majority constant classifier.
		total := make([]float64, k)
		for i := range samples {
			total[index[labels[i]]] += weights[i]
		}
		mi, mw := argmaxWeighted(total)
		best = stump{Feature: 0, Threshold: math.Inf(1), LeftClass: classes[mi], RightClass: classes[mi]}
		bestErr = 1 - mw
	}
	return best, bestErr
}

// argmaxWeighted returns the index and value of the largest element in w, with
// ties broken toward the lowest index.
func argmaxWeighted(w []float64) (idx int, val float64) {
	idx = 0
	val = w[0]
	for i := 1; i < len(w); i++ {
		if w[i] > val {
			val = w[i]
			idx = i
		}
	}
	return idx, val
}

// Classes returns the sorted set of class labels the ensemble was trained on.
func (m *Boost) Classes() []int {
	return append([]int(nil), m.classes...)
}

// DecisionScores returns the accumulated stump weight backing each class, in
// sorted class order (see [Boost.Classes]). It panics if the model is untrained
// or the sample has the wrong length.
func (m *Boost) DecisionScores(sample []float64) []float64 {
	if !m.trained {
		panic(ErrNotTrained)
	}
	if len(sample) != m.dim {
		panic("ml: Boost sample has wrong feature count")
	}
	scores := make([]float64, len(m.classes))
	for t, st := range m.stumps {
		scores[m.index[st.predict(sample)]] += m.alphas[t]
	}
	return scores
}

// Predict classifies a single sample as the class with the greatest summed
// stump weight. It panics if the model is untrained or the sample has the wrong
// length.
func (m *Boost) Predict(sample []float64) int {
	return m.classes[argmax(m.DecisionScores(sample))]
}

// PredictBatch classifies every sample and returns the predicted labels in
// order.
func (m *Boost) PredictBatch(samples [][]float64) []int {
	out := make([]int, len(samples))
	for i, s := range samples {
		out[i] = m.Predict(s)
	}
	return out
}
