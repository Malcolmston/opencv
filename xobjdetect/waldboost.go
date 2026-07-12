package xobjdetect

import (
	"errors"
	"math"
	"sort"
)

// Stump is a confidence-rated decision stump: the atomic weak classifier of a
// [WaldBoost] ensemble. It probes one entry of the feature vector and emits
// +Alpha or -Alpha depending on which side of Threshold the value falls. Fields
// are exported for gob serialisation.
type Stump struct {
	// Feature is the index into the feature vector this stump reads.
	Feature int
	// Threshold is the decision boundary on that feature value.
	Threshold float64
	// Polarity is +1 or -1. When +1 a value >= Threshold votes for the positive
	// class; when -1 the sense is reversed.
	Polarity int
	// Alpha is the stump's boosting weight (its vote magnitude).
	Alpha float64
}

// vote returns the stump's signed contribution to the ensemble score for the
// given feature value: +Alpha for a positive vote, -Alpha for a negative one.
func (s Stump) vote(value float64) float64 {
	side := -1
	if value >= s.Threshold {
		side = 1
	}
	if s.Polarity < 0 {
		side = -side
	}
	return float64(side) * s.Alpha
}

// WaldBoost is a WaldBoost classifier: a discrete-AdaBoost ensemble of
// [Stump]s equipped with a sequential-probability-ratio-test (SPRT) early-exit
// cascade. Training fits Rounds stumps and, after each one, records a rejection
// threshold on the running score; prediction accumulates that score stump by
// stump and rejects the moment it falls below the current threshold.
//
// Construct with [NewWaldBoost], set Rounds and (optionally) DetectionRate, and
// call [WaldBoost.Train]. The zero value is not usable.
type WaldBoost struct {
	// Rounds is the maximum number of stumps to fit.
	Rounds int
	// DetectionRate is the fraction of positive training samples the SPRT
	// thresholds must retain at every stage, in (0,1]. Values outside that range
	// default to 0.995. Lower values prune more aggressively.
	DetectionRate float64

	// Stumps is the fitted ensemble, in evaluation order.
	Stumps []Stump
	// SPRT holds one rejection threshold per stump: after evaluating stump i the
	// running score must be >= SPRT[i] or the sample is rejected.
	SPRT []float64
	// Trained reports whether Train has completed successfully.
	Trained bool
}

// NewWaldBoost returns a WaldBoost that fits up to rounds stumps. It panics if
// rounds is not positive.
func NewWaldBoost(rounds int) *WaldBoost {
	if rounds <= 0 {
		panic("xobjdetect: NewWaldBoost requires rounds > 0")
	}
	return &WaldBoost{Rounds: rounds, DetectionRate: 0.995}
}

// Train fits the ensemble from positive and negative feature vectors. pos and
// neg are slices of equal-length feature vectors (one per sample). It returns an
// error if either class is empty or the vectors are ragged.
//
// Each round fits the stump with the lowest weighted error over all features,
// assigns it an AdaBoost weight, reweights the samples, and records an SPRT
// rejection threshold chosen so that at least DetectionRate of the positive
// samples survive the cascade up to that point.
func (w *WaldBoost) Train(pos, neg [][]float64) error {
	if len(pos) == 0 || len(neg) == 0 {
		return errors.New("xobjdetect: WaldBoost.Train needs both positive and negative samples")
	}
	dim := len(pos[0])
	if dim == 0 {
		return errors.New("xobjdetect: WaldBoost.Train got zero-length feature vectors")
	}
	n := len(pos) + len(neg)
	X := make([][]float64, 0, n)
	y := make([]int, 0, n)
	for _, v := range pos {
		if len(v) != dim {
			return errors.New("xobjdetect: WaldBoost.Train ragged feature vectors")
		}
		X = append(X, v)
		y = append(y, 1)
	}
	for _, v := range neg {
		if len(v) != dim {
			return errors.New("xobjdetect: WaldBoost.Train ragged feature vectors")
		}
		X = append(X, v)
		y = append(y, -1)
	}

	detRate := w.DetectionRate
	if detRate <= 0 || detRate > 1 {
		detRate = 0.995
	}

	// Balanced initial weights so neither class dominates.
	weights := make([]float64, n)
	nPos, nNeg := len(pos), len(neg)
	for i := 0; i < n; i++ {
		if y[i] > 0 {
			weights[i] = 0.5 / float64(nPos)
		} else {
			weights[i] = 0.5 / float64(nNeg)
		}
	}

	w.Stumps = w.Stumps[:0]
	w.SPRT = w.SPRT[:0]
	running := make([]float64, n) // cumulative ensemble score per sample

	for t := 0; t < w.Rounds; t++ {
		st, errRate := fitBestStump(X, y, weights, dim)
		errRate = math.Max(errRate, 1e-10)
		if errRate >= 0.5 {
			// No stump beats chance on the current weights; stop.
			break
		}
		alpha := 0.5 * math.Log((1-errRate)/errRate)
		st.Alpha = alpha
		w.Stumps = append(w.Stumps, st)

		// Reweight and accumulate the running score.
		var norm float64
		for i := 0; i < n; i++ {
			v := st.vote(X[i][st.Feature])
			running[i] += v
			// v has magnitude alpha; sign matches the class prediction.
			hi := 1.0
			if v < 0 {
				hi = -1.0
			}
			weights[i] *= math.Exp(-alpha * float64(y[i]) * hi)
			norm += weights[i]
		}
		if norm > 0 {
			for i := range weights {
				weights[i] /= norm
			}
		}

		// SPRT threshold: keep at least detRate of the positives.
		posScores := make([]float64, 0, nPos)
		for i := 0; i < n; i++ {
			if y[i] > 0 {
				posScores = append(posScores, running[i])
			}
		}
		sort.Float64s(posScores)
		k := int(math.Floor((1 - detRate) * float64(nPos)))
		if k >= nPos {
			k = nPos - 1
		}
		w.SPRT = append(w.SPRT, posScores[k]-1e-9)
	}

	if len(w.Stumps) == 0 {
		return errors.New("xobjdetect: WaldBoost.Train fit no usable stump")
	}
	w.Trained = true
	return nil
}

// Predict runs the SPRT cascade over feat and returns the accumulated score and
// whether the sample was accepted. A sample is rejected the first time the
// running score drops below the corresponding SPRT threshold; a sample that
// survives every stump is accepted when its final score is positive. The score
// is the raw ensemble margin and doubles as a confidence: larger is more
// confidently positive. It panics if the classifier is untrained.
func (w *WaldBoost) Predict(feat []float64) (score float64, accepted bool) {
	if !w.Trained {
		panic("xobjdetect: WaldBoost.Predict on untrained classifier")
	}
	for i, st := range w.Stumps {
		score += st.vote(feat[st.Feature])
		if score < w.SPRT[i] {
			return score, false
		}
	}
	return score, score > 0
}

// fitBestStump finds the decision stump with the lowest weighted classification
// error over every feature and returns it (without Alpha set) and that error.
func fitBestStump(X [][]float64, y []int, weights []float64, dim int) (Stump, float64) {
	n := len(X)
	var wPos, wNeg float64
	for i := 0; i < n; i++ {
		if y[i] > 0 {
			wPos += weights[i]
		} else {
			wNeg += weights[i]
		}
	}

	best := Stump{Polarity: 1}
	bestErr := math.Inf(1)
	idx := make([]int, n)

	for f := 0; f < dim; f++ {
		for i := 0; i < n; i++ {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return X[idx[a]][f] < X[idx[b]][f] })

		var leftPos, leftNeg float64
		consider := func(threshold float64) {
			// Polarity +1: value >= threshold -> positive.
			//   errors = positives on the left + negatives on the right.
			errP := leftPos + (wNeg - leftNeg)
			// Polarity -1: value >= threshold -> negative.
			errM := leftNeg + (wPos - leftPos)
			if errP < bestErr {
				bestErr = errP
				best = Stump{Feature: f, Threshold: threshold, Polarity: 1}
			}
			if errM < bestErr {
				bestErr = errM
				best = Stump{Feature: f, Threshold: threshold, Polarity: -1}
			}
		}

		// Split before the smallest value: everything is on the right.
		consider(math.Inf(-1))
		for k := 0; k < n; k++ {
			i := idx[k]
			if y[i] > 0 {
				leftPos += weights[i]
			} else {
				leftNeg += weights[i]
			}
			var threshold float64
			if k+1 < n {
				vk := X[idx[k]][f]
				vk1 := X[idx[k+1]][f]
				if vk == vk1 {
					// Cannot split between equal values; wait for a change.
					continue
				}
				threshold = (vk + vk1) / 2
			} else {
				threshold = math.Inf(1)
			}
			consider(threshold)
		}
	}
	return best, bestErr
}
