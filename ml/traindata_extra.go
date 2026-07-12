package ml

import (
	"math/rand"
	"sort"
)

// Fold is one train/test partition produced by [TrainData.KFold].
type Fold struct {
	// Train holds the samples used for fitting in this fold.
	Train *TrainData
	// Test holds the held-out samples used for evaluation in this fold.
	Test *TrainData
}

// gather builds a TrainData from the receiver's rows selected by idx, carrying
// Labels and Responses along when present.
func (td *TrainData) gather(idx []int) *TrainData {
	out := &TrainData{}
	hasLabels := td.Labels != nil
	hasResp := td.Responses != nil
	for _, i := range idx {
		out.Samples = append(out.Samples, td.Samples[i])
		if hasLabels {
			out.Labels = append(out.Labels, td.Labels[i])
		}
		if hasResp {
			out.Responses = append(out.Responses, td.Responses[i])
		}
	}
	return out
}

// KFold partitions the data into k folds for cross-validation. It returns k
// [Fold]s; in fold i the i-th contiguous block of a seeded shuffle is the test
// set and the remaining k-1 blocks form the training set, so every sample is
// tested exactly once across the folds. The shuffle is reproducible for a fixed
// seed. It panics if k is not in [2, Len].
func (td *TrainData) KFold(k int, seed int64) []Fold {
	n := len(td.Samples)
	if k < 2 || k > n {
		panic("ml: KFold requires 2 <= k <= number of samples")
	}
	perm := rand.New(rand.NewSource(seed)).Perm(n)
	folds := make([]Fold, k)
	for f := 0; f < k; f++ {
		start := f * n / k
		end := (f + 1) * n / k
		var testIdx, trainIdx []int
		for i, idx := range perm {
			if i >= start && i < end {
				testIdx = append(testIdx, idx)
			} else {
				trainIdx = append(trainIdx, idx)
			}
		}
		folds[f] = Fold{Train: td.gather(trainIdx), Test: td.gather(testIdx)}
	}
	return folds
}

// StratifiedSplit partitions labelled data into a training and a test set while
// preserving each class's proportion in both parts. Within every class a seeded
// shuffle assigns a trainRatio fraction of that class's samples to the training
// set. It requires Labels to be set and panics otherwise.
func (td *TrainData) StratifiedSplit(trainRatio float64, seed int64) (train, test *TrainData) {
	if td.Labels == nil {
		panic("ml: StratifiedSplit requires labels")
	}
	byClass := make(map[int][]int)
	var order []int
	for i, l := range td.Labels {
		if _, ok := byClass[l]; !ok {
			order = append(order, l)
		}
		byClass[l] = append(byClass[l], i)
	}
	sort.Ints(order)
	rng := rand.New(rand.NewSource(seed))
	var trainIdx, testIdx []int
	for _, l := range order {
		idx := byClass[l]
		rng.Shuffle(len(idx), func(a, b int) { idx[a], idx[b] = idx[b], idx[a] })
		nTrain := int(float64(len(idx)) * trainRatio)
		trainIdx = append(trainIdx, idx[:nTrain]...)
		testIdx = append(testIdx, idx[nTrain:]...)
	}
	return td.gather(trainIdx), td.gather(testIdx)
}

// Shuffle returns a new TrainData with the samples (and their labels/responses)
// randomly permuted by a seeded generator. The receiver is left unchanged.
func (td *TrainData) Shuffle(seed int64) *TrainData {
	perm := rand.New(rand.NewSource(seed)).Perm(len(td.Samples))
	return td.gather(perm)
}
