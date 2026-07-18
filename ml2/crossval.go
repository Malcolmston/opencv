package ml2

import "math/rand"

// KFoldSplit partitions the indices [0, n) into k folds for cross-validation.
// The assignment is a reproducible shuffle seeded by seed; fold sizes differ by
// at most one. It returns a slice of k index slices and panics if k < 2 or
// k > n.
func KFoldSplit(n, k int, seed int64) [][]int {
	if k < 2 {
		panic("ml2: KFoldSplit requires k >= 2")
	}
	if k > n {
		panic("ml2: KFoldSplit requires k <= n")
	}
	perm := rand.New(rand.NewSource(seed)).Perm(n)
	folds := make([][]int, k)
	for i, idx := range perm {
		f := i % k
		folds[f] = append(folds[f], idx)
	}
	return folds
}

// CrossValScore runs k-fold cross-validation and returns the accuracy on each
// held-out fold. newModel is called once per fold to obtain a fresh, untrained
// [Classifier], which is trained on the other folds and scored on the held-out
// one. The split is seeded by seed for reproducibility. It panics if k < 2 or
// k exceeds the number of samples, and skips a fold (scoring it 0) only if
// model training returns an error.
func CrossValScore(newModel func() Classifier, samples [][]float64, labels []int, k int, seed int64) []float64 {
	folds := KFoldSplit(len(samples), k, seed)
	scores := make([]float64, k)
	for fi := 0; fi < k; fi++ {
		test := folds[fi]
		var trainX [][]float64
		var trainY []int
		for fj := 0; fj < k; fj++ {
			if fj == fi {
				continue
			}
			for _, idx := range folds[fj] {
				trainX = append(trainX, samples[idx])
				trainY = append(trainY, labels[idx])
			}
		}
		model := newModel()
		if err := model.Fit(trainX, trainY); err != nil {
			scores[fi] = 0
			continue
		}
		yTrue := make([]int, len(test))
		yPred := make([]int, len(test))
		for i, idx := range test {
			yTrue[i] = labels[idx]
			yPred[i] = model.Predict(samples[idx])
		}
		scores[fi] = Accuracy(yTrue, yPred)
	}
	return scores
}

// MeanScore returns the arithmetic mean of a slice of scores, or zero for an
// empty slice. It is a convenience for summarising [CrossValScore] output.
func MeanScore(scores []float64) float64 {
	if len(scores) == 0 {
		return 0
	}
	var s float64
	for _, v := range scores {
		s += v
	}
	return s / float64(len(scores))
}
