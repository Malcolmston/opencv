package ml2

import (
	"errors"
	"math"
	"math/rand"
)

// RandomForest is a bootstrap-aggregated ensemble of CART decision trees. Each
// tree is grown on a bootstrap resample of the training data and, at every
// split, considers only a random subset of features. Predictions are decided by
// majority vote across the trees. All randomness is driven by the configured
// seed, so training is fully deterministic.
type RandomForest struct {
	nTrees      int
	maxDepth    int
	maxFeatures int
	seed        int64
	classes     int
	trees       []*DecisionTree
}

// NewRandomForest returns an untrained forest of nTrees trees, each grown to at
// most maxDepth (<= 0 for unlimited). maxFeatures is the number of features
// considered per split; pass 0 to use the floor(sqrt(numFeatures)) default. It
// panics if nTrees < 1.
func NewRandomForest(nTrees, maxDepth, maxFeatures int, seed int64) *RandomForest {
	if nTrees < 1 {
		panic("ml2: NewRandomForest requires nTrees >= 1")
	}
	return &RandomForest{nTrees: nTrees, maxDepth: maxDepth, maxFeatures: maxFeatures, seed: seed}
}

// Fit trains the ensemble. It returns an error for empty or mismatched input.
func (f *RandomForest) Fit(samples [][]float64, labels []int) error {
	if len(samples) == 0 {
		return errors.New("ml2: RandomForest.Fit given no samples")
	}
	if len(samples) != len(labels) {
		return errors.New("ml2: RandomForest.Fit requires len(samples) == len(labels)")
	}
	f.classes = ml2numClasses(labels)
	d := len(samples[0])
	mf := f.maxFeatures
	if mf <= 0 {
		mf = int(math.Sqrt(float64(d)))
		if mf < 1 {
			mf = 1
		}
	}
	n := len(samples)
	f.trees = make([]*DecisionTree, f.nTrees)
	for t := 0; t < f.nTrees; t++ {
		rng := rand.New(rand.NewSource(f.seed + int64(t)))
		bx := make([][]float64, n)
		by := make([]int, n)
		for i := 0; i < n; i++ {
			r := rng.Intn(n)
			bx[i] = samples[r]
			by[i] = labels[r]
		}
		tree := NewDecisionTree(f.maxDepth, 2)
		tree.maxFeatures = mf
		tree.rng = rng
		// Ensure every tree spans the full label space for consistent voting.
		tree.classes = f.classes
		if err := f.fitTree(tree, bx, by); err != nil {
			return err
		}
		f.trees[t] = tree
	}
	return nil
}

// fitTree grows a single tree while preserving the forest-wide class count.
func (f *RandomForest) fitTree(tree *DecisionTree, samples [][]float64, labels []int) error {
	idx := make([]int, len(samples))
	for i := range idx {
		idx[i] = i
	}
	tree.classes = f.classes
	tree.root = tree.build(samples, labels, idx, 0)
	return nil
}

// Predict returns the majority class across all trees for sample, breaking ties
// toward the lower class index. It panics before Fit.
func (f *RandomForest) Predict(sample []float64) int {
	if f.trees == nil {
		panic("ml2: RandomForest.Predict before Fit")
	}
	votes := make([]int, f.classes)
	for _, t := range f.trees {
		votes[t.Predict(sample)]++
	}
	best := 0
	for c := 1; c < f.classes; c++ {
		if votes[c] > votes[best] {
			best = c
		}
	}
	return best
}

// PredictBatch classifies every sample in x.
func (f *RandomForest) PredictBatch(x [][]float64) []int {
	out := make([]int, len(x))
	for i, s := range x {
		out[i] = f.Predict(s)
	}
	return out
}

// PredictProba returns the fraction of trees voting for each class, a simple
// ensemble probability estimate. It panics before Fit.
func (f *RandomForest) PredictProba(sample []float64) []float64 {
	if f.trees == nil {
		panic("ml2: RandomForest.PredictProba before Fit")
	}
	votes := make([]float64, f.classes)
	for _, t := range f.trees {
		votes[t.Predict(sample)]++
	}
	for c := range votes {
		votes[c] /= float64(len(f.trees))
	}
	return votes
}
