package ml

import (
	"math"
	"math/rand"
	"sort"
)

// RTrees is a random forest of CART classification trees, mirroring OpenCV's
// cv::ml::RTrees. Each tree is grown on a bootstrap resample of the training set
// (bagging) and, at every node, considers only a random subset of the features
// when searching for the best split (feature subsampling). Predictions are the
// majority vote across all trees, which lowers variance relative to a single
// [DecisionTree].
//
// Because roughly a third of the samples are left out of each tree's bootstrap
// bag, RTrees also estimates its generalisation error for free: the
// out-of-bag (OOB) error, available from [RTrees.OOBError] after training.
//
// Construct with [NewRTrees]. Tune the exported fields before calling Train.
type RTrees struct {
	// NTrees is the number of trees in the forest; it must be positive.
	NTrees int
	// MaxDepth bounds each tree's height. Zero or negative means unbounded.
	MaxDepth int
	// MinSamplesSplit is the fewest samples a node must hold to be eligible for
	// splitting. Values below 2 are treated as 2.
	MinSamplesSplit int
	// MaxFeatures is the number of features sampled as split candidates at each
	// node. Zero or negative selects floor(sqrt(numFeatures)), the usual default
	// for classification forests.
	MaxFeatures int
	// Seed makes bootstrap resampling and feature subsampling reproducible.
	Seed int64

	trees    []*treeNode
	classes  []int
	index    map[int]int
	dim      int
	oobError float64
	trained  bool
}

// NewRTrees returns a random forest with nTrees trees and otherwise sensible
// defaults (unbounded depth, sqrt feature subsampling). It panics if nTrees is
// not positive.
func NewRTrees(nTrees int) *RTrees {
	if nTrees <= 0 {
		panic("ml: NewRTrees requires nTrees > 0")
	}
	return &RTrees{NTrees: nTrees, MinSamplesSplit: 2, Seed: 1}
}

// Train grows the forest. It returns an error if the input is empty, ragged, or
// the label count does not match. After a successful call the out-of-bag error
// estimate is available from [RTrees.OOBError].
func (m *RTrees) Train(samples [][]float64, labels []int) error {
	dim, err := validateSamples(samples, len(labels))
	if err != nil {
		return err
	}
	if m.NTrees <= 0 {
		m.NTrees = 100
	}
	if m.MinSamplesSplit < 2 {
		m.MinSamplesSplit = 2
	}
	maxFeatures := m.MaxFeatures
	if maxFeatures <= 0 {
		maxFeatures = int(math.Sqrt(float64(dim)))
		if maxFeatures < 1 {
			maxFeatures = 1
		}
	}
	if maxFeatures > dim {
		maxFeatures = dim
	}

	m.classes, m.index = classInfo(labels)
	m.dim = dim
	n := len(samples)
	m.trees = make([]*treeNode, 0, m.NTrees)

	// oobVotes[i][c] counts the trees that left sample i out of their bag and
	// then classified it as the c-th class.
	oobVotes := make([][]int, n)
	for i := range oobVotes {
		oobVotes[i] = make([]int, len(m.classes))
	}

	for t := 0; t < m.NTrees; t++ {
		rng := rand.New(rand.NewSource(m.Seed + int64(t)))
		bag := make([]int, n)
		inBag := make([]bool, n)
		for i := 0; i < n; i++ {
			j := rng.Intn(n)
			bag[i] = j
			inBag[j] = true
		}
		root := buildRFTree(samples, labels, bag, 0, m.MaxDepth, m.MinSamplesSplit, dim, maxFeatures, rng)
		m.trees = append(m.trees, root)
		for i := 0; i < n; i++ {
			if !inBag[i] {
				pred := predictTree(root, samples[i])
				oobVotes[i][m.index[pred]]++
			}
		}
	}

	// Out-of-bag error: aggregate each sample's OOB votes and compare the
	// majority to the true label. Samples that were never out-of-bag (rare) are
	// skipped.
	var counted, wrong int
	for i := 0; i < n; i++ {
		total := 0
		for _, c := range oobVotes[i] {
			total += c
		}
		if total == 0 {
			continue
		}
		counted++
		if m.classes[argmaxInt(oobVotes[i])] != labels[i] {
			wrong++
		}
	}
	if counted > 0 {
		m.oobError = float64(wrong) / float64(counted)
	}
	m.trained = true
	return nil
}

// buildRFTree grows a single randomized CART tree over the sample indices in
// idx, considering maxFeatures randomly chosen features at each split.
func buildRFTree(samples [][]float64, labels, idx []int, depth, maxDepth, minSplit, dim, maxFeatures int, rng *rand.Rand) *treeNode {
	majority, pure := majorityClass(labels, idx)
	if pure || len(idx) < minSplit || (maxDepth > 0 && depth >= maxDepth) {
		return &treeNode{leaf: true, prediction: majority}
	}
	feature, threshold, ok := rfBestSplit(samples, labels, idx, dim, maxFeatures, rng)
	if !ok {
		return &treeNode{leaf: true, prediction: majority}
	}
	var leftIdx, rightIdx []int
	for _, i := range idx {
		if samples[i][feature] <= threshold {
			leftIdx = append(leftIdx, i)
		} else {
			rightIdx = append(rightIdx, i)
		}
	}
	if len(leftIdx) == 0 || len(rightIdx) == 0 {
		return &treeNode{leaf: true, prediction: majority}
	}
	return &treeNode{
		feature:   feature,
		threshold: threshold,
		left:      buildRFTree(samples, labels, leftIdx, depth+1, maxDepth, minSplit, dim, maxFeatures, rng),
		right:     buildRFTree(samples, labels, rightIdx, depth+1, maxDepth, minSplit, dim, maxFeatures, rng),
	}
}

// rfBestSplit searches maxFeatures randomly selected features for the split that
// most reduces the weighted child Gini impurity.
func rfBestSplit(samples [][]float64, labels, idx []int, dim, maxFeatures int, rng *rand.Rand) (feature int, threshold float64, ok bool) {
	perm := rng.Perm(dim)
	bestGini := giniImpurity(labels, idx)
	for fi := 0; fi < maxFeatures; fi++ {
		f := perm[fi]
		order := append([]int(nil), idx...)
		sort.SliceStable(order, func(a, b int) bool {
			return samples[order[a]][f] < samples[order[b]][f]
		})
		for p := 0; p < len(order)-1; p++ {
			v1 := samples[order[p]][f]
			v2 := samples[order[p+1]][f]
			if v1 == v2 {
				continue
			}
			thr := (v1 + v2) / 2
			g := weightedGini(samples, labels, idx, f, thr)
			if g < bestGini {
				bestGini = g
				feature = f
				threshold = thr
				ok = true
			}
		}
	}
	return feature, threshold, ok
}

// predictTree routes sample down node's decision path and returns the leaf's
// stored class label.
func predictTree(node *treeNode, sample []float64) int {
	for !node.leaf {
		if sample[node.feature] <= node.threshold {
			node = node.left
		} else {
			node = node.right
		}
	}
	return node.prediction
}

// argmaxInt returns the index of the largest element of v, breaking ties toward
// the lowest index. It panics on an empty slice.
func argmaxInt(v []int) int {
	if len(v) == 0 {
		panic("ml: argmaxInt of empty slice")
	}
	best := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

// OOBError returns the out-of-bag misclassification rate estimated during Train,
// a value in [0, 1]. It panics if the model is untrained.
func (m *RTrees) OOBError() float64 {
	if !m.trained {
		panic(ErrNotTrained)
	}
	return m.oobError
}

// Classes returns the sorted set of class labels the forest was trained on.
func (m *RTrees) Classes() []int {
	return append([]int(nil), m.classes...)
}

// Predict classifies a single sample by majority vote across all trees. It
// panics if the model is untrained or the sample has the wrong length.
func (m *RTrees) Predict(sample []float64) int {
	if !m.trained {
		panic(ErrNotTrained)
	}
	if len(sample) != m.dim {
		panic("ml: RTrees sample has wrong feature count")
	}
	votes := make([]int, len(m.classes))
	for _, root := range m.trees {
		votes[m.index[predictTree(root, sample)]]++
	}
	return m.classes[argmaxInt(votes)]
}

// PredictBatch classifies every sample and returns the predicted labels in
// order.
func (m *RTrees) PredictBatch(samples [][]float64) []int {
	out := make([]int, len(samples))
	for i, s := range samples {
		out[i] = m.Predict(s)
	}
	return out
}
