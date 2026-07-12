package ml

import "sort"

// DecisionTree is a CART classification tree. It is grown greedily by choosing,
// at each node, the feature and threshold whose binary split most reduces the
// Gini impurity, stopping when a node is pure, too small to split, or the
// maximum depth is reached. Prediction routes a sample down the learned
// decision path to a leaf and returns that leaf's majority class.
//
// Construct with [NewDecisionTree]. The zero value is not usable.
type DecisionTree struct {
	// MaxDepth bounds the tree height. Zero or negative means unbounded.
	MaxDepth int
	// MinSamplesSplit is the fewest samples a node must hold to be eligible
	// for splitting; nodes below it become leaves. Values below 2 are treated
	// as 2.
	MinSamplesSplit int

	root    *treeNode
	classes []int
	dim     int
	trained bool
}

// treeNode is one node of a fitted DecisionTree. Leaves carry a prediction;
// internal nodes carry a split.
type treeNode struct {
	leaf       bool
	prediction int
	feature    int
	threshold  float64
	left       *treeNode
	right      *treeNode
}

// NewDecisionTree returns a CART tree bounded to maxDepth (0 for unbounded).
func NewDecisionTree(maxDepth int) *DecisionTree {
	return &DecisionTree{MaxDepth: maxDepth, MinSamplesSplit: 2}
}

// Train grows the tree from samples and labels. It returns an error if the
// input is empty, ragged, or the label count does not match.
func (m *DecisionTree) Train(samples [][]float64, labels []int) error {
	dim, err := validateSamples(samples, len(labels))
	if err != nil {
		return err
	}
	if m.MinSamplesSplit < 2 {
		m.MinSamplesSplit = 2
	}
	m.classes, _ = classInfo(labels)
	m.dim = dim
	// Work on index slices so splitting does not copy feature vectors.
	idx := make([]int, len(samples))
	for i := range idx {
		idx[i] = i
	}
	m.root = m.build(samples, labels, idx, 0)
	m.trained = true
	return nil
}

// build recursively constructs a subtree over the sample indices in idx.
func (m *DecisionTree) build(samples [][]float64, labels, idx []int, depth int) *treeNode {
	majority, pure := majorityClass(labels, idx)
	if pure || len(idx) < m.MinSamplesSplit || (m.MaxDepth > 0 && depth >= m.MaxDepth) {
		return &treeNode{leaf: true, prediction: majority}
	}
	feature, threshold, ok := m.bestSplit(samples, labels, idx)
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
		left:      m.build(samples, labels, leftIdx, depth+1),
		right:     m.build(samples, labels, rightIdx, depth+1),
	}
}

// bestSplit searches every feature and candidate threshold for the split that
// minimises the weighted child Gini impurity. It reports ok=false when no split
// separates the samples.
func (m *DecisionTree) bestSplit(samples [][]float64, labels, idx []int) (feature int, threshold float64, ok bool) {
	bestGini := giniImpurity(labels, idx)
	for f := 0; f < m.dim; f++ {
		// Candidate thresholds are midpoints between consecutive distinct
		// feature values, evaluated in sorted order.
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

// weightedGini returns the sample-weighted Gini impurity of the two children
// produced by splitting idx on feature f at threshold thr.
func weightedGini(samples [][]float64, labels, idx []int, f int, thr float64) float64 {
	var left, right []int
	for _, i := range idx {
		if samples[i][f] <= thr {
			left = append(left, i)
		} else {
			right = append(right, i)
		}
	}
	n := float64(len(idx))
	gl := giniImpurity(labels, left)
	gr := giniImpurity(labels, right)
	return (float64(len(left))*gl + float64(len(right))*gr) / n
}

// giniImpurity returns the Gini impurity of the labels selected by idx.
func giniImpurity(labels, idx []int) float64 {
	if len(idx) == 0 {
		return 0
	}
	counts := make(map[int]int, 8)
	for _, i := range idx {
		counts[labels[i]]++
	}
	n := float64(len(idx))
	var sum float64
	for _, c := range counts {
		p := float64(c) / n
		sum += p * p
	}
	return 1 - sum
}

// majorityClass returns the most frequent label among idx (ties broken by the
// lowest label value) and whether the set is pure (a single label).
func majorityClass(labels, idx []int) (label int, pure bool) {
	counts := make(map[int]int, 8)
	for _, i := range idx {
		counts[labels[i]]++
	}
	best, bestCount := 0, -1
	// Iterate over a sorted key set for deterministic tie-breaking.
	keys := make([]int, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Ints(keys)
	for _, k := range keys {
		if counts[k] > bestCount {
			bestCount = counts[k]
			best = k
		}
	}
	return best, len(counts) == 1
}

// Predict classifies a single sample. It panics if the model is untrained or
// the sample has the wrong length.
func (m *DecisionTree) Predict(sample []float64) int {
	if !m.trained {
		panic(ErrNotTrained)
	}
	if len(sample) != m.dim {
		panic("ml: DecisionTree sample has wrong feature count")
	}
	node := m.root
	for !node.leaf {
		if sample[node.feature] <= node.threshold {
			node = node.left
		} else {
			node = node.right
		}
	}
	return node.prediction
}

// PredictBatch classifies every sample and returns the predicted labels in
// order.
func (m *DecisionTree) PredictBatch(samples [][]float64) []int {
	out := make([]int, len(samples))
	for i, s := range samples {
		out[i] = m.Predict(s)
	}
	return out
}
