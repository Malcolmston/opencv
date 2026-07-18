package ml2

import (
	"errors"
	"math/rand"
)

// ml2node is a single node of a CART decision tree. A leaf has feature == -1
// and carries a class label; an internal node splits on feature <= threshold.
type ml2node struct {
	feature   int
	threshold float64
	label     int
	left      *ml2node
	right     *ml2node
}

// DecisionTree is a CART classification tree that splits to minimise the Gini
// impurity. Growth is bounded by a maximum depth and a minimum number of
// samples required to attempt a split; leaves predict the majority class of the
// samples that reach them.
type DecisionTree struct {
	maxDepth        int
	minSamplesSplit int
	classes         int
	root            *ml2node

	// maxFeatures, when > 0, restricts each split to a random subset of that
	// many features; rng supplies the randomness. Both are set internally by
	// RandomForest and are zero-valued (all features, deterministic) otherwise.
	maxFeatures int
	rng         *rand.Rand
}

// NewDecisionTree returns an untrained tree. maxDepth <= 0 means unlimited
// depth; minSamplesSplit is clamped up to 2.
func NewDecisionTree(maxDepth, minSamplesSplit int) *DecisionTree {
	if minSamplesSplit < 2 {
		minSamplesSplit = 2
	}
	return &DecisionTree{maxDepth: maxDepth, minSamplesSplit: minSamplesSplit}
}

// gini returns the Gini impurity of a label multiset described by class counts.
func ml2gini(counts []int, total int) float64 {
	if total == 0 {
		return 0
	}
	imp := 1.0
	ft := float64(total)
	for _, c := range counts {
		p := float64(c) / ft
		imp -= p * p
	}
	return imp
}

// Fit grows the tree. It returns an error for empty or mismatched input.
func (t *DecisionTree) Fit(samples [][]float64, labels []int) error {
	if len(samples) == 0 {
		return errors.New("ml2: DecisionTree.Fit given no samples")
	}
	if len(samples) != len(labels) {
		return errors.New("ml2: DecisionTree.Fit requires len(samples) == len(labels)")
	}
	t.classes = ml2numClasses(labels)
	idx := make([]int, len(samples))
	for i := range idx {
		idx[i] = i
	}
	t.root = t.build(samples, labels, idx, 0)
	return nil
}

// classCounts tallies labels of the referenced rows.
func (t *DecisionTree) classCounts(labels []int, idx []int) ([]int, int) {
	counts := make([]int, t.classes)
	best := 0
	for _, i := range idx {
		counts[labels[i]]++
	}
	for c, n := range counts {
		if n > counts[best] {
			best = c
		}
	}
	return counts, best
}

// featureSubset returns the feature indices considered for a split at a node.
func (t *DecisionTree) featureSubset(d int) []int {
	if t.maxFeatures <= 0 || t.maxFeatures >= d || t.rng == nil {
		all := make([]int, d)
		for i := range all {
			all[i] = i
		}
		return all
	}
	perm := t.rng.Perm(d)
	return perm[:t.maxFeatures]
}

// build recursively constructs a subtree over the referenced rows.
func (t *DecisionTree) build(samples [][]float64, labels, idx []int, depth int) *ml2node {
	counts, majority := t.classCounts(labels, idx)
	node := &ml2node{feature: -1, label: majority}

	// Stop on purity, depth, or too-few samples.
	pure := true
	for _, c := range counts {
		if c != 0 && c != len(idx) {
			pure = false
			break
		}
	}
	if pure || len(idx) < t.minSamplesSplit || (t.maxDepth > 0 && depth >= t.maxDepth) {
		return node
	}

	d := len(samples[0])
	parentImp := ml2gini(counts, len(idx))
	total := float64(len(idx))
	bestGain := 0.0
	bestFeat := -1
	var bestThr float64

	for _, feat := range t.featureSubset(d) {
		vals := make([]float64, len(idx))
		for k, i := range idx {
			vals[k] = samples[i][feat]
		}
		uniq := ml2sortedUnique(vals)
		if len(uniq) < 2 {
			continue
		}
		for u := 0; u+1 < len(uniq); u++ {
			thr := (uniq[u] + uniq[u+1]) / 2
			lc := make([]int, t.classes)
			rc := make([]int, t.classes)
			ln, rn := 0, 0
			for _, i := range idx {
				if samples[i][feat] <= thr {
					lc[labels[i]]++
					ln++
				} else {
					rc[labels[i]]++
					rn++
				}
			}
			if ln == 0 || rn == 0 {
				continue
			}
			childImp := (float64(ln)/total)*ml2gini(lc, ln) + (float64(rn)/total)*ml2gini(rc, rn)
			gain := parentImp - childImp
			if gain > bestGain {
				bestGain, bestFeat, bestThr = gain, feat, thr
			}
		}
	}

	if bestFeat < 0 {
		return node
	}

	var leftIdx, rightIdx []int
	for _, i := range idx {
		if samples[i][bestFeat] <= bestThr {
			leftIdx = append(leftIdx, i)
		} else {
			rightIdx = append(rightIdx, i)
		}
	}
	node.feature = bestFeat
	node.threshold = bestThr
	node.left = t.build(samples, labels, leftIdx, depth+1)
	node.right = t.build(samples, labels, rightIdx, depth+1)
	return node
}

// Predict returns the class of the leaf that sample falls into. It panics
// before Fit.
func (t *DecisionTree) Predict(sample []float64) int {
	if t.root == nil {
		panic("ml2: DecisionTree.Predict before Fit")
	}
	n := t.root
	for n.feature >= 0 {
		if sample[n.feature] <= n.threshold {
			n = n.left
		} else {
			n = n.right
		}
	}
	return n.label
}

// PredictBatch classifies every sample in x.
func (t *DecisionTree) PredictBatch(x [][]float64) []int {
	out := make([]int, len(x))
	for i, s := range x {
		out[i] = t.Predict(s)
	}
	return out
}

// Depth returns the number of edges on the longest root-to-leaf path. A tree
// with only a root leaf has depth zero. It panics before Fit.
func (t *DecisionTree) Depth() int {
	if t.root == nil {
		panic("ml2: DecisionTree.Depth before Fit")
	}
	var depth func(n *ml2node) int
	depth = func(n *ml2node) int {
		if n == nil || n.feature < 0 {
			return 0
		}
		l := depth(n.left)
		r := depth(n.right)
		if l > r {
			return l + 1
		}
		return r + 1
	}
	return depth(t.root)
}
