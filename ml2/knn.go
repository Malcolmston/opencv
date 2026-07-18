package ml2

import (
	"errors"
	"sort"
)

// KNN is a k-nearest-neighbours classifier. It stores the training set and, for
// each query, votes among the k closest samples by Euclidean distance. It is a
// lazy learner: Fit merely records the data.
type KNN struct {
	k       int
	samples [][]float64
	labels  []int
	classes int
}

// NewKNN returns a KNN classifier using k neighbours. It panics if k < 1.
func NewKNN(k int) *KNN {
	if k < 1 {
		panic("ml2: NewKNN requires k >= 1")
	}
	return &KNN{k: k}
}

// Fit records the training samples and labels. It returns an error if the
// inputs are empty or of mismatched length.
func (m *KNN) Fit(samples [][]float64, labels []int) error {
	if len(samples) == 0 {
		return errors.New("ml2: KNN.Fit given no samples")
	}
	if len(samples) != len(labels) {
		return errors.New("ml2: KNN.Fit requires len(samples) == len(labels)")
	}
	m.samples = samples
	m.labels = labels
	m.classes = ml2numClasses(labels)
	return nil
}

// neighbourDist pairs a training index with its distance to a query.
type ml2neighbour struct {
	idx  int
	dist float64
}

// KNeighbors returns the indices of the k nearest training samples to the query
// and their Euclidean distances, sorted nearest first. Ties in distance are
// broken toward the lower training index. It panics before Fit.
func (m *KNN) KNeighbors(sample []float64) (indices []int, dists []float64) {
	if m.samples == nil {
		panic("ml2: KNN.KNeighbors before Fit")
	}
	ns := make([]ml2neighbour, len(m.samples))
	for i, s := range m.samples {
		ns[i] = ml2neighbour{idx: i, dist: ml2euclidean(sample, s)}
	}
	sort.SliceStable(ns, func(a, b int) bool {
		if ns[a].dist != ns[b].dist {
			return ns[a].dist < ns[b].dist
		}
		return ns[a].idx < ns[b].idx
	})
	k := m.k
	if k > len(ns) {
		k = len(ns)
	}
	indices = make([]int, k)
	dists = make([]float64, k)
	for i := 0; i < k; i++ {
		indices[i] = ns[i].idx
		dists[i] = ns[i].dist
	}
	return indices, dists
}

// Predict returns the majority class among the k nearest neighbours of sample,
// breaking ties toward the lower class index. It panics before Fit.
func (m *KNN) Predict(sample []float64) int {
	idx, _ := m.KNeighbors(sample)
	votes := make([]int, len(idx))
	for i, id := range idx {
		votes[i] = m.labels[id]
	}
	return ml2majority(votes, m.classes)
}

// PredictBatch classifies every sample in x, returning one label each.
func (m *KNN) PredictBatch(x [][]float64) []int {
	out := make([]int, len(x))
	for i, s := range x {
		out[i] = m.Predict(s)
	}
	return out
}
