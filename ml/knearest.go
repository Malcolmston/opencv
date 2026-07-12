package ml

import (
	"math"
	"sort"
)

// KNearest is a k-nearest-neighbours classifier. Training simply memorises the
// samples; classification finds the K closest training samples (by Euclidean
// distance) to a query and returns the most common label among them. When
// Weighted is true, neighbours vote with weight 1/(distance²+ε) so that closer
// neighbours count for more; otherwise every neighbour casts an equal vote.
//
// Construct with [NewKNearest]. The zero value is not usable.
type KNearest struct {
	// K is the number of neighbours consulted per query; it must be positive.
	K int
	// Weighted selects inverse-distance-squared weighting over a plain
	// majority vote.
	Weighted bool

	samples [][]float64
	labels  []int
	classes []int
	dim     int
	trained bool
}

// NewKNearest returns a KNearest that consults k neighbours with a plain
// majority vote. It panics if k is not positive.
func NewKNearest(k int) *KNearest {
	if k <= 0 {
		panic("ml: NewKNearest requires k > 0")
	}
	return &KNearest{K: k}
}

// Train memorises the training samples and their labels. It returns an error if
// the input is empty, ragged, or the label count does not match.
func (m *KNearest) Train(samples [][]float64, labels []int) error {
	dim, err := validateSamples(samples, len(labels))
	if err != nil {
		return err
	}
	m.samples = make([][]float64, len(samples))
	for i, s := range samples {
		m.samples[i] = cloneSample(s)
	}
	m.labels = append([]int(nil), labels...)
	m.classes, _ = classInfo(labels)
	m.dim = dim
	m.trained = true
	return nil
}

// neighbour pairs a training index with its squared distance to a query.
type neighbour struct {
	dist2 float64
	label int
}

// FindNearest returns the labels and Euclidean distances of the k training
// samples closest to sample, ordered nearest first. k is clamped to the number
// of stored samples. It panics if the model is untrained or sample has the
// wrong length.
func (m *KNearest) FindNearest(sample []float64, k int) (labels []int, dists []float64) {
	m.checkQuery(sample)
	if k <= 0 {
		k = 1
	}
	nb := m.sortedNeighbours(sample)
	if k > len(nb) {
		k = len(nb)
	}
	labels = make([]int, k)
	dists = make([]float64, k)
	for i := 0; i < k; i++ {
		labels[i] = nb[i].label
		dists[i] = math.Sqrt(nb[i].dist2)
	}
	return labels, dists
}

// Predict classifies a single sample. It panics if the model is untrained or
// sample has the wrong length.
func (m *KNearest) Predict(sample []float64) int {
	m.checkQuery(sample)
	nb := m.sortedNeighbours(sample)
	k := m.K
	if k > len(nb) {
		k = len(nb)
	}
	votes := make(map[int]float64, k)
	for i := 0; i < k; i++ {
		w := 1.0
		if m.Weighted {
			w = 1.0 / (nb[i].dist2 + 1e-12)
		}
		votes[nb[i].label] += w
	}
	// Resolve ties deterministically by preferring the lowest label value.
	best := m.classes[0]
	bestVote := math.Inf(-1)
	for _, c := range m.classes {
		if v, ok := votes[c]; ok && v > bestVote {
			bestVote = v
			best = c
		}
	}
	return best
}

// PredictBatch classifies every sample and returns the predicted labels in
// order.
func (m *KNearest) PredictBatch(samples [][]float64) []int {
	out := make([]int, len(samples))
	for i, s := range samples {
		out[i] = m.Predict(s)
	}
	return out
}

// sortedNeighbours returns every stored sample paired with its squared distance
// to sample, ordered nearest first with ties broken by lower label for
// determinism.
func (m *KNearest) sortedNeighbours(sample []float64) []neighbour {
	nb := make([]neighbour, len(m.samples))
	for i, s := range m.samples {
		nb[i] = neighbour{dist2: squaredEuclidean(sample, s), label: m.labels[i]}
	}
	sort.SliceStable(nb, func(a, b int) bool {
		if nb[a].dist2 != nb[b].dist2 {
			return nb[a].dist2 < nb[b].dist2
		}
		return nb[a].label < nb[b].label
	})
	return nb
}

func (m *KNearest) checkQuery(sample []float64) {
	if !m.trained {
		panic(ErrNotTrained)
	}
	if len(sample) != m.dim {
		panic("ml: KNearest sample has wrong feature count")
	}
}
