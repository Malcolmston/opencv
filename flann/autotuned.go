package flann

import "math/rand"

// Autotuning parameters.
const (
	autotuneSampleSize = 100 // queries sampled from the dataset to score configs
	autotuneK          = 5   // neighbours per query used when scoring precision
	autotuneTrees      = 4   // trees in the candidate k-d forest
)

// AutotunedIndex chooses an approximate search configuration automatically to
// meet a target precision, mirroring OpenCV's flann AutotunedIndex. At
// construction it builds a randomized k-d forest and a hierarchical k-means tree
// over the data, samples a set of queries from the dataset, computes their exact
// neighbours as ground truth, then searches for the smallest check budget at
// which one of the two structures reaches the requested precision. That winning
// structure and budget back every subsequent query. Because an unbounded search
// is exact, any target in [0, 1] is always achievable.
//
// The chosen configuration is fixed after construction, so an AutotunedIndex is
// safe for concurrent searches.
type AutotunedIndex struct {
	data   [][]float64
	forest *KDForestIndex
	kmeans *KMeansIndex
	exact  *LinearIndex[[]float64]

	algo     string // "kdforest", "kmeans" or "linear" (empty dataset)
	checks   int    // chosen budget (0 means exact)
	achieved float64
	target   float64
	seedVal  int64
}

// NewAutotunedIndex builds an index tuned to answer k-NN queries at about
// targetPrecision (clamped to [0, 1]), the expected fraction of true nearest
// neighbours returned. seed makes both the candidate structures and the query
// sampling reproducible. It panics if the dataset is ragged. An empty dataset is
// allowed and yields empty searches.
func NewAutotunedIndex(data [][]float64, targetPrecision float64, seed int64) *AutotunedIndex {
	validateFloatData(data, "NewAutotunedIndex")
	if targetPrecision < 0 {
		targetPrecision = 0
	}
	if targetPrecision > 1 {
		targetPrecision = 1
	}
	a := &AutotunedIndex{
		data:    data,
		exact:   NewLinearIndex(data),
		target:  targetPrecision,
		seedVal: seed,
	}
	if len(data) == 0 {
		a.algo = "linear"
		a.achieved = 1
		return a
	}
	a.forest = NewKDForestIndex(data, autotuneTrees, seed)
	a.kmeans = NewKMeansIndex(data, 0, 0, seed)
	a.tune(seed)
	return a
}

// tune samples queries, then scans candidate budgets in increasing order and
// locks in the first structure that reaches the target precision.
func (a *AutotunedIndex) tune(seed int64) {
	rng := rand.New(rand.NewSource(seed))
	m := autotuneSampleSize
	if m > len(a.data) {
		m = len(a.data)
	}
	perm := rng.Perm(len(a.data))[:m]
	queries := make([][]float64, m)
	for i, p := range perm {
		queries[i] = cloneVec(a.data[p])
	}
	k := autotuneK
	if k > len(a.data) {
		k = len(a.data)
	}

	// Candidate budgets in increasing cost, ending with 0 (unbounded/exact),
	// which is guaranteed to satisfy any target.
	var budgets []int
	for b := 8; b < len(a.data); b *= 2 {
		budgets = append(budgets, b)
	}
	budgets = append(budgets, 0)

	for _, b := range budgets {
		a.forest.MaxChecks = b
		if p := Recall[[]float64](a.forest, a.exact, queries, k); p >= a.target {
			a.selectForest(b, p)
			return
		}
		a.kmeans.Checks = b
		if p := Recall[[]float64](a.kmeans, a.exact, queries, k); p >= a.target {
			a.selectKMeans(b, p)
			return
		}
	}
	// Unreachable: the final budget 0 makes the forest exact.
	a.selectForest(0, 1)
}

// selectForest fixes the forest as the backing structure at budget b.
func (a *AutotunedIndex) selectForest(b int, p float64) {
	a.algo = "kdforest"
	a.checks = b
	a.achieved = p
	a.forest.MaxChecks = b
}

// selectKMeans fixes the k-means tree as the backing structure at budget b.
func (a *AutotunedIndex) selectKMeans(b int, p float64) {
	a.algo = "kmeans"
	a.checks = b
	a.achieved = p
	a.kmeans.Checks = b
}

// Size returns the number of points in the index.
func (a *AutotunedIndex) Size() int { return len(a.data) }

// Algorithm reports the structure autotuning selected: "kdforest", "kmeans", or
// "linear" for an empty dataset.
func (a *AutotunedIndex) Algorithm() string { return a.algo }

// Checks reports the check budget autotuning selected; 0 means the search is
// exact.
func (a *AutotunedIndex) Checks() int { return a.checks }

// TargetPrecision reports the precision the index was asked to reach.
func (a *AutotunedIndex) TargetPrecision() float64 { return a.target }

// AchievedPrecision reports the precision the selected configuration attained on
// the tuning sample, an estimate of its true precision.
func (a *AutotunedIndex) AchievedPrecision() float64 { return a.achieved }

// KnnSearch returns the k nearest neighbours of query using the selected
// configuration.
func (a *AutotunedIndex) KnnSearch(query []float64, k int) []Neighbor {
	switch a.algo {
	case "kmeans":
		return a.kmeans.KnnSearch(query, k)
	case "kdforest":
		return a.forest.KnnSearch(query, k)
	default:
		return a.exact.KnnSearch(query, k)
	}
}

// RadiusSearch returns every point within radius of query. The backing
// structures answer radius queries exactly, so the result is exact.
func (a *AutotunedIndex) RadiusSearch(query []float64, radius float64) []Neighbor {
	switch a.algo {
	case "kmeans":
		return a.kmeans.RadiusSearch(query, radius)
	case "kdforest":
		return a.forest.RadiusSearch(query, radius)
	default:
		return a.exact.RadiusSearch(query, radius)
	}
}
