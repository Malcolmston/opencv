package features2d

import (
	"math"
	"math/bits"
	"sort"
)

// NormType selects the distance used to compare descriptors.
type NormType int

const (
	// NormHamming counts differing bits between binary descriptors. It is the
	// correct norm for [BRIEF] and [ORB].
	NormHamming NormType = iota
	// NormL2 is the Euclidean distance between float descriptors.
	NormL2
)

// BFMatcher is a brute-force descriptor matcher: it compares every query
// descriptor against every train descriptor. Set Norm to match the descriptor
// type ([NormHamming] for binary, [NormL2] for float). When CrossCheck is true,
// [BFMatcher.Match] only returns a pair (i, j) when j is the best match for i
// and i is also the best match for j, which discards many false matches.
type BFMatcher struct {
	// Norm is the distance metric.
	Norm NormType
	// CrossCheck enables mutual-best-match filtering in Match.
	CrossCheck bool
}

// NewBFMatcher returns a brute-force matcher using the given norm.
func NewBFMatcher(norm NormType) *BFMatcher {
	return &BFMatcher{Norm: norm}
}

// HammingDistance returns the number of differing bits between two equal-length
// bit-packed descriptors. It panics if the lengths differ.
func HammingDistance(a, b []byte) int {
	if len(a) != len(b) {
		panic("features2d: HammingDistance length mismatch")
	}
	d := 0
	for i := range a {
		d += bits.OnesCount8(a[i] ^ b[i])
	}
	return d
}

// l2Distance returns the Euclidean distance between two equal-length float
// descriptors.
func l2Distance(a, b []float64) float64 {
	if len(a) != len(b) {
		panic("features2d: l2Distance length mismatch")
	}
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return math.Sqrt(s)
}

// distance computes the configured distance between query row qi and train row
// ti of the two descriptor sets.
func (m *BFMatcher) distance(query, train Descriptors, qi, ti int) float64 {
	if m.Norm == NormHamming {
		return float64(HammingDistance(query.Binary[qi], train.Binary[ti]))
	}
	return l2Distance(query.Float[qi], train.Float[ti])
}

// validate checks that both descriptor sets are non-empty and of the type the
// configured norm expects.
func (m *BFMatcher) validate(query, train Descriptors) {
	binary := m.Norm == NormHamming
	if binary && (query.Binary == nil || train.Binary == nil) {
		panic("features2d: NormHamming requires binary descriptors")
	}
	if !binary && (query.Float == nil || train.Float == nil) {
		panic("features2d: NormL2 requires float descriptors")
	}
}

// bestTrain returns the index and distance of the train descriptor closest to
// query row qi, or (-1, 0) when train is empty. Ties are broken by the lower
// train index for determinism.
func (m *BFMatcher) bestTrain(query, train Descriptors, qi int) (int, float64) {
	best, bestDist := -1, math.Inf(1)
	for ti := 0; ti < train.Len(); ti++ {
		d := m.distance(query, train, qi, ti)
		if d < bestDist {
			bestDist, best = d, ti
		}
	}
	if best < 0 {
		return -1, 0
	}
	return best, bestDist
}

// Match returns the best train match for each query descriptor. With CrossCheck
// disabled it returns one DMatch per query, in query order. With CrossCheck
// enabled it returns only mutually-best pairs, sorted by ascending distance. It
// panics if the descriptor type does not match the configured norm.
func (m *BFMatcher) Match(query, train Descriptors) []DMatch {
	if query.Len() == 0 || train.Len() == 0 {
		return nil
	}
	m.validate(query, train)

	if !m.CrossCheck {
		out := make([]DMatch, 0, query.Len())
		for qi := 0; qi < query.Len(); qi++ {
			ti, d := m.bestTrain(query, train, qi)
			out = append(out, DMatch{QueryIdx: qi, TrainIdx: ti, Distance: d})
		}
		return out
	}

	// Precompute, for each train descriptor, its best query, so cross-check is
	// O(Q*T) rather than O(Q*T) twice with allocation per query.
	trainBestQuery := make([]int, train.Len())
	trainBestDist := make([]float64, train.Len())
	for ti := range trainBestQuery {
		trainBestQuery[ti] = -1
		trainBestDist[ti] = math.Inf(1)
	}
	for qi := 0; qi < query.Len(); qi++ {
		for ti := 0; ti < train.Len(); ti++ {
			d := m.distance(query, train, qi, ti)
			if d < trainBestDist[ti] {
				trainBestDist[ti] = d
				trainBestQuery[ti] = qi
			}
		}
	}
	var out []DMatch
	for qi := 0; qi < query.Len(); qi++ {
		ti, d := m.bestTrain(query, train, qi)
		if ti >= 0 && trainBestQuery[ti] == qi {
			out = append(out, DMatch{QueryIdx: qi, TrainIdx: ti, Distance: d})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Distance < out[j].Distance })
	return out
}

// KnnMatch returns, for each query descriptor, its k nearest train descriptors
// as a slice of DMatch sorted by ascending distance (ties broken by lower train
// index). The outer slice is in query order. CrossCheck is ignored. It panics if
// k < 1 or the descriptor type does not match the configured norm.
func (m *BFMatcher) KnnMatch(query, train Descriptors, k int) [][]DMatch {
	if k < 1 {
		panic("features2d: KnnMatch requires k >= 1")
	}
	if query.Len() == 0 || train.Len() == 0 {
		return nil
	}
	m.validate(query, train)

	out := make([][]DMatch, query.Len())
	for qi := 0; qi < query.Len(); qi++ {
		cands := make([]DMatch, train.Len())
		for ti := 0; ti < train.Len(); ti++ {
			cands[ti] = DMatch{QueryIdx: qi, TrainIdx: ti, Distance: m.distance(query, train, qi, ti)}
		}
		sort.SliceStable(cands, func(a, b int) bool {
			if cands[a].Distance != cands[b].Distance {
				return cands[a].Distance < cands[b].Distance
			}
			return cands[a].TrainIdx < cands[b].TrainIdx
		})
		if len(cands) > k {
			cands = cands[:k]
		}
		out[qi] = cands
	}
	return out
}

// RatioTest applies Lowe's ratio test to a k>=2 KnnMatch result: a query's best
// match is kept only when its distance is strictly less than ratio times the
// distance of the second-best match, which rejects ambiguous matches. Rows with
// a single candidate are kept unconditionally, and empty rows are skipped. The
// surviving best matches are returned in query order.
func RatioTest(knn [][]DMatch, ratio float64) []DMatch {
	var out []DMatch
	for _, row := range knn {
		if len(row) == 0 {
			continue
		}
		if len(row) == 1 || row[0].Distance < ratio*row[1].Distance {
			out = append(out, row[0])
		}
	}
	return out
}
