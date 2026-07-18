package matching2

import (
	"math"
	"sort"
)

// BFMatcher is a brute-force matcher for floating-point descriptors: it
// compares every query descriptor against every train descriptor under the
// configured [NormType]. Construct one with [NewBFMatcher].
type BFMatcher struct {
	// Norm is the distance metric applied to descriptor pairs.
	Norm NormType
	// CrossCheck, when true, makes Match return only mutually-best pairs: a
	// query/train pair survives only when each is the other's nearest neighbour.
	CrossCheck bool
}

// NewBFMatcher returns a brute-force float matcher using the given norm with
// cross-check disabled.
func NewBFMatcher(norm NormType) *BFMatcher {
	return &BFMatcher{Norm: norm}
}

// Match returns the single best train match for each query descriptor. With
// CrossCheck disabled it returns one [DMatch] per query, in query order. With
// CrossCheck enabled it returns only mutually-best pairs, sorted by ascending
// distance. Query and train are descriptor sets, one row per descriptor; all
// rows must share the same length. Empty inputs yield a nil result.
func (m *BFMatcher) Match(query, train [][]float64) []DMatch {
	if len(query) == 0 || len(train) == 0 {
		return nil
	}
	if !m.CrossCheck {
		out := make([]DMatch, 0, len(query))
		for qi := range query {
			ti, d := m.best(query[qi], train)
			out = append(out, DMatch{QueryIdx: qi, TrainIdx: ti, Distance: d})
		}
		return out
	}
	trainBestQ := make([]int, len(train))
	trainBestD := make([]float64, len(train))
	for ti := range trainBestQ {
		trainBestQ[ti] = -1
		trainBestD[ti] = math.Inf(1)
	}
	for qi := range query {
		for ti := range train {
			d := FloatDistance(m.Norm, query[qi], train[ti])
			if d < trainBestD[ti] {
				trainBestD[ti] = d
				trainBestQ[ti] = qi
			}
		}
	}
	var out []DMatch
	for qi := range query {
		ti, d := m.best(query[qi], train)
		if ti >= 0 && trainBestQ[ti] == qi {
			out = append(out, DMatch{QueryIdx: qi, TrainIdx: ti, Distance: d})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Distance < out[j].Distance })
	return out
}

// KnnMatch returns, for each query descriptor, its k nearest train descriptors
// as a slice of [DMatch] sorted by ascending distance (ties broken by lower
// train index). The outer slice is in query order. If k exceeds the number of
// train descriptors, every train descriptor is returned. CrossCheck is ignored.
func (m *BFMatcher) KnnMatch(query, train [][]float64, k int) [][]DMatch {
	if len(query) == 0 || len(train) == 0 || k <= 0 {
		return nil
	}
	out := make([][]DMatch, len(query))
	for qi := range query {
		cand := make([]DMatch, len(train))
		for ti := range train {
			cand[ti] = DMatch{QueryIdx: qi, TrainIdx: ti, Distance: FloatDistance(m.Norm, query[qi], train[ti])}
		}
		sortMatches(cand)
		if k < len(cand) {
			cand = cand[:k]
		}
		out[qi] = cand
	}
	return out
}

// RadiusMatch returns, for each query descriptor, every train descriptor within
// maxDistance, sorted by ascending distance. The outer slice is in query order;
// an entry is nil when nothing lies inside the radius.
func (m *BFMatcher) RadiusMatch(query, train [][]float64, maxDistance float64) [][]DMatch {
	if len(query) == 0 || len(train) == 0 {
		return nil
	}
	out := make([][]DMatch, len(query))
	for qi := range query {
		var cand []DMatch
		for ti := range train {
			d := FloatDistance(m.Norm, query[qi], train[ti])
			if d <= maxDistance {
				cand = append(cand, DMatch{QueryIdx: qi, TrainIdx: ti, Distance: d})
			}
		}
		sortMatches(cand)
		out[qi] = cand
	}
	return out
}

// best returns the index and distance of the train descriptor nearest to q, or
// (-1, 0) when train is empty. Ties break to the lower index.
func (m *BFMatcher) best(q []float64, train [][]float64) (int, float64) {
	bi, bd := -1, math.Inf(1)
	for ti := range train {
		d := FloatDistance(m.Norm, q, train[ti])
		if d < bd {
			bd, bi = d, ti
		}
	}
	if bi < 0 {
		return -1, 0
	}
	return bi, bd
}

// HammingMatcher is a brute-force matcher for bit-packed binary descriptors,
// such as those produced by ORB and BRIEF, compared with the Hamming distance.
// Construct one with [NewHammingMatcher].
type HammingMatcher struct {
	// CrossCheck, when true, keeps only mutually-best pairs in Match.
	CrossCheck bool
}

// NewHammingMatcher returns a brute-force Hamming matcher with cross-check
// disabled.
func NewHammingMatcher() *HammingMatcher {
	return &HammingMatcher{}
}

// Match returns the best train match for each binary query descriptor, mirroring
// [BFMatcher.Match] but with the Hamming distance. Distances are whole numbers
// of differing bits stored as float64.
func (m *HammingMatcher) Match(query, train [][]byte) []DMatch {
	if len(query) == 0 || len(train) == 0 {
		return nil
	}
	if !m.CrossCheck {
		out := make([]DMatch, 0, len(query))
		for qi := range query {
			ti, d := hammBest(query[qi], train)
			out = append(out, DMatch{QueryIdx: qi, TrainIdx: ti, Distance: d})
		}
		return out
	}
	trainBestQ := make([]int, len(train))
	trainBestD := make([]int, len(train))
	for ti := range trainBestQ {
		trainBestQ[ti] = -1
		trainBestD[ti] = math.MaxInt
	}
	for qi := range query {
		for ti := range train {
			d := HammingDistance(query[qi], train[ti])
			if d < trainBestD[ti] {
				trainBestD[ti] = d
				trainBestQ[ti] = qi
			}
		}
	}
	var out []DMatch
	for qi := range query {
		ti, d := hammBest(query[qi], train)
		if ti >= 0 && trainBestQ[ti] == qi {
			out = append(out, DMatch{QueryIdx: qi, TrainIdx: ti, Distance: d})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Distance < out[j].Distance })
	return out
}

// KnnMatch returns, for each binary query descriptor, its k nearest train
// descriptors sorted by ascending Hamming distance.
func (m *HammingMatcher) KnnMatch(query, train [][]byte, k int) [][]DMatch {
	if len(query) == 0 || len(train) == 0 || k <= 0 {
		return nil
	}
	out := make([][]DMatch, len(query))
	for qi := range query {
		cand := make([]DMatch, len(train))
		for ti := range train {
			cand[ti] = DMatch{QueryIdx: qi, TrainIdx: ti, Distance: float64(HammingDistance(query[qi], train[ti]))}
		}
		sortMatches(cand)
		if k < len(cand) {
			cand = cand[:k]
		}
		out[qi] = cand
	}
	return out
}

// hammBest returns the nearest train index and its Hamming distance for query q.
func hammBest(q []byte, train [][]byte) (int, float64) {
	bi, bd := -1, math.MaxInt
	for ti := range train {
		d := HammingDistance(q, train[ti])
		if d < bd {
			bd, bi = d, ti
		}
	}
	if bi < 0 {
		return -1, 0
	}
	return bi, float64(bd)
}

// sortMatches orders matches by ascending distance, breaking ties by lower
// train index, in place and deterministically.
func sortMatches(ms []DMatch) {
	sort.SliceStable(ms, func(i, j int) bool {
		if ms[i].Distance != ms[j].Distance {
			return ms[i].Distance < ms[j].Distance
		}
		return ms[i].TrainIdx < ms[j].TrainIdx
	})
}

// RatioTest applies Lowe's ratio test to the output of a k-nearest-neighbour
// match with k ≥ 2: it keeps a query's best match only when the best distance is
// below ratio times the second-best distance, which discards ambiguous matches.
// A typical ratio is 0.75. Entries with fewer than two neighbours are dropped.
// The surviving matches are returned in query order.
func RatioTest(knn [][]DMatch, ratio float64) []DMatch {
	var out []DMatch
	for _, nbrs := range knn {
		if len(nbrs) < 2 {
			continue
		}
		if nbrs[1].Distance == 0 {
			// Degenerate: both neighbours coincide. Accept only an exact match.
			if nbrs[0].Distance == 0 {
				out = append(out, nbrs[0])
			}
			continue
		}
		if nbrs[0].Distance < ratio*nbrs[1].Distance {
			out = append(out, nbrs[0])
		}
	}
	return out
}

// CrossCheck keeps only the matches in forward (query→train) that are confirmed
// by backward (train→query): a pair (q, t) survives when backward reports q as
// the best match for t. Both arguments are one-best match lists as returned by
// Match with CrossCheck disabled. The result is sorted by ascending distance.
func CrossCheck(forward, backward []DMatch) []DMatch {
	back := make(map[int]int, len(backward))
	for _, b := range backward {
		back[b.QueryIdx] = b.TrainIdx
	}
	var out []DMatch
	for _, f := range forward {
		if f.TrainIdx < 0 {
			continue
		}
		if q, ok := back[f.TrainIdx]; ok && q == f.QueryIdx {
			out = append(out, f)
		}
	}
	sortMatches(out)
	return out
}

// FilterMatchesByDistance returns the subset of matches whose distance does not
// exceed maxDistance, preserving input order.
func FilterMatchesByDistance(matches []DMatch, maxDistance float64) []DMatch {
	var out []DMatch
	for _, mm := range matches {
		if mm.Distance <= maxDistance {
			out = append(out, mm)
		}
	}
	return out
}

// SortMatchesByDistance returns a copy of matches sorted by ascending distance,
// breaking ties by lower train index. The input is not modified.
func SortMatchesByDistance(matches []DMatch) []DMatch {
	out := make([]DMatch, len(matches))
	copy(out, matches)
	sortMatches(out)
	return out
}

// MinMaxDistance returns the smallest and largest distance among the matches,
// or (0, 0) when matches is empty. It is commonly used to derive an adaptive
// threshold for [FilterMatchesByDistance].
func MinMaxDistance(matches []DMatch) (min, max float64) {
	if len(matches) == 0 {
		return 0, 0
	}
	min, max = math.Inf(1), math.Inf(-1)
	for _, mm := range matches {
		if mm.Distance < min {
			min = mm.Distance
		}
		if mm.Distance > max {
			max = mm.Distance
		}
	}
	return min, max
}
