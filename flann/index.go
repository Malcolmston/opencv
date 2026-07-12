package flann

import (
	"fmt"
	"sort"
)

// Neighbor is a single search result: the row index of a point in the dataset
// the index was built over, and that point's Distance to the query. The units
// of Distance are those of the index's distance function ([DistL2] for the
// real-valued indices, [DistHamming] for the binary ones).
type Neighbor struct {
	// Index is the row of the matching point in the original dataset.
	Index int
	// Distance is the point's distance to the query.
	Distance float64
}

// Index is the common interface implemented by every FLANN search structure.
// The type parameter T is the element type of both the dataset and the query:
// []float64 for the real-valued indices ([KDTreeIndex], [KMeansIndex] and the
// float [LinearIndex]) and []byte for the binary [LSHIndex] and binary
// [LinearIndex].
//
// KnnSearch returns the k nearest neighbours of query (fewer if the dataset
// holds fewer than k points, none for k <= 0). RadiusSearch returns every
// point whose distance to query is <= radius. Both results are sorted ascending
// by distance, ties broken by ascending Index.
type Index[T any] interface {
	KnnSearch(query T, k int) []Neighbor
	RadiusSearch(query T, radius float64) []Neighbor
}

// lessNeighbor is the canonical ordering of results: nearest first, ties broken
// by the smaller dataset index. It makes every search deterministic.
func lessNeighbor(a, b Neighbor) bool {
	if a.Distance != b.Distance {
		return a.Distance < b.Distance
	}
	return a.Index < b.Index
}

// sortNeighbors sorts a result slice into the canonical order in place.
func sortNeighbors(n []Neighbor) {
	sort.Slice(n, func(i, j int) bool { return lessNeighbor(n[i], n[j]) })
}

// knnSet maintains the k best neighbours seen so far, kept sorted ascending so
// that the worst is always last. It is the collector shared by every k-NN
// search: add offers a candidate, worst reports the current pruning bound, and
// full reports whether k candidates have been accumulated.
type knnSet struct {
	k int
	n []Neighbor
}

// full reports whether k candidates have been collected.
func (s *knnSet) full() bool { return len(s.n) >= s.k }

// worst returns the distance of the current k-th neighbour, or +Inf while fewer
// than k have been collected (so any candidate is admitted).
func (s *knnSet) worst() float64 {
	if !s.full() {
		return positiveInf
	}
	return s.n[len(s.n)-1].Distance
}

// add offers a candidate, inserting it in sorted position and discarding the
// worst if the set already holds k neighbours.
func (s *knnSet) add(nb Neighbor) {
	if s.k <= 0 {
		return
	}
	if s.full() && !lessNeighbor(nb, s.n[len(s.n)-1]) {
		return
	}
	pos := sort.Search(len(s.n), func(i int) bool { return lessNeighbor(nb, s.n[i]) })
	s.n = append(s.n, Neighbor{})
	copy(s.n[pos+1:], s.n[pos:])
	s.n[pos] = nb
	if len(s.n) > s.k {
		s.n = s.n[:s.k]
	}
}

// LinearIndex is the exact brute-force baseline: KnnSearch and RadiusSearch
// scan the entire dataset. Being exact and free of tuning, it is the reference
// implementation the approximate indices are validated against. It is generic
// over the element type T and uses whatever [DistanceFunc] it was constructed
// with; use [NewLinearIndex] for L2 float search, [NewLinearBinaryIndex] for
// Hamming binary search, or [NewLinearIndexFunc] for a custom distance.
type LinearIndex[T any] struct {
	data []T
	dist DistanceFunc[T]
}

// NewLinearIndex builds an exact L2 (Euclidean) brute-force index over the
// real-valued dataset. It panics if the dataset is ragged (its rows differ in
// length). An empty dataset is allowed and yields empty searches.
func NewLinearIndex(data [][]float64) *LinearIndex[[]float64] {
	validateFloatData(data, "NewLinearIndex")
	return &LinearIndex[[]float64]{data: data, dist: DistL2}
}

// NewLinearBinaryIndex builds an exact Hamming brute-force index over the
// binary dataset. It panics if the dataset is ragged. An empty dataset is
// allowed and yields empty searches.
func NewLinearBinaryIndex(data [][]byte) *LinearIndex[[]byte] {
	validateByteData(data, "NewLinearBinaryIndex")
	return &LinearIndex[[]byte]{data: data, dist: DistHamming}
}

// NewLinearIndexFunc builds an exact brute-force index over an arbitrary
// dataset scored by the supplied distance function. It panics if dist is nil.
func NewLinearIndexFunc[T any](data []T, dist DistanceFunc[T]) *LinearIndex[T] {
	if dist == nil {
		panic("flann: NewLinearIndexFunc requires a non-nil distance function")
	}
	return &LinearIndex[T]{data: data, dist: dist}
}

// Size returns the number of points in the index.
func (li *LinearIndex[T]) Size() int { return len(li.data) }

// KnnSearch returns the k nearest neighbours of query by scanning every point.
func (li *LinearIndex[T]) KnnSearch(query T, k int) []Neighbor {
	if k <= 0 {
		return nil
	}
	res := &knnSet{k: k}
	for i := range li.data {
		res.add(Neighbor{Index: i, Distance: li.dist(query, li.data[i])})
	}
	return res.n
}

// RadiusSearch returns every point within radius of query, sorted ascending by
// distance.
func (li *LinearIndex[T]) RadiusSearch(query T, radius float64) []Neighbor {
	var out []Neighbor
	for i := range li.data {
		d := li.dist(query, li.data[i])
		if d <= radius {
			out = append(out, Neighbor{Index: i, Distance: d})
		}
	}
	sortNeighbors(out)
	return out
}

// validateFloatData panics if data is ragged. It returns the dimensionality
// (0 for an empty dataset).
func validateFloatData(data [][]float64, who string) int {
	if len(data) == 0 {
		return 0
	}
	dim := len(data[0])
	for i, row := range data {
		if len(row) != dim {
			panic(fmt.Sprintf("flann: %s dataset is ragged: row %d has length %d, want %d", who, i, len(row), dim))
		}
	}
	return dim
}

// validateByteData panics if data is ragged. It returns the descriptor length
// in bytes (0 for an empty dataset).
func validateByteData(data [][]byte, who string) int {
	if len(data) == 0 {
		return 0
	}
	dim := len(data[0])
	for i, row := range data {
		if len(row) != dim {
			panic(fmt.Sprintf("flann: %s dataset is ragged: row %d has length %d, want %d", who, i, len(row), dim))
		}
	}
	return dim
}

// Compile-time checks that every index satisfies the common interface.
var (
	_ Index[[]float64] = (*LinearIndex[[]float64])(nil)
	_ Index[[]byte]    = (*LinearIndex[[]byte])(nil)
	_ Index[[]float64] = (*KDTreeIndex)(nil)
	_ Index[[]float64] = (*KMeansIndex)(nil)
	_ Index[[]byte]    = (*LSHIndex)(nil)
)
