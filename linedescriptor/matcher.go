package linedescriptor

import (
	"math/bits"
	"sort"
)

// DMatch records a correspondence between a query descriptor and a train
// descriptor, as produced by a [BinaryDescriptorMatcher]. It mirrors OpenCV's
// cv::DMatch.
type DMatch struct {
	// QueryIdx is the index of the descriptor in the query set.
	QueryIdx int
	// TrainIdx is the index of the matched descriptor in the train set.
	TrainIdx int
	// Distance is the Hamming distance between the two codes; smaller means
	// more similar.
	Distance int
}

// HammingDistance returns the number of differing bits between two equal-length
// bit-packed line descriptors. It panics if the lengths differ.
func HammingDistance(a, b []byte) int {
	if len(a) != len(b) {
		panic("linedescriptor: HammingDistance length mismatch")
	}
	d := 0
	for i := range a {
		d += bits.OnesCount8(a[i] ^ b[i])
	}
	return d
}

// BinaryDescriptorMatcher is a brute-force matcher over the binary line codes
// produced by [BinaryDescriptor]. It compares every query code against every
// train code with the Hamming distance. Its methods are deterministic: all
// ordering ties are broken by the lower train index.
type BinaryDescriptorMatcher struct{}

// NewBinaryDescriptorMatcher returns a ready-to-use matcher.
func NewBinaryDescriptorMatcher() *BinaryDescriptorMatcher {
	return &BinaryDescriptorMatcher{}
}

// Match returns, for each query descriptor, its single closest train
// descriptor. The result has one [DMatch] per query, in query order. Empty
// query or train sets yield a nil result.
func (m *BinaryDescriptorMatcher) Match(query, train [][]byte) []DMatch {
	if len(query) == 0 || len(train) == 0 {
		return nil
	}
	out := make([]DMatch, 0, len(query))
	for qi, q := range query {
		best, bestDist := -1, int(^uint(0)>>1)
		for ti, t := range train {
			if d := HammingDistance(q, t); d < bestDist {
				bestDist, best = d, ti
			}
		}
		out = append(out, DMatch{QueryIdx: qi, TrainIdx: best, Distance: bestDist})
	}
	return out
}

// KnnMatch returns, for each query descriptor, its k nearest train descriptors
// as a slice of [DMatch] sorted by ascending distance (ties broken by the lower
// train index). The outer slice is in query order. It panics if k < 1; empty
// query or train sets yield a nil result.
func (m *BinaryDescriptorMatcher) KnnMatch(query, train [][]byte, k int) [][]DMatch {
	if k < 1 {
		panic("linedescriptor: KnnMatch requires k >= 1")
	}
	if len(query) == 0 || len(train) == 0 {
		return nil
	}
	out := make([][]DMatch, len(query))
	for qi, q := range query {
		cands := make([]DMatch, len(train))
		for ti, t := range train {
			cands[ti] = DMatch{QueryIdx: qi, TrainIdx: ti, Distance: HammingDistance(q, t)}
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
