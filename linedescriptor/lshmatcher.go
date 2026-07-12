package linedescriptor

import "sort"

// RadiusMatch returns, for each query descriptor, every train descriptor within
// maxDistance Hamming bits of it, as a slice of [DMatch] sorted by ascending
// distance (ties by lower train index). The outer slice is in query order,
// mirroring cv::DescriptorMatcher::radiusMatch. A query with no train
// descriptor inside the radius yields an empty (non-nil) inner slice. maxDistance
// must be non-negative; empty query or train sets yield a nil result.
func (m *BinaryDescriptorMatcher) RadiusMatch(query, train [][]byte, maxDistance int) [][]DMatch {
	if maxDistance < 0 {
		panic("linedescriptor: RadiusMatch requires maxDistance >= 0")
	}
	if len(query) == 0 || len(train) == 0 {
		return nil
	}
	out := make([][]DMatch, len(query))
	for qi, q := range query {
		matches := make([]DMatch, 0)
		for ti, t := range train {
			if d := HammingDistance(q, t); d <= maxDistance {
				matches = append(matches, DMatch{QueryIdx: qi, TrainIdx: ti, Distance: d})
			}
		}
		sort.SliceStable(matches, func(a, b int) bool {
			if matches[a].Distance != matches[b].Distance {
				return matches[a].Distance < matches[b].Distance
			}
			return matches[a].TrainIdx < matches[b].TrainIdx
		})
		out[qi] = matches
	}
	return out
}

// LSHIndex is a multi-index locality-sensitive-hashing structure over binary
// line descriptors, modelled on the FLANN LSH matcher that the upstream
// [BinaryDescriptorMatcher] uses to avoid an exhaustive scan. It maintains
// several independent hash tables, each keyed on a different deterministic
// subset of the descriptor bits. A stored descriptor is a candidate for a query
// whenever the two collide in *any* table, so near-duplicates (which differ in
// only a few bits) are very likely to share a bucket in at least one table even
// though a single table might separate them. Candidates gathered from the tables
// are then ranked by exact Hamming distance.
//
// The index is deterministic: table bit-assignments follow a fixed interleaving
// and all ordering ties break by the lower train index.
type LSHIndex struct {
	numTables int
	keyBits   int
	// tableBits[t] lists the descriptor bit positions that table t hashes.
	tableBits [][]int
	// tables[t] maps a bucket key to the train indices stored in it.
	tables []map[uint64][]int
	train  [][]byte
}

// NewLSHIndex creates an empty multi-index with numTables hash tables, each
// keyed on keyBits descriptor bits. More tables raise recall at the cost of
// memory; more key bits shrink buckets, raising precision but lowering recall.
// Both arguments must be positive and keyBits must not exceed 64.
func NewLSHIndex(numTables, keyBits int) *LSHIndex {
	if numTables < 1 || keyBits < 1 {
		panic("linedescriptor: NewLSHIndex requires positive numTables and keyBits")
	}
	if keyBits > 64 {
		panic("linedescriptor: NewLSHIndex keyBits must be <= 64")
	}
	return &LSHIndex{
		numTables: numTables,
		keyBits:   keyBits,
		tables:    make([]map[uint64][]int, 0, numTables),
	}
}

// Add stores train descriptors in the index, assigning them indices starting at
// the current descriptor count so repeated calls accumulate. All descriptors
// must share the byte length of the first one added. The bit-to-table
// assignment is derived from that length on the first call.
func (x *LSHIndex) Add(train [][]byte) {
	if len(train) == 0 {
		return
	}
	if x.tableBits == nil {
		x.initTables(len(train[0]) * 8)
	}
	base := len(x.train)
	for i, code := range train {
		idx := base + i
		x.train = append(x.train, code)
		for t := 0; t < x.numTables; t++ {
			key := extractBits(code, x.tableBits[t])
			x.tables[t][key] = append(x.tables[t][key], idx)
		}
	}
}

// initTables assigns each table a deterministic, interleaved subset of the
// totalBits descriptor bit positions and allocates the empty buckets.
func (x *LSHIndex) initTables(totalBits int) {
	if totalBits == 0 {
		panic("linedescriptor: LSHIndex cannot index zero-length descriptors")
	}
	x.tableBits = make([][]int, x.numTables)
	x.tables = make([]map[uint64][]int, x.numTables)
	for t := 0; t < x.numTables; t++ {
		positions := make([]int, x.keyBits)
		for j := 0; j < x.keyBits; j++ {
			// Interleave so different tables sample different, spread-out bits.
			positions[j] = (t + j*x.numTables) % totalBits
		}
		x.tableBits[t] = positions
		x.tables[t] = make(map[uint64][]int)
	}
}

// candidates returns the deduplicated, ascending train indices that share a
// bucket with code in at least one table.
func (x *LSHIndex) candidates(code []byte) []int {
	seen := make(map[int]struct{})
	for t := 0; t < x.numTables; t++ {
		key := extractBits(code, x.tableBits[t])
		for _, idx := range x.tables[t][key] {
			seen[idx] = struct{}{}
		}
	}
	out := make([]int, 0, len(seen))
	for idx := range seen {
		out = append(out, idx)
	}
	sort.Ints(out)
	return out
}

// KnnMatch returns, for each query descriptor, its k nearest indexed
// descriptors ranked by exact Hamming distance, restricted to the candidates
// that collide with the query in at least one hash table. The outer slice is in
// query order; a query whose buckets are empty yields an empty inner slice. It
// panics if k < 1 or the index is empty.
func (x *LSHIndex) KnnMatch(query [][]byte, k int) [][]DMatch {
	if k < 1 {
		panic("linedescriptor: LSHIndex.KnnMatch requires k >= 1")
	}
	if len(x.train) == 0 {
		panic("linedescriptor: LSHIndex.KnnMatch on an empty index")
	}
	out := make([][]DMatch, len(query))
	for qi, q := range query {
		cands := x.candidates(q)
		matches := make([]DMatch, 0, len(cands))
		for _, ti := range cands {
			matches = append(matches, DMatch{QueryIdx: qi, TrainIdx: ti, Distance: HammingDistance(q, x.train[ti])})
		}
		sortMatches(matches)
		if len(matches) > k {
			matches = matches[:k]
		}
		out[qi] = matches
	}
	return out
}

// RadiusMatch returns, for each query descriptor, every candidate indexed
// descriptor within maxDistance Hamming bits, ranked by ascending distance. Only
// descriptors that collide with the query in at least one hash table are
// considered. maxDistance must be non-negative; the index must be non-empty.
func (x *LSHIndex) RadiusMatch(query [][]byte, maxDistance int) [][]DMatch {
	if maxDistance < 0 {
		panic("linedescriptor: LSHIndex.RadiusMatch requires maxDistance >= 0")
	}
	if len(x.train) == 0 {
		panic("linedescriptor: LSHIndex.RadiusMatch on an empty index")
	}
	out := make([][]DMatch, len(query))
	for qi, q := range query {
		matches := make([]DMatch, 0)
		for _, ti := range x.candidates(q) {
			if d := HammingDistance(q, x.train[ti]); d <= maxDistance {
				matches = append(matches, DMatch{QueryIdx: qi, TrainIdx: ti, Distance: d})
			}
		}
		sortMatches(matches)
		out[qi] = matches
	}
	return out
}

// Size reports how many descriptors are currently stored in the index.
func (x *LSHIndex) Size() int { return len(x.train) }

// extractBits packs the descriptor bits at the given positions into a single
// uint64 bucket key, most-significant first.
func extractBits(code []byte, positions []int) uint64 {
	var key uint64
	for _, p := range positions {
		key <<= 1
		if code[p/8]&(1<<uint(7-p%8)) != 0 {
			key |= 1
		}
	}
	return key
}

// sortMatches orders matches by ascending distance, ties by lower train index.
func sortMatches(matches []DMatch) {
	sort.SliceStable(matches, func(a, b int) bool {
		if matches[a].Distance != matches[b].Distance {
			return matches[a].Distance < matches[b].Distance
		}
		return matches[a].TrainIdx < matches[b].TrainIdx
	})
}
