package flann

import (
	"fmt"
	"math/rand"
)

// Default construction parameters for the LSH index.
const (
	defaultLSHTables  = 8
	defaultLSHKeySize = 16
)

// lshTable is one hash table: a fixed set of bit positions sampled from the
// descriptor, and the buckets that group dataset points sharing the same value
// on those bits.
type lshTable struct {
	positions []int            // len == keySize, values in [0, 8*dim)
	buckets   map[uint64][]int // hash key -> dataset indices
}

// LSHIndex is a multi-table locality-sensitive hash over binary descriptors,
// using the Hamming distance. Each of its tables hashes a random subset of the
// descriptor's bits; two descriptors that agree on those bits land in the same
// bucket. A query gathers the candidates from the matching bucket of every
// table, then ranks that (much smaller) candidate set by exact Hamming
// distance. Because an exact duplicate agrees on every bit, it collides in
// every table and is therefore always retrieved, which makes LSH very effective
// at finding a near-exact match among many distractors.
type LSHIndex struct {
	data    [][]byte
	dim     int // descriptor length in bytes
	keySize int
	tables  []lshTable
}

// NewLSHIndex builds a multi-table LSH index over the binary dataset.
// tableNumber is the number of hash tables (>= 1) and keySize is the number of
// bits hashed per table (1..64); pass 0 for either to use sensible defaults
// (8 tables, 16 bits). Bit positions are sampled using seed so the index is
// reproducible. It panics if the dataset is ragged, if keySize exceeds 64 or
// the descriptor bit width, or on other out-of-range parameters. An empty
// dataset is allowed and yields empty searches.
func NewLSHIndex(data [][]byte, tableNumber, keySize int, seed int64) *LSHIndex {
	dim := validateByteData(data, "NewLSHIndex")
	if tableNumber <= 0 {
		tableNumber = defaultLSHTables
	}
	if keySize <= 0 {
		keySize = defaultLSHKeySize
	}
	if keySize > 64 {
		panic(fmt.Sprintf("flann: NewLSHIndex keySize %d exceeds 64 bits", keySize))
	}
	idx := &LSHIndex{data: data, dim: dim, keySize: keySize}
	if len(data) == 0 {
		return idx
	}
	totalBits := dim * 8
	if keySize > totalBits {
		panic(fmt.Sprintf("flann: NewLSHIndex keySize %d exceeds descriptor width %d bits", keySize, totalBits))
	}
	rng := rand.New(rand.NewSource(seed))
	idx.tables = make([]lshTable, tableNumber)
	for t := range idx.tables {
		positions := samplePositions(totalBits, keySize, rng)
		buckets := make(map[uint64][]int)
		for i, desc := range data {
			key := hashBits(desc, positions)
			buckets[key] = append(buckets[key], i)
		}
		idx.tables[t] = lshTable{positions: positions, buckets: buckets}
	}
	return idx
}

// Size returns the number of points in the index.
func (idx *LSHIndex) Size() int { return len(idx.data) }

// samplePositions returns keySize distinct bit positions drawn uniformly from
// [0, totalBits) using a partial Fisher-Yates shuffle, so the choice is
// unbiased and reproducible for a given rng.
func samplePositions(totalBits, keySize int, rng *rand.Rand) []int {
	perm := make([]int, totalBits)
	for i := range perm {
		perm[i] = i
	}
	for i := 0; i < keySize; i++ {
		j := i + rng.Intn(totalBits-i)
		perm[i], perm[j] = perm[j], perm[i]
	}
	out := make([]int, keySize)
	copy(out, perm[:keySize])
	return out
}

// hashBits packs the selected bit positions of desc into a hash key, bit i of
// the key being the descriptor bit at positions[i].
func hashBits(desc []byte, positions []int) uint64 {
	var key uint64
	for i, p := range positions {
		if desc[p>>3]&(1<<uint(p&7)) != 0 {
			key |= 1 << uint(i)
		}
	}
	return key
}

// candidates gathers, without duplicates, the dataset indices sharing a bucket
// with query in any table.
func (idx *LSHIndex) candidates(query []byte) []int {
	seen := make(map[int]struct{})
	var out []int
	for t := range idx.tables {
		key := hashBits(query, idx.tables[t].positions)
		for _, id := range idx.tables[t].buckets[key] {
			if _, ok := seen[id]; !ok {
				seen[id] = struct{}{}
				out = append(out, id)
			}
		}
	}
	return out
}

// KnnSearch returns up to the k nearest neighbours of query among the LSH
// candidate set, ranked by exact Hamming distance. The result is approximate:
// a true neighbour is returned only if it collides with the query in at least
// one table (an exact match always does).
func (idx *LSHIndex) KnnSearch(query []byte, k int) []Neighbor {
	if k <= 0 {
		return nil
	}
	res := &knnSet{k: k}
	for _, id := range idx.candidates(query) {
		res.add(Neighbor{Index: id, Distance: DistHamming(query, idx.data[id])})
	}
	return res.n
}

// RadiusSearch returns the candidates within Hamming distance radius of query,
// sorted ascending by distance. It is approximate for the same reason as
// KnnSearch: only points that collide with the query in some table are
// considered.
func (idx *LSHIndex) RadiusSearch(query []byte, radius float64) []Neighbor {
	var out []Neighbor
	for _, id := range idx.candidates(query) {
		d := DistHamming(query, idx.data[id])
		if d <= radius {
			out = append(out, Neighbor{Index: id, Distance: d})
		}
	}
	sortNeighbors(out)
	return out
}
