package imghash2

import "fmt"

// IsDuplicate reports whether two equal-length binary hashes are near-duplicates
// under an absolute Hamming-distance threshold: it returns true when their
// [HammingDistance] is at most maxDistance. For 64-bit hashes a threshold of
// about 6–10 bits is a common near-duplicate cutoff. It panics on a length
// mismatch.
func IsDuplicate(a, b Hash, maxDistance int) bool {
	return a.Hamming(b) <= maxDistance
}

// NearDuplicate reports whether two equal-length binary hashes are
// near-duplicates under a normalised-distance threshold: it returns true when
// their [NormalizedHamming] is at most maxNormalizedDistance (a value in
// [0, 1], commonly around 0.1). Normalising lets one threshold apply across
// hash sizes. It panics if the hashes differ in length or are empty.
func NearDuplicate(a, b Hash, maxNormalizedDistance float64) bool {
	return a.NormalizedHamming(b) <= maxNormalizedDistance
}

// Match pairs a stored identifier with its Hamming distance to a query hash,
// the result element of [Index.Within] and [Index.Nearest].
type Match struct {
	// ID is the identifier supplied when the matching hash was added.
	ID string
	// Distance is the Hamming distance from the query to the matching hash.
	Distance int
}

// Pair names two identifiers found to be near-duplicates and the Hamming
// distance between their hashes, the result element of [DuplicatePairs].
type Pair struct {
	// A and B are the identifiers of the two near-duplicate entries.
	A, B string
	// Distance is the Hamming distance between their hashes.
	Distance int
}

// Entry associates an identifier with a hash, the input element of
// [DuplicatePairs].
type Entry struct {
	// ID identifies the image the hash was computed from.
	ID string
	// Hash is the image's binary fingerprint.
	Hash Hash
}

// FindDuplicates groups a slice of equal-length hashes into clusters of
// near-duplicates. Two hashes join the same cluster when their [HammingDistance]
// is at most maxDistance, transitively (single-linkage): the result is the
// connected components of the near-duplicate graph. Each returned group holds
// the original indices into hashes in ascending order, and groups appear in
// ascending order of their smallest index. Singletons are included as
// one-element groups, so every index appears exactly once. It panics if the
// hashes are not all the same length.
func FindDuplicates(hashes []Hash, maxDistance int) [][]int {
	n := len(hashes)
	for i := 1; i < n; i++ {
		if len(hashes[i]) != len(hashes[0]) {
			panic("imghash2: FindDuplicates requires equal-length hashes")
		}
	}
	// Union-find over the near-duplicate graph.
	parent := make([]int, n)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	union := func(a, b int) {
		ra, rb := find(a), find(b)
		if ra != rb {
			parent[ra] = rb
		}
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if hashes[i].Hamming(hashes[j]) <= maxDistance {
				union(i, j)
			}
		}
	}
	groupsByRoot := make(map[int][]int)
	for i := 0; i < n; i++ {
		r := find(i)
		groupsByRoot[r] = append(groupsByRoot[r], i)
	}
	// Order groups by their smallest member for determinism.
	out := make([][]int, 0, len(groupsByRoot))
	seen := make(map[int]bool, len(groupsByRoot))
	for i := 0; i < n; i++ {
		r := find(i)
		if seen[r] {
			continue
		}
		seen[r] = true
		out = append(out, groupsByRoot[r])
	}
	return out
}

// DuplicatePairs returns every pair of entries whose hashes are within
// maxDistance Hamming distance of each other. Each unordered pair is reported
// once with A's index less than B's index in the input, in ascending order of
// (A index, B index). It panics if the entries' hashes are not all the same
// length.
func DuplicatePairs(entries []Entry, maxDistance int) []Pair {
	for i := 1; i < len(entries); i++ {
		if len(entries[i].Hash) != len(entries[0].Hash) {
			panic("imghash2: DuplicatePairs requires equal-length hashes")
		}
	}
	var out []Pair
	for i := 0; i < len(entries); i++ {
		for j := i + 1; j < len(entries); j++ {
			d := entries[i].Hash.Hamming(entries[j].Hash)
			if d <= maxDistance {
				out = append(out, Pair{A: entries[i].ID, B: entries[j].ID, Distance: d})
			}
		}
	}
	return out
}

// imghash2bkNode is a node of the BK-tree backing an [Index]. Its children are
// keyed by their exact Hamming distance to this node's hash.
type imghash2bkNode struct {
	id       string
	hash     Hash
	children map[int]*imghash2bkNode
}

// Index is an incremental store of equal-length binary hashes that answers
// nearest-neighbour and radius queries. It is backed by a BK-tree, which
// exploits the triangle inequality of the Hamming metric to prune the search,
// so lookups touch far fewer than all stored hashes on typical corpora. All
// hashes added to one Index must share a bit length. The zero value is not
// usable; construct one with [NewIndex]. An Index is not safe for concurrent
// use.
type Index struct {
	root *imghash2bkNode
	size int
	bits int
}

// NewIndex returns an empty [Index] ready to accept hashes.
func NewIndex() *Index { return &Index{} }

// Len returns the number of hashes stored in the index.
func (idx *Index) Len() int { return idx.size }

// Bits returns the bit length shared by every hash in the index, or 0 while the
// index is empty.
func (idx *Index) Bits() int { return idx.bits }

// Add inserts a hash under the given identifier. The first Add fixes the index's
// bit length; a later Add whose hash has a different length panics. Duplicate
// identifiers are allowed and stored independently.
func (idx *Index) Add(id string, h Hash) {
	if idx.root == nil {
		idx.root = &imghash2bkNode{id: id, hash: h.Clone(), children: map[int]*imghash2bkNode{}}
		idx.bits = h.Bits()
		idx.size = 1
		return
	}
	if h.Bits() != idx.bits {
		panic(fmt.Sprintf("imghash2: Index.Add hash has %d bits, index holds %d", h.Bits(), idx.bits))
	}
	node := idx.root
	for {
		d := node.hash.Hamming(h)
		child, ok := node.children[d]
		if !ok {
			node.children[d] = &imghash2bkNode{id: id, hash: h.Clone(), children: map[int]*imghash2bkNode{}}
			idx.size++
			return
		}
		node = child
	}
}

// Within returns every stored entry whose hash is at most maxDistance Hamming
// distance from h, in ascending order of distance (ties broken by insertion
// order encountered during the search). It returns nil when the index is empty.
// It panics if h's bit length differs from the index's.
func (idx *Index) Within(h Hash, maxDistance int) []Match {
	if idx.root == nil {
		return nil
	}
	if h.Bits() != idx.bits {
		panic(fmt.Sprintf("imghash2: Index.Within hash has %d bits, index holds %d", h.Bits(), idx.bits))
	}
	var out []Match
	stack := []*imghash2bkNode{idx.root}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		d := node.hash.Hamming(h)
		if d <= maxDistance {
			out = append(out, Match{ID: node.id, Distance: d})
		}
		lo, hi := d-maxDistance, d+maxDistance
		for k, child := range node.children {
			if k >= lo && k <= hi {
				stack = append(stack, child)
			}
		}
	}
	stableSortMatches(out)
	return out
}

// Nearest returns the stored entry closest to h in Hamming distance. The bool
// result is false when the index is empty. When several entries tie at the
// minimum distance, the one discovered first during the search is returned. It
// panics if h's bit length differs from the index's.
func (idx *Index) Nearest(h Hash) (Match, bool) {
	if idx.root == nil {
		return Match{}, false
	}
	if h.Bits() != idx.bits {
		panic(fmt.Sprintf("imghash2: Index.Nearest hash has %d bits, index holds %d", h.Bits(), idx.bits))
	}
	best := Match{Distance: idx.bits + 1}
	found := false
	stack := []*imghash2bkNode{idx.root}
	for len(stack) > 0 {
		node := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		d := node.hash.Hamming(h)
		if !found || d < best.Distance {
			best = Match{ID: node.id, Distance: d}
			found = true
		}
		lo, hi := d-best.Distance, d+best.Distance
		for k, child := range node.children {
			if k >= lo && k <= hi {
				stack = append(stack, child)
			}
		}
	}
	return best, found
}

// stableSortMatches sorts matches ascending by distance with a stable insertion
// sort, preserving discovery order among equal distances.
func stableSortMatches(m []Match) {
	for i := 1; i < len(m); i++ {
		v := m[i]
		j := i - 1
		for j >= 0 && m[j].Distance > v.Distance {
			m[j+1] = m[j]
			j--
		}
		m[j+1] = v
	}
}
