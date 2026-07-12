package flann

import "sort"

// defaultKDLeafSize is the number of points below which a k-d tree node becomes
// a leaf that is scanned linearly.
const defaultKDLeafSize = 10

// kdNode is a node of the k-d tree. A leaf holds the indices of the points in
// its bucket; an internal node splits on dimension dim at value split, sending
// points whose dim-th coordinate is < split to left and the rest to right.
type kdNode struct {
	leaf    bool
	indices []int // valid when leaf

	dim   int
	split float64
	left  *kdNode
	right *kdNode
}

// KDTreeIndex is a single median-split k-d tree over a real-valued dataset,
// searched with backtracking. With MaxChecks == 0 (the default) the search is
// exact: its results are identical to those of a float [LinearIndex] over the
// same data. A positive MaxChecks caps the number of points examined during
// backtracking, trading recall for speed — useful in higher dimensions where an
// exact k-d search degenerates towards brute force.
type KDTreeIndex struct {
	// MaxChecks bounds the points examined per search; 0 (the default) means
	// unlimited, i.e. an exact search.
	MaxChecks int

	data     [][]float64
	dim      int
	leafSize int
	root     *kdNode
}

// NewKDTreeIndex builds a k-d tree over the real-valued dataset. It panics if
// the dataset is ragged. An empty dataset is allowed and yields empty searches.
// The tree is exact by default; set [KDTreeIndex.MaxChecks] to bound the search.
func NewKDTreeIndex(data [][]float64) *KDTreeIndex {
	dim := validateFloatData(data, "NewKDTreeIndex")
	t := &KDTreeIndex{data: data, dim: dim, leafSize: defaultKDLeafSize}
	if len(data) > 0 {
		idx := make([]int, len(data))
		for i := range idx {
			idx[i] = i
		}
		t.root = t.build(idx)
	}
	return t
}

// Size returns the number of points in the index.
func (t *KDTreeIndex) Size() int { return len(t.data) }

// build recursively partitions indices into a k-d subtree.
func (t *KDTreeIndex) build(indices []int) *kdNode {
	if len(indices) <= t.leafSize {
		return &kdNode{leaf: true, indices: indices}
	}
	dim := t.maxVarianceDim(indices)

	vals := make([]float64, len(indices))
	for i, id := range indices {
		vals[i] = t.data[id][dim]
	}
	sort.Float64s(vals)
	split := vals[len(vals)/2]

	left := indices[:0:0]
	var right []int
	for _, id := range indices {
		if t.data[id][dim] < split {
			left = append(left, id)
		} else {
			right = append(right, id)
		}
	}
	// A degenerate split (all points share the split coordinate) cannot make
	// progress; fall back to a leaf to guarantee termination.
	if len(left) == 0 || len(right) == 0 {
		return &kdNode{leaf: true, indices: indices}
	}
	return &kdNode{
		dim:   dim,
		split: split,
		left:  t.build(left),
		right: t.build(right),
	}
}

// maxVarianceDim returns the dimension of greatest variance over the given
// points, the classic k-d split heuristic. Ties resolve to the lower index, so
// the choice is deterministic.
func (t *KDTreeIndex) maxVarianceDim(indices []int) int {
	n := float64(len(indices))
	bestDim, bestVar := 0, -1.0
	for d := 0; d < t.dim; d++ {
		var sum, sumSq float64
		for _, id := range indices {
			v := t.data[id][d]
			sum += v
			sumSq += v * v
		}
		mean := sum / n
		variance := sumSq/n - mean*mean
		if variance > bestVar {
			bestVar = variance
			bestDim = d
		}
	}
	return bestDim
}

// KnnSearch returns the k nearest neighbours of query. With MaxChecks == 0 the
// result is exact.
func (t *KDTreeIndex) KnnSearch(query []float64, k int) []Neighbor {
	if k <= 0 || t.root == nil {
		return nil
	}
	res := &knnSet{k: k}
	checks := 0
	t.searchKnn(t.root, query, res, &checks)
	return res.n
}

func (t *KDTreeIndex) searchKnn(node *kdNode, query []float64, res *knnSet, checks *int) {
	if node.leaf {
		for _, id := range node.indices {
			res.add(Neighbor{Index: id, Distance: DistL2(query, t.data[id])})
			*checks++
		}
		return
	}
	diff := query[node.dim] - node.split
	near, far := node.left, node.right
	if diff >= 0 {
		near, far = node.right, node.left
	}
	t.searchKnn(near, query, res, checks)
	if t.MaxChecks > 0 && *checks >= t.MaxChecks {
		return
	}
	// Visit the far side only if the splitting plane lies within the current
	// worst distance (always, while the set is not yet full: worst is +Inf).
	w := res.worst()
	if diff*diff <= w*w {
		t.searchKnn(far, query, res, checks)
	}
}

// RadiusSearch returns every point within radius of query, sorted ascending by
// distance. It ignores MaxChecks and is always exact.
func (t *KDTreeIndex) RadiusSearch(query []float64, radius float64) []Neighbor {
	if t.root == nil {
		return nil
	}
	var out []Neighbor
	t.searchRadius(t.root, query, radius, &out)
	sortNeighbors(out)
	return out
}

func (t *KDTreeIndex) searchRadius(node *kdNode, query []float64, radius float64, out *[]Neighbor) {
	if node.leaf {
		for _, id := range node.indices {
			d := DistL2(query, t.data[id])
			if d <= radius {
				*out = append(*out, Neighbor{Index: id, Distance: d})
			}
		}
		return
	}
	diff := query[node.dim] - node.split
	near, far := node.left, node.right
	if diff >= 0 {
		near, far = node.right, node.left
	}
	t.searchRadius(near, query, radius, out)
	if diff*diff <= radius*radius {
		t.searchRadius(far, query, radius, out)
	}
}
