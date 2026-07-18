package matching2

import (
	"math"
	"sort"
)

// KDTree is a static k-d tree over float vectors (descriptors or points) that
// answers nearest-neighbour, k-nearest and radius queries under the squared
// Euclidean distance. It is the exact search structure behind the approximate
// [FLANNMatcher]. Build one with [BuildKDTree]; it is read-only and safe for
// concurrent queries afterwards.
type KDTree struct {
	points [][]float64
	dim    int
	nodes  []matching2kdNode
	root   int
}

// matching2kdNode is one node of the k-d tree. Leaf nodes have left == right ==
// -1 and reference a single point via idx.
type matching2kdNode struct {
	idx         int
	axis        int
	left, right int
}

// BuildKDTree constructs a balanced k-d tree over the given points, which must
// all share the same non-zero dimension. It panics if the dimensions differ.
// Building an empty set returns a tree that answers every query with "not
// found".
func BuildKDTree(points [][]float64) *KDTree {
	t := &KDTree{root: -1}
	if len(points) == 0 {
		return t
	}
	t.dim = len(points[0])
	t.points = make([][]float64, len(points))
	for i, p := range points {
		if len(p) != t.dim {
			panic("matching2: BuildKDTree dimension mismatch")
		}
		t.points[i] = p
	}
	idx := make([]int, len(points))
	for i := range idx {
		idx[i] = i
	}
	t.root = t.build(idx, 0)
	return t
}

// build recursively partitions idx along the median of the current axis and
// returns the index of the created node.
func (t *KDTree) build(idx []int, depth int) int {
	if len(idx) == 0 {
		return -1
	}
	axis := depth % t.dim
	sort.SliceStable(idx, func(i, j int) bool {
		if t.points[idx[i]][axis] != t.points[idx[j]][axis] {
			return t.points[idx[i]][axis] < t.points[idx[j]][axis]
		}
		return idx[i] < idx[j]
	})
	mid := len(idx) / 2
	node := matching2kdNode{idx: idx[mid], axis: axis}
	id := len(t.nodes)
	t.nodes = append(t.nodes, node)
	left := t.build(idx[:mid], depth+1)
	right := t.build(idx[mid+1:], depth+1)
	t.nodes[id].left = left
	t.nodes[id].right = right
	return id
}

// Size returns the number of points stored in the tree.
func (t *KDTree) Size() int { return len(t.points) }

// Dim returns the dimension of the stored points, or 0 for an empty tree.
func (t *KDTree) Dim() int { return t.dim }

// Nearest returns the index of the point closest to query and the squared
// Euclidean distance to it. It returns (-1, +Inf) for an empty tree. query must
// have the tree's dimension.
func (t *KDTree) Nearest(query []float64) (int, float64) {
	if t.root < 0 {
		return -1, math.Inf(1)
	}
	best, bestD := -1, math.Inf(1)
	t.searchNearest(t.root, query, &best, &bestD)
	return best, bestD
}

// searchNearest walks the tree with backtracking to find the single nearest
// neighbour, pruning subtrees whose splitting plane is farther than the current
// best.
func (t *KDTree) searchNearest(node int, q []float64, best *int, bestD *float64) {
	if node < 0 {
		return
	}
	n := t.nodes[node]
	d := sqDist(t.points[n.idx], q)
	if d < *bestD {
		*bestD, *best = d, n.idx
	}
	diff := q[n.axis] - t.points[n.idx][n.axis]
	var near, far int
	if diff < 0 {
		near, far = n.left, n.right
	} else {
		near, far = n.right, n.left
	}
	t.searchNearest(near, q, best, bestD)
	if diff*diff < *bestD {
		t.searchNearest(far, q, best, bestD)
	}
}

// KNearest returns the indices of the k points closest to query together with
// their squared distances, both ordered nearest-first. If k exceeds the tree
// size, every point is returned. Empty trees yield nil slices.
func (t *KDTree) KNearest(query []float64, k int) (indices []int, distances []float64) {
	if t.root < 0 || k <= 0 {
		return nil, nil
	}
	h := &matching2kHeap{}
	t.searchKNN(t.root, query, k, h)
	res := h.sorted()
	indices = make([]int, len(res))
	distances = make([]float64, len(res))
	for i, e := range res {
		indices[i] = e.idx
		distances[i] = e.dist
	}
	return indices, distances
}

// searchKNN walks the tree maintaining a bounded max-heap of the k best
// candidates seen so far, pruning by the current worst distance.
func (t *KDTree) searchKNN(node int, q []float64, k int, h *matching2kHeap) {
	if node < 0 {
		return
	}
	n := t.nodes[node]
	d := sqDist(t.points[n.idx], q)
	h.push(matching2kEntry{idx: n.idx, dist: d}, k)
	diff := q[n.axis] - t.points[n.idx][n.axis]
	var near, far int
	if diff < 0 {
		near, far = n.left, n.right
	} else {
		near, far = n.right, n.left
	}
	t.searchKNN(near, q, k, h)
	if h.len() < k || diff*diff < h.worst() {
		t.searchKNN(far, q, k, h)
	}
}

// RadiusSearch returns the indices of every point within radius of query and
// their squared distances, ordered nearest-first. radius is a true (non-squared)
// distance. Empty results yield nil slices.
func (t *KDTree) RadiusSearch(query []float64, radius float64) (indices []int, distances []float64) {
	if t.root < 0 || radius < 0 {
		return nil, nil
	}
	r2 := radius * radius
	var entries []matching2kEntry
	t.searchRadius(t.root, query, r2, &entries)
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].dist != entries[j].dist {
			return entries[i].dist < entries[j].dist
		}
		return entries[i].idx < entries[j].idx
	})
	indices = make([]int, len(entries))
	distances = make([]float64, len(entries))
	for i, e := range entries {
		indices[i] = e.idx
		distances[i] = e.dist
	}
	return indices, distances
}

// searchRadius collects every point whose squared distance to q is within r2.
func (t *KDTree) searchRadius(node int, q []float64, r2 float64, out *[]matching2kEntry) {
	if node < 0 {
		return
	}
	n := t.nodes[node]
	d := sqDist(t.points[n.idx], q)
	if d <= r2 {
		*out = append(*out, matching2kEntry{idx: n.idx, dist: d})
	}
	diff := q[n.axis] - t.points[n.idx][n.axis]
	var near, far int
	if diff < 0 {
		near, far = n.left, n.right
	} else {
		near, far = n.right, n.left
	}
	t.searchRadius(near, q, r2, out)
	if diff*diff <= r2 {
		t.searchRadius(far, q, r2, out)
	}
}

// sqDist returns the squared Euclidean distance between two equal-length
// vectors.
func sqDist(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// matching2kEntry is a candidate neighbour: a point index and its squared
// distance to the query.
type matching2kEntry struct {
	idx  int
	dist float64
}

// matching2kHeap is a bounded max-heap (by distance) used to retain the k
// nearest candidates during a k-NN search.
type matching2kHeap struct {
	data []matching2kEntry
}

func (h *matching2kHeap) len() int { return len(h.data) }

// worst returns the largest distance currently retained, or +Inf when empty.
func (h *matching2kHeap) worst() float64 {
	if len(h.data) == 0 {
		return math.Inf(1)
	}
	return h.data[0].dist
}

// push inserts e, keeping at most k entries by evicting the current worst.
func (h *matching2kHeap) push(e matching2kEntry, k int) {
	if len(h.data) < k {
		h.data = append(h.data, e)
		h.up(len(h.data) - 1)
		return
	}
	if e.dist < h.data[0].dist {
		h.data[0] = e
		h.down(0)
	}
}

func (h *matching2kHeap) up(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.data[parent].dist >= h.data[i].dist {
			break
		}
		h.data[parent], h.data[i] = h.data[i], h.data[parent]
		i = parent
	}
}

func (h *matching2kHeap) down(i int) {
	n := len(h.data)
	for {
		l, r, largest := 2*i+1, 2*i+2, i
		if l < n && h.data[l].dist > h.data[largest].dist {
			largest = l
		}
		if r < n && h.data[r].dist > h.data[largest].dist {
			largest = r
		}
		if largest == i {
			break
		}
		h.data[i], h.data[largest] = h.data[largest], h.data[i]
		i = largest
	}
}

// sorted returns the retained entries ordered nearest-first, breaking ties by
// lower index for determinism.
func (h *matching2kHeap) sorted() []matching2kEntry {
	out := make([]matching2kEntry, len(h.data))
	copy(out, h.data)
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].dist != out[j].dist {
			return out[i].dist < out[j].dist
		}
		return out[i].idx < out[j].idx
	})
	return out
}

// FLANNMatcher matches float descriptors using a [KDTree] built over the train
// set, giving the same results as a brute-force L2 matcher but with sub-linear
// average query cost. Construct one with [NewFLANNMatcher].
type FLANNMatcher struct {
	tree *KDTree
}

// NewFLANNMatcher builds a kd-tree index over the train descriptors and returns
// a matcher that queries it. All train descriptors must share one dimension.
func NewFLANNMatcher(train [][]float64) *FLANNMatcher {
	return &FLANNMatcher{tree: BuildKDTree(train)}
}

// Match returns the single nearest train descriptor for each query descriptor,
// one [DMatch] per query in query order. Distances are true Euclidean distances.
func (f *FLANNMatcher) Match(query [][]float64) []DMatch {
	if f.tree.Size() == 0 || len(query) == 0 {
		return nil
	}
	out := make([]DMatch, len(query))
	for qi := range query {
		ti, d2 := f.tree.Nearest(query[qi])
		out[qi] = DMatch{QueryIdx: qi, TrainIdx: ti, Distance: math.Sqrt(d2)}
	}
	return out
}

// KnnMatch returns, for each query descriptor, its k nearest train descriptors
// sorted by ascending Euclidean distance, in query order.
func (f *FLANNMatcher) KnnMatch(query [][]float64, k int) [][]DMatch {
	if f.tree.Size() == 0 || len(query) == 0 || k <= 0 {
		return nil
	}
	out := make([][]DMatch, len(query))
	for qi := range query {
		idxs, d2 := f.tree.KNearest(query[qi], k)
		ms := make([]DMatch, len(idxs))
		for j := range idxs {
			ms[j] = DMatch{QueryIdx: qi, TrainIdx: idxs[j], Distance: math.Sqrt(d2[j])}
		}
		out[qi] = ms
	}
	return out
}
