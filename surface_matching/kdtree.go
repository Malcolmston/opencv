package surface_matching

import (
	"math"
	"sort"
)

// KDTree3D is a static, balanced 3-D k-d tree over a set of points. It replaces
// the brute-force O(n) nearest-neighbour scans used elsewhere in the package
// with O(log n) expected queries, which is what makes accelerated ICP
// ([ICP.RegisterKD], [ICP.RegisterMultiScale]), radius-based normal estimation
// ([ComputeNormalsRadius]) and pose scoring ([ScorePose]) practical on larger
// clouds.
//
// The tree is immutable once built and safe for concurrent read-only queries.
// Construction is deterministic: splits are chosen at the median of a stable
// ordering, so the same points always yield the same tree and the same query
// results. Build one with [NewKDTree3D].
type KDTree3D struct {
	points []Vec3
	nodes  []kdNode
	root   int
}

// kdNode is one interior/leaf node of the tree. idx indexes into points; left
// and right are child node indices (-1 when absent); axis is the split axis
// (0=x, 1=y, 2=z) at this node.
type kdNode struct {
	idx         int
	left, right int
	axis        int
}

// Neighbor is a query result: the index of a point in the tree's underlying
// slice and its squared Euclidean distance to the query point.
type Neighbor struct {
	Index  int
	SqDist float64
}

// NewKDTree3D builds a balanced k-d tree over the given points. The slice is
// retained (not copied); indices returned by queries refer to positions in it.
// It returns an empty, queryable tree for a nil or empty slice.
func NewKDTree3D(points []Vec3) *KDTree3D {
	t := &KDTree3D{points: points, root: -1}
	if len(points) == 0 {
		return t
	}
	idx := make([]int, len(points))
	for i := range idx {
		idx[i] = i
	}
	t.nodes = make([]kdNode, 0, len(points))
	t.root = t.build(idx, 0)
	return t
}

// build recursively partitions the index set on the cycling axis, storing nodes
// in t.nodes and returning the node index of the subtree root (-1 when empty).
func (t *KDTree3D) build(idx []int, depth int) int {
	if len(idx) == 0 {
		return -1
	}
	axis := depth % 3
	// Stable median split: sort by the split coordinate, break ties by original
	// index so the tree is fully determined by the input.
	sort.SliceStable(idx, func(a, b int) bool {
		pa, pb := t.points[idx[a]], t.points[idx[b]]
		if pa[axis] != pb[axis] {
			return pa[axis] < pb[axis]
		}
		return idx[a] < idx[b]
	})
	mid := len(idx) / 2
	nodeID := len(t.nodes)
	t.nodes = append(t.nodes, kdNode{idx: idx[mid], axis: axis, left: -1, right: -1})
	left := t.build(idx[:mid], depth+1)
	right := t.build(idx[mid+1:], depth+1)
	t.nodes[nodeID].left = left
	t.nodes[nodeID].right = right
	return nodeID
}

// Len reports how many points the tree indexes.
func (t *KDTree3D) Len() int { return len(t.points) }

// Nearest returns the index of the point closest to q and the squared distance
// to it. It returns (-1, +Inf) for an empty tree.
func (t *KDTree3D) Nearest(q Vec3) (int, float64) {
	best := Neighbor{Index: -1, SqDist: math.Inf(1)}
	if t.root >= 0 {
		t.nearest(t.root, q, &best)
	}
	return best.Index, best.SqDist
}

// nearest is the pruning recursion behind [KDTree3D.Nearest]: descend toward q,
// then only visit the far side of a split when the splitting plane is closer
// than the best distance found so far.
func (t *KDTree3D) nearest(node int, q Vec3, best *Neighbor) {
	if node < 0 {
		return
	}
	n := t.nodes[node]
	d := sqDist(q, t.points[n.idx])
	if d < best.SqDist {
		best.SqDist = d
		best.Index = n.idx
	}
	diff := q[n.axis] - t.points[n.idx][n.axis]
	near, far := n.left, n.right
	if diff > 0 {
		near, far = n.right, n.left
	}
	t.nearest(near, q, best)
	if diff*diff < best.SqDist {
		t.nearest(far, q, best)
	}
}

// NearestK returns up to k neighbours of q ordered by increasing squared
// distance (with equal distances further ordered by point index). It never
// returns more than the tree holds, and returns nil for k <= 0 or an empty
// tree. The result is fully deterministic for a given tree and query.
func (t *KDTree3D) NearestK(q Vec3, k int) []Neighbor {
	if k <= 0 || t.root < 0 {
		return nil
	}
	if k > len(t.points) {
		k = len(t.points)
	}
	h := &neighborHeap{}
	t.nearestK(t.root, q, k, h)
	out := make([]Neighbor, len(h.items))
	copy(out, h.items)
	sort.Slice(out, func(a, b int) bool {
		if out[a].SqDist != out[b].SqDist {
			return out[a].SqDist < out[b].SqDist
		}
		return out[a].Index < out[b].Index
	})
	return out
}

// nearestK is the bounded-heap recursion behind [KDTree3D.NearestK]. It keeps at
// most k candidates and prunes any subtree whose split plane is farther than the
// current worst kept distance once the heap is full.
func (t *KDTree3D) nearestK(node int, q Vec3, k int, h *neighborHeap) {
	if node < 0 {
		return
	}
	n := t.nodes[node]
	d := sqDist(q, t.points[n.idx])
	h.consider(Neighbor{Index: n.idx, SqDist: d}, k)
	diff := q[n.axis] - t.points[n.idx][n.axis]
	near, far := n.left, n.right
	if diff > 0 {
		near, far = n.right, n.left
	}
	t.nearestK(near, q, k, h)
	if len(h.items) < k || diff*diff < h.worst() {
		t.nearestK(far, q, k, h)
	}
}

// RadiusSearch returns the indices of all points within radius of q, ordered by
// increasing distance. A non-positive radius yields no results.
func (t *KDTree3D) RadiusSearch(q Vec3, radius float64) []int {
	if radius <= 0 || t.root < 0 {
		return nil
	}
	r2 := radius * radius
	var found []Neighbor
	t.radius(t.root, q, r2, &found)
	sort.Slice(found, func(a, b int) bool {
		if found[a].SqDist != found[b].SqDist {
			return found[a].SqDist < found[b].SqDist
		}
		return found[a].Index < found[b].Index
	})
	out := make([]int, len(found))
	for i := range found {
		out[i] = found[i].Index
	}
	return out
}

// radius collects, into found, every point within squared radius r2 of q,
// pruning subtrees whose split plane lies beyond the radius.
func (t *KDTree3D) radius(node int, q Vec3, r2 float64, found *[]Neighbor) {
	if node < 0 {
		return
	}
	n := t.nodes[node]
	d := sqDist(q, t.points[n.idx])
	if d <= r2 {
		*found = append(*found, Neighbor{Index: n.idx, SqDist: d})
	}
	diff := q[n.axis] - t.points[n.idx][n.axis]
	near, far := n.left, n.right
	if diff > 0 {
		near, far = n.right, n.left
	}
	t.radius(near, q, r2, found)
	if diff*diff <= r2 {
		t.radius(far, q, r2, found)
	}
}

// neighborHeap is a bounded max-heap keyed on squared distance, used to retain
// the k smallest-distance neighbours during a k-NN search: the largest kept
// distance sits at the root so it can be evicted when a closer point arrives.
type neighborHeap struct {
	items []Neighbor
}

// worst returns the largest squared distance currently held, or +Inf if empty.
func (h *neighborHeap) worst() float64 {
	if len(h.items) == 0 {
		return math.Inf(1)
	}
	return h.items[0].SqDist
}

// consider inserts nb if the heap holds fewer than k items, or replaces the
// current worst when nb is closer, maintaining the max-heap property.
func (h *neighborHeap) consider(nb Neighbor, k int) {
	if len(h.items) < k {
		h.items = append(h.items, nb)
		h.up(len(h.items) - 1)
		return
	}
	if nb.SqDist < h.items[0].SqDist {
		h.items[0] = nb
		h.down(0)
	}
}

// up restores the max-heap order by sifting the element at i toward the root.
func (h *neighborHeap) up(i int) {
	for i > 0 {
		parent := (i - 1) / 2
		if h.items[parent].SqDist >= h.items[i].SqDist {
			break
		}
		h.items[parent], h.items[i] = h.items[i], h.items[parent]
		i = parent
	}
}

// down restores the max-heap order by sifting the element at i toward the leaves.
func (h *neighborHeap) down(i int) {
	n := len(h.items)
	for {
		largest := i
		l, r := 2*i+1, 2*i+2
		if l < n && h.items[l].SqDist > h.items[largest].SqDist {
			largest = l
		}
		if r < n && h.items[r].SqDist > h.items[largest].SqDist {
			largest = r
		}
		if largest == i {
			break
		}
		h.items[i], h.items[largest] = h.items[largest], h.items[i]
		i = largest
	}
}
