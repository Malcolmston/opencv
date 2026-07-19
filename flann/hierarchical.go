package flann

import (
	"container/heap"
	"math/rand"
)

// Default construction parameters for a hierarchical clustering tree.
const (
	defaultHCLBranching = 8
	defaultHCLLeafSize  = 16
	defaultHCLTrees     = 1
)

// hclNode is a node of a hierarchical clustering tree. Every node stores the
// cluster centre (an actual dataset point) used to order the best-bin-first
// traversal; a leaf additionally holds its point indices.
type hclNode struct {
	center   []float64
	leaf     bool
	indices  []int
	children []*hclNode
}

// HierarchicalClusteringIndex partitions a real-valued dataset by recursive
// clustering: at each node it picks Branching cluster centres at random from the
// points beneath it, assigns every point to its nearest centre, and recurses
// until a node holds no more than LeafSize points. Unlike [KMeansIndex] the
// centres are real data points and are never refined, which makes construction
// cheap and lets the index work under any [DistanceFunc], not just L2. Several
// independently randomized trees can be built and searched together to raise
// recall.
//
// Queries descend best-bin-first under a Checks budget. With Checks == 0 (the
// default) the traversal visits every leaf and is therefore exact.
type HierarchicalClusteringIndex struct {
	// Checks bounds the points examined per search; 0 (the default) means
	// unlimited, i.e. an exact search.
	Checks int

	data      [][]float64
	dim       int
	branching int
	leafSize  int
	numTrees  int
	seed      int64
	dist      DistanceFunc[[]float64]
	roots     []*hclNode
}

// NewHierarchicalClusteringIndex builds a hierarchical clustering tree under the
// Euclidean ([DistL2]) distance. branching is the number of clusters per node
// (>= 2), leafSize the point count below which a node becomes a leaf (>= 1) and
// trees the number of independently randomized trees (>= 1); pass 0 for any of
// them to use sensible defaults (8, 16, 1). seed makes construction
// reproducible. It panics if the dataset is ragged. An empty dataset is allowed
// and yields empty searches.
func NewHierarchicalClusteringIndex(data [][]float64, branching, leafSize, trees int, seed int64) *HierarchicalClusteringIndex {
	return NewHierarchicalClusteringIndexFunc(data, branching, leafSize, trees, seed, DistL2)
}

// NewHierarchicalClusteringIndexFunc is [NewHierarchicalClusteringIndex] with an
// explicit [DistanceFunc], so the tree can cluster and search under L1, cosine,
// chi-square or any other metric. It panics if the dataset is ragged or dist is
// nil.
func NewHierarchicalClusteringIndexFunc(data [][]float64, branching, leafSize, trees int, seed int64, dist DistanceFunc[[]float64]) *HierarchicalClusteringIndex {
	dim := validateFloatData(data, "NewHierarchicalClusteringIndex")
	if dist == nil {
		panic("flann: NewHierarchicalClusteringIndexFunc requires a non-nil distance function")
	}
	if branching <= 0 {
		branching = defaultHCLBranching
	}
	if branching < 2 {
		branching = 2
	}
	if leafSize <= 0 {
		leafSize = defaultHCLLeafSize
	}
	if trees <= 0 {
		trees = defaultHCLTrees
	}
	h := &HierarchicalClusteringIndex{
		data:      data,
		dim:       dim,
		branching: branching,
		leafSize:  leafSize,
		numTrees:  trees,
		seed:      seed,
		dist:      dist,
	}
	h.buildAll()
	return h
}

// buildAll constructs every tree from the current data/params.
func (h *HierarchicalClusteringIndex) buildAll() {
	h.roots = nil
	if len(h.data) == 0 {
		return
	}
	for t := 0; t < h.numTrees; t++ {
		all := make([]int, len(h.data))
		for i := range all {
			all[i] = i
		}
		rng := rand.New(rand.NewSource(h.seed + int64(t)*0x27D4EB2F))
		h.roots = append(h.roots, h.build(all, rng))
	}
}

// Size returns the number of points in the index.
func (h *HierarchicalClusteringIndex) Size() int { return len(h.data) }

// Trees returns the number of randomized trees in the index.
func (h *HierarchicalClusteringIndex) Trees() int { return h.numTrees }

// build recursively clusters indices into one subtree. center is the point
// representing this node in the parent's traversal.
func (h *HierarchicalClusteringIndex) build(indices []int, rng *rand.Rand) *hclNode {
	center := cloneVec(h.data[indices[0]])
	if len(indices) <= h.leafSize || len(indices) <= h.branching {
		return &hclNode{center: center, leaf: true, indices: indices}
	}
	centers := h.pickCenters(indices, rng)
	clusters := make([][]int, len(centers))
	for _, id := range indices {
		best, bestD := 0, positiveInf
		for c, ctr := range centers {
			if d := h.dist(h.data[id], ctr); d < bestD {
				bestD, best = d, c
			}
		}
		clusters[best] = append(clusters[best], id)
	}
	nonEmpty := 0
	for _, c := range clusters {
		if len(c) > 0 {
			nonEmpty++
		}
	}
	if nonEmpty <= 1 {
		return &hclNode{center: center, leaf: true, indices: indices}
	}
	node := &hclNode{center: center}
	for c, cl := range clusters {
		if len(cl) == 0 {
			continue
		}
		child := h.build(cl, rng)
		// Anchor the child's traversal centre at its chosen cluster centre.
		child.center = cloneVec(centers[c])
		node.children = append(node.children, child)
	}
	return node
}

// pickCenters chooses up to branching distinct points as cluster centres via a
// partial Fisher-Yates shuffle, so the choice is unbiased and reproducible.
func (h *HierarchicalClusteringIndex) pickCenters(indices []int, rng *rand.Rand) [][]float64 {
	k := h.branching
	if k > len(indices) {
		k = len(indices)
	}
	perm := make([]int, len(indices))
	copy(perm, indices)
	for i := 0; i < k; i++ {
		j := i + rng.Intn(len(perm)-i)
		perm[i], perm[j] = perm[j], perm[i]
	}
	centers := make([][]float64, k)
	for i := 0; i < k; i++ {
		centers[i] = cloneVec(h.data[perm[i]])
	}
	return centers
}

// hclBranch is a pending subtree keyed by distance from the query to its centre.
type hclBranch struct {
	node *hclNode
	dist float64
}

type hclBranchHeap []hclBranch

// Len reports the number of pending branches in the heap, implementing
// sort.Interface as required by container/heap.
func (q hclBranchHeap) Len() int { return len(q) }

// Less reports whether branch i is closer to the query than branch j, ordering
// the heap as a min-heap on dist so the nearest centre is popped first.
func (q hclBranchHeap) Less(i, j int) bool { return q[i].dist < q[j].dist }

// Swap exchanges branches i and j, implementing sort.Interface as required by
// container/heap.
func (q hclBranchHeap) Swap(i, j int) { q[i], q[j] = q[j], q[i] }

// Push appends x, which must be an hclBranch, to the heap. It implements
// container/heap.Interface and is invoked through heap.Push rather than directly.
func (q *hclBranchHeap) Push(x any) { *q = append(*q, x.(hclBranch)) }

// Pop removes and returns the last branch in the underlying slice. It implements
// container/heap.Interface and is invoked through heap.Pop rather than directly.
func (q *hclBranchHeap) Pop() any {
	old := *q
	n := len(old)
	it := old[n-1]
	*q = old[:n-1]
	return it
}

// KnnSearch returns the k nearest neighbours of query, traversing every tree
// best-bin-first through a shared queue. With Checks == 0 the result is exact.
func (h *HierarchicalClusteringIndex) KnnSearch(query []float64, k int) []Neighbor {
	if k <= 0 || len(h.roots) == 0 {
		return nil
	}
	res := &knnSet{k: k}
	seen := make([]bool, len(h.data))
	pq := &hclBranchHeap{}
	for _, root := range h.roots {
		*pq = append(*pq, hclBranch{node: root, dist: h.dist(query, root.center)})
	}
	heap.Init(pq)

	checks := 0
	for pq.Len() > 0 {
		if h.Checks > 0 && checks >= h.Checks {
			break
		}
		it := heap.Pop(pq).(hclBranch)
		node := it.node
		if node.leaf {
			for _, id := range node.indices {
				if seen[id] {
					continue
				}
				seen[id] = true
				res.add(Neighbor{Index: id, Distance: h.dist(query, h.data[id])})
				checks++
			}
			continue
		}
		for _, child := range node.children {
			heap.Push(pq, hclBranch{node: child, dist: h.dist(query, child.center)})
		}
	}
	return res.n
}

// RadiusSearch returns every point within radius of query, sorted ascending by
// distance. It ignores Checks and is always exact.
func (h *HierarchicalClusteringIndex) RadiusSearch(query []float64, radius float64) []Neighbor {
	if len(h.roots) == 0 {
		return nil
	}
	seen := make([]bool, len(h.data))
	var out []Neighbor
	for _, root := range h.roots {
		h.radius(root, query, radius, seen, &out)
	}
	sortNeighbors(out)
	return out
}

func (h *HierarchicalClusteringIndex) radius(node *hclNode, query []float64, radius float64, seen []bool, out *[]Neighbor) {
	if node.leaf {
		for _, id := range node.indices {
			if seen[id] {
				continue
			}
			seen[id] = true
			if d := h.dist(query, h.data[id]); d <= radius {
				*out = append(*out, Neighbor{Index: id, Distance: d})
			}
		}
		return
	}
	for _, child := range node.children {
		h.radius(child, query, radius, seen, out)
	}
}
