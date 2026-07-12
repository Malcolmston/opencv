package flann

import (
	"container/heap"
	"math/rand"
)

// Default construction parameters for a hierarchical k-means tree.
const (
	defaultBranching  = 8
	defaultKMLeafSize = 16
	defaultKMMaxIter  = 11
)

// kmeansNode is a node of the hierarchical k-means tree. Every node stores the
// centre (mean) of the points beneath it, used to order the best-bin-first
// traversal. A leaf additionally holds the indices of its points; an internal
// node holds its child clusters.
type kmeansNode struct {
	center   []float64
	leaf     bool
	indices  []int         // valid when leaf
	children []*kmeansNode // valid when internal
}

// KMeansIndex is a hierarchical k-means tree over a real-valued dataset. The
// points are recursively partitioned into Branching clusters by Lloyd's
// algorithm and queries descend the tree best-bin-first. With Checks == 0 (the
// default) the search is exhaustive and therefore exact; a positive Checks
// bounds the number of points examined, yielding an approximate result whose
// recall rises with Checks.
type KMeansIndex struct {
	// Checks bounds the points examined per search; 0 (the default) means
	// unlimited, i.e. an exact search.
	Checks int

	data      [][]float64
	dim       int
	branching int
	leafSize  int
	maxIter   int
	root      *kmeansNode
}

// NewKMeansIndex builds a hierarchical k-means tree over the real-valued
// dataset. branching is the number of clusters per internal node (>= 2) and
// leafSize is the point count below which a node becomes a leaf (>= 1); pass 0
// for either to use sensible defaults (8 and 16). Clustering is seeded by seed
// so the tree is reproducible. It panics if the dataset is ragged. An empty
// dataset is allowed and yields empty searches.
func NewKMeansIndex(data [][]float64, branching, leafSize int, seed int64) *KMeansIndex {
	dim := validateFloatData(data, "NewKMeansIndex")
	if branching <= 0 {
		branching = defaultBranching
	}
	if branching < 2 {
		branching = 2
	}
	if leafSize <= 0 {
		leafSize = defaultKMLeafSize
	}
	idx := &KMeansIndex{
		data:      data,
		dim:       dim,
		branching: branching,
		leafSize:  leafSize,
		maxIter:   defaultKMMaxIter,
	}
	if len(data) > 0 {
		all := make([]int, len(data))
		for i := range all {
			all[i] = i
		}
		rng := rand.New(rand.NewSource(seed))
		idx.root = idx.build(all, rng)
	}
	return idx
}

// Size returns the number of points in the index.
func (idx *KMeansIndex) Size() int { return len(idx.data) }

// build recursively clusters indices into a k-means subtree.
func (idx *KMeansIndex) build(indices []int, rng *rand.Rand) *kmeansNode {
	center := idx.mean(indices)
	if len(indices) <= idx.leafSize || len(indices) <= idx.branching {
		return &kmeansNode{center: center, leaf: true, indices: indices}
	}
	clusters := idx.cluster(indices, rng)
	// If clustering could not separate the points (only one non-empty
	// cluster), stop here to guarantee termination.
	nonEmpty := 0
	for _, c := range clusters {
		if len(c) > 0 {
			nonEmpty++
		}
	}
	if nonEmpty <= 1 {
		return &kmeansNode{center: center, leaf: true, indices: indices}
	}
	node := &kmeansNode{center: center}
	for _, c := range clusters {
		if len(c) == 0 {
			continue
		}
		node.children = append(node.children, idx.build(c, rng))
	}
	return node
}

// cluster runs k-means (k-means++ seeding, then Lloyd refinement) on the given
// points and returns the resulting clusters as index lists.
func (idx *KMeansIndex) cluster(indices []int, rng *rand.Rand) [][]int {
	k := idx.branching
	centroids := idx.kmeansPP(indices, k, rng)

	assign := make([]int, len(indices))
	for iter := 0; iter < idx.maxIter; iter++ {
		changed := false
		for i, id := range indices {
			best, bestDist := 0, positiveInf
			for c := range centroids {
				d := distL2Sq(idx.data[id], centroids[c])
				if d < bestDist {
					bestDist, best = d, c
				}
			}
			if assign[i] != best {
				assign[i] = best
				changed = true
			}
		}
		// Recompute centroids as the mean of their members.
		sums := make([][]float64, k)
		counts := make([]int, k)
		for c := range sums {
			sums[c] = make([]float64, idx.dim)
		}
		for i, id := range indices {
			c := assign[i]
			counts[c]++
			for d := 0; d < idx.dim; d++ {
				sums[c][d] += idx.data[id][d]
			}
		}
		for c := range centroids {
			if counts[c] == 0 {
				continue // keep the previous centroid for an empty cluster
			}
			for d := 0; d < idx.dim; d++ {
				centroids[c][d] = sums[c][d] / float64(counts[c])
			}
		}
		if !changed && iter > 0 {
			break
		}
	}

	clusters := make([][]int, k)
	for i, id := range indices {
		clusters[assign[i]] = append(clusters[assign[i]], id)
	}
	return clusters
}

// kmeansPP chooses k initial centroids from the points using the k-means++
// weighting, returning independent copies so refinement does not mutate the
// dataset.
func (idx *KMeansIndex) kmeansPP(indices []int, k int, rng *rand.Rand) [][]float64 {
	centroids := make([][]float64, 0, k)
	first := indices[rng.Intn(len(indices))]
	centroids = append(centroids, cloneVec(idx.data[first]))

	d2 := make([]float64, len(indices))
	for len(centroids) < k {
		var sum float64
		last := centroids[len(centroids)-1]
		for i, id := range indices {
			dist := distL2Sq(idx.data[id], last)
			if len(centroids) == 1 || dist < d2[i] {
				d2[i] = dist
			}
			sum += d2[i]
		}
		if sum == 0 {
			// All remaining points coincide with a chosen centroid; pad with a
			// copy so we always return k centroids.
			centroids = append(centroids, cloneVec(idx.data[indices[0]]))
			continue
		}
		target := rng.Float64() * sum
		chosen := indices[len(indices)-1]
		var acc float64
		for i, id := range indices {
			acc += d2[i]
			if acc >= target {
				chosen = id
				break
			}
		}
		centroids = append(centroids, cloneVec(idx.data[chosen]))
	}
	return centroids
}

// mean returns the coordinate-wise mean of the given points.
func (idx *KMeansIndex) mean(indices []int) []float64 {
	m := make([]float64, idx.dim)
	if len(indices) == 0 {
		return m
	}
	for _, id := range indices {
		for d := 0; d < idx.dim; d++ {
			m[d] += idx.data[id][d]
		}
	}
	for d := range m {
		m[d] /= float64(len(indices))
	}
	return m
}

func cloneVec(v []float64) []float64 {
	out := make([]float64, len(v))
	copy(out, v)
	return out
}

// branchItem is a pending subtree to explore, keyed by the distance from the
// query to the subtree's centre.
type branchItem struct {
	node *kmeansNode
	dist float64
}

// branchHeap is a min-heap of pending subtrees, nearest centre first.
type branchHeap []branchItem

func (h branchHeap) Len() int           { return len(h) }
func (h branchHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h branchHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *branchHeap) Push(x any)        { *h = append(*h, x.(branchItem)) }
func (h *branchHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

// KnnSearch returns the k nearest neighbours of query, traversing the tree
// best-bin-first. With Checks == 0 the traversal is exhaustive and the result
// is exact.
func (idx *KMeansIndex) KnnSearch(query []float64, k int) []Neighbor {
	if k <= 0 || idx.root == nil {
		return nil
	}
	res := &knnSet{k: k}
	checks := 0
	pq := &branchHeap{{node: idx.root, dist: DistL2(query, idx.root.center)}}
	for pq.Len() > 0 {
		if idx.Checks > 0 && checks >= idx.Checks {
			break
		}
		it := heap.Pop(pq).(branchItem)
		node := it.node
		if node.leaf {
			for _, id := range node.indices {
				res.add(Neighbor{Index: id, Distance: DistL2(query, idx.data[id])})
				checks++
			}
			continue
		}
		for _, child := range node.children {
			heap.Push(pq, branchItem{node: child, dist: DistL2(query, child.center)})
		}
	}
	return res.n
}

// RadiusSearch returns every point within radius of query. It ignores Checks
// and is always exact: every leaf is visited.
func (idx *KMeansIndex) RadiusSearch(query []float64, radius float64) []Neighbor {
	if idx.root == nil {
		return nil
	}
	var out []Neighbor
	idx.radius(idx.root, query, radius, &out)
	sortNeighbors(out)
	return out
}

func (idx *KMeansIndex) radius(node *kmeansNode, query []float64, radius float64, out *[]Neighbor) {
	if node.leaf {
		for _, id := range node.indices {
			d := DistL2(query, idx.data[id])
			if d <= radius {
				*out = append(*out, Neighbor{Index: id, Distance: d})
			}
		}
		return
	}
	for _, child := range node.children {
		idx.radius(child, query, radius, out)
	}
}
