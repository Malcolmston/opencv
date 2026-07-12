package features2d

import (
	"container/heap"
	"math"
	"math/rand"
	"sort"
)

// FlannBasedMatcher is an approximate nearest-neighbour descriptor matcher built
// on a forest of randomised k-d trees, mirroring OpenCV's cv::FlannBasedMatcher
// with its default KDTreeIndexParams. It targets float descriptors compared with
// the Euclidean ([NormL2]) distance, such as those from [SIFT] and [KAZE]; it
// panics on binary descriptors (use [BFMatcher] with [NormHamming] for those).
//
// Each tree splits on a dimension chosen at random from the few highest-variance
// dimensions and searches with best-bin-first backtracking bounded by a checks
// budget, so results are approximate: with the default budget most, but not
// necessarily all, queries return their true nearest neighbour. Set Checks to 0
// for an exact (exhaustive-fallback) search. Randomisation is driven by a fixed
// seed, so results are deterministic across runs.
//
// The zero value is usable and applies the defaults (4 trees, 32 checks);
// construct a customised instance with [NewFlannBasedMatcher].
type FlannBasedMatcher struct {
	// Trees is the number of randomised k-d trees. Zero means the default (4).
	Trees int
	// Checks bounds the number of leaf points inspected per query. Zero means
	// the default (32). A negative value means exhaustive (exact) search.
	Checks int
}

// NewFlannBasedMatcher returns a FLANN matcher with the given number of trees
// and search-check budget. Pass zero for either to use the defaults.
func NewFlannBasedMatcher(trees, checks int) *FlannBasedMatcher {
	return &FlannBasedMatcher{Trees: trees, Checks: checks}
}

func (m *FlannBasedMatcher) trees() int {
	if m.Trees > 0 {
		return m.Trees
	}
	return 4
}

func (m *FlannBasedMatcher) checks() int {
	if m.Checks != 0 {
		return m.Checks
	}
	return 32
}

// kdNode is a node of a randomised k-d tree. Leaves carry a point index and
// idx >= 0; internal nodes carry a split dimension and value.
type kdNode struct {
	left, right *kdNode
	splitDim    int
	splitVal    float64
	idx         int
}

// flannIndex is a built forest over a set of float points.
type flannIndex struct {
	pts   [][]float64
	trees []*kdNode
}

// buildFlannIndex constructs a forest of numTrees randomised k-d trees over
// pts, seeded deterministically.
func buildFlannIndex(pts [][]float64, numTrees int) *flannIndex {
	rng := rand.New(rand.NewSource(0x71a2b))
	idx := &flannIndex{pts: pts}
	for t := 0; t < numTrees; t++ {
		all := make([]int, len(pts))
		for i := range all {
			all[i] = i
		}
		idx.trees = append(idx.trees, buildKDTree(pts, all, rng))
	}
	return idx
}

// buildKDTree recursively builds one randomised k-d tree over the given point
// indices. It splits on a random dimension among the five highest-variance
// dimensions, at the median value.
func buildKDTree(pts [][]float64, indices []int, rng *rand.Rand) *kdNode {
	if len(indices) == 1 {
		return &kdNode{idx: indices[0], splitDim: -1}
	}
	dim := chooseSplitDim(pts, indices, rng)
	// Sort indices by the chosen dimension to find the median.
	sort.SliceStable(indices, func(i, j int) bool {
		return pts[indices[i]][dim] < pts[indices[j]][dim]
	})
	mid := len(indices) / 2
	node := &kdNode{
		splitDim: dim,
		splitVal: pts[indices[mid]][dim],
		idx:      -1,
	}
	left := make([]int, mid)
	right := make([]int, len(indices)-mid)
	copy(left, indices[:mid])
	copy(right, indices[mid:])
	node.left = buildKDTree(pts, left, rng)
	node.right = buildKDTree(pts, right, rng)
	return node
}

// chooseSplitDim returns a dimension chosen at random from the five with the
// highest variance over the given point subset (FLANN's randomised heuristic).
func chooseSplitDim(pts [][]float64, indices []int, rng *rand.Rand) int {
	d := len(pts[indices[0]])
	mean := make([]float64, d)
	for _, ix := range indices {
		for j := 0; j < d; j++ {
			mean[j] += pts[ix][j]
		}
	}
	n := float64(len(indices))
	for j := 0; j < d; j++ {
		mean[j] /= n
	}
	variance := make([]float64, d)
	for _, ix := range indices {
		for j := 0; j < d; j++ {
			diff := pts[ix][j] - mean[j]
			variance[j] += diff * diff
		}
	}
	type dv struct {
		dim int
		v   float64
	}
	dvs := make([]dv, d)
	for j := 0; j < d; j++ {
		dvs[j] = dv{j, variance[j]}
	}
	sort.SliceStable(dvs, func(i, j int) bool { return dvs[i].v > dvs[j].v })
	top := 5
	if top > d {
		top = d
	}
	return dvs[rng.Intn(top)].dim
}

// branch is a pending subtree to explore in best-bin-first search, keyed by the
// squared distance from the query to the splitting boundary.
type branch struct {
	node    *kdNode
	minDist float64
}

type branchHeap []branch

func (h branchHeap) Len() int            { return len(h) }
func (h branchHeap) Less(i, j int) bool  { return h[i].minDist < h[j].minDist }
func (h branchHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *branchHeap) Push(x interface{}) { *h = append(*h, x.(branch)) }
func (h *branchHeap) Pop() interface{} {
	old := *h
	n := len(old)
	b := old[n-1]
	*h = old[:n-1]
	return b
}

// neighbour is one candidate result (squared distance).
type neighbour struct {
	idx   int
	dist2 float64
}

// knn searches the forest for the k nearest neighbours of query, inspecting at
// most `checks` leaf points (or all of them when checks < 0). It returns
// neighbours sorted by ascending distance.
func (fi *flannIndex) knn(query []float64, k, checks int) []neighbour {
	pq := &branchHeap{}
	heap.Init(pq)
	visited := make(map[int]struct{})
	var best []neighbour // kept sorted ascending, length <= k

	consider := func(idx int) {
		if _, ok := visited[idx]; ok {
			return
		}
		visited[idx] = struct{}{}
		d := squaredL2(query, fi.pts[idx])
		if len(best) < k {
			best = append(best, neighbour{idx, d})
			sort.Slice(best, func(i, j int) bool { return best[i].dist2 < best[j].dist2 })
			return
		}
		if d < best[len(best)-1].dist2 {
			best[len(best)-1] = neighbour{idx, d}
			sort.Slice(best, func(i, j int) bool { return best[i].dist2 < best[j].dist2 })
		}
	}

	// Descend each tree to a leaf, queuing alternate branches.
	descend := func(n *kdNode) {
		for n != nil {
			if n.idx >= 0 {
				consider(n.idx)
				return
			}
			diff := query[n.splitDim] - n.splitVal
			var near, far *kdNode
			if diff <= 0 {
				near, far = n.left, n.right
			} else {
				near, far = n.right, n.left
			}
			heap.Push(pq, branch{node: far, minDist: diff * diff})
			n = near
		}
	}
	for _, root := range fi.trees {
		descend(root)
	}

	budget := checks
	for pq.Len() > 0 {
		if checks >= 0 && len(best) >= k && budget <= 0 {
			break
		}
		b := heap.Pop(pq).(branch)
		// Prune branches that cannot beat the current k-th best.
		if len(best) >= k && b.minDist > best[len(best)-1].dist2 {
			continue
		}
		descend(b.node)
		budget--
	}
	return best
}

// squaredL2 returns the squared Euclidean distance between equal-length vectors.
func squaredL2(a, b []float64) float64 {
	var s float64
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

func (m *FlannBasedMatcher) validate(query, train Descriptors) {
	if query.Float == nil || train.Float == nil {
		panic("features2d: FlannBasedMatcher requires float descriptors (use BFMatcher for binary)")
	}
}

// Match returns the single best train match for each query descriptor, in query
// order. It panics if the descriptors are not float.
func (m *FlannBasedMatcher) Match(query, train Descriptors) []DMatch {
	if query.Len() == 0 || train.Len() == 0 {
		return nil
	}
	m.validate(query, train)
	fi := buildFlannIndex(train.Float, m.trees())
	out := make([]DMatch, 0, query.Len())
	for qi := 0; qi < query.Len(); qi++ {
		res := fi.knn(query.Float[qi], 1, m.checks())
		if len(res) == 0 {
			continue
		}
		out = append(out, DMatch{QueryIdx: qi, TrainIdx: res[0].idx, Distance: math.Sqrt(res[0].dist2)})
	}
	return out
}

// KnnMatch returns, for each query descriptor, up to its k approximate nearest
// train descriptors sorted by ascending distance. The outer slice is in query
// order. It panics if k < 1 or the descriptors are not float.
func (m *FlannBasedMatcher) KnnMatch(query, train Descriptors, k int) [][]DMatch {
	if k < 1 {
		panic("features2d: KnnMatch requires k >= 1")
	}
	if query.Len() == 0 || train.Len() == 0 {
		return nil
	}
	m.validate(query, train)
	fi := buildFlannIndex(train.Float, m.trees())
	out := make([][]DMatch, query.Len())
	for qi := 0; qi < query.Len(); qi++ {
		res := fi.knn(query.Float[qi], k, m.checks())
		row := make([]DMatch, len(res))
		for i, r := range res {
			row[i] = DMatch{QueryIdx: qi, TrainIdx: r.idx, Distance: math.Sqrt(r.dist2)}
		}
		out[qi] = row
	}
	return out
}
