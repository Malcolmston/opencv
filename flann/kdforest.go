package flann

import (
	"container/heap"
	"math/rand"
	"sort"
)

// defaultForestTrees is the number of randomized trees a KD-forest builds when
// the caller passes 0.
const defaultForestTrees = 4

// randDimCandidates is how many of the highest-variance dimensions a randomized
// tree draws its split dimension from. Randomizing the split among the top few
// variance dimensions is what decorrelates the trees so that, together, they
// explore complementary regions of the space.
const randDimCandidates = 5

// kdRandNode is a node of one randomized k-d tree. Leaves hold point indices;
// internal nodes split dimension dim at value split.
type kdRandNode struct {
	leaf    bool
	indices []int

	dim   int
	split float64
	left  *kdRandNode
	right *kdRandNode
}

// KDForestIndex is a randomized multi-tree k-d forest over a real-valued
// dataset, the classic FLANN structure for approximate nearest-neighbour search
// in moderate to high dimension. Each tree indexes the whole dataset but splits
// on a dimension chosen randomly from the few of highest variance, so the trees
// are decorrelated; a query is searched across all of them with a single shared
// priority queue (best-bin-first) under one combined check budget.
//
// With MaxChecks == 0 (the default) the search is exhaustive with correct
// lower-bound pruning and therefore exact: its results match a float
// [LinearIndex]. A positive MaxChecks caps the points examined, trading recall
// for speed; more trees raise recall at a given budget.
type KDForestIndex struct {
	// MaxChecks bounds the points examined per search; 0 (the default) means
	// unlimited, i.e. an exact search.
	MaxChecks int

	data     [][]float64
	dim      int
	leafSize int
	numTrees int
	seed     int64
	trees    []*kdRandNode
}

// NewKDForestIndex builds a randomized k-d forest over the real-valued dataset.
// trees is the number of randomized trees (>= 1; pass 0 for the default of 4)
// and seed makes the randomized splits reproducible. It panics if the dataset
// is ragged. An empty dataset is allowed and yields empty searches.
func NewKDForestIndex(data [][]float64, trees int, seed int64) *KDForestIndex {
	dim := validateFloatData(data, "NewKDForestIndex")
	if trees <= 0 {
		trees = defaultForestTrees
	}
	f := &KDForestIndex{
		data:     data,
		dim:      dim,
		leafSize: defaultKDLeafSize,
		numTrees: trees,
		seed:     seed,
	}
	f.buildAll()
	return f
}

// buildAll constructs every tree from the current data/params, seeded so that
// each tree draws an independent but reproducible sequence of split choices.
func (f *KDForestIndex) buildAll() {
	f.trees = nil
	if len(f.data) == 0 {
		return
	}
	for t := 0; t < f.numTrees; t++ {
		idx := make([]int, len(f.data))
		for i := range idx {
			idx[i] = i
		}
		rng := rand.New(rand.NewSource(f.seed + int64(t)*0x9E3779B1))
		f.trees = append(f.trees, f.build(idx, rng))
	}
}

// Size returns the number of points in the index.
func (f *KDForestIndex) Size() int { return len(f.data) }

// Trees returns the number of randomized trees in the forest.
func (f *KDForestIndex) Trees() int { return f.numTrees }

// build recursively partitions indices into one randomized subtree.
func (f *KDForestIndex) build(indices []int, rng *rand.Rand) *kdRandNode {
	if len(indices) <= f.leafSize {
		return &kdRandNode{leaf: true, indices: indices}
	}
	dim := f.randomSplitDim(indices, rng)

	vals := make([]float64, len(indices))
	for i, id := range indices {
		vals[i] = f.data[id][dim]
	}
	sort.Float64s(vals)
	split := vals[len(vals)/2]

	left := indices[:0:0]
	var right []int
	for _, id := range indices {
		if f.data[id][dim] < split {
			left = append(left, id)
		} else {
			right = append(right, id)
		}
	}
	if len(left) == 0 || len(right) == 0 {
		return &kdRandNode{leaf: true, indices: indices}
	}
	return &kdRandNode{
		dim:   dim,
		split: split,
		left:  f.build(left, rng),
		right: f.build(right, rng),
	}
}

// randomSplitDim picks the split dimension at random among the highest-variance
// dimensions, the source of the forest's decorrelation.
func (f *KDForestIndex) randomSplitDim(indices []int, rng *rand.Rand) int {
	type dv struct {
		dim int
		v   float64
	}
	n := float64(len(indices))
	vs := make([]dv, f.dim)
	for d := 0; d < f.dim; d++ {
		var sum, sumSq float64
		for _, id := range indices {
			v := f.data[id][d]
			sum += v
			sumSq += v * v
		}
		mean := sum / n
		vs[d] = dv{dim: d, v: sumSq/n - mean*mean}
	}
	sort.SliceStable(vs, func(i, j int) bool { return vs[i].v > vs[j].v })
	cand := randDimCandidates
	if cand > f.dim {
		cand = f.dim
	}
	return vs[rng.Intn(cand)].dim
}

// kdBranch is a pending subtree keyed by a lower bound on the distance from the
// query to any point it contains.
type kdBranch struct {
	node   *kdRandNode
	mindsq float64
}

// kdBranchHeap is a min-heap of pending subtrees, smallest lower bound first.
type kdBranchHeap []kdBranch

// Len reports the number of pending subtrees in the heap, implementing
// sort.Interface as required by container/heap.
func (h kdBranchHeap) Len() int { return len(h) }

// Less reports whether subtree i has a smaller distance lower bound than
// subtree j, ordering the heap as a min-heap on mindsq.
func (h kdBranchHeap) Less(i, j int) bool { return h[i].mindsq < h[j].mindsq }

// Swap exchanges subtrees i and j, implementing sort.Interface as required by
// container/heap.
func (h kdBranchHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

// Push appends x, which must be a kdBranch, to the heap. It implements
// container/heap.Interface and is invoked through heap.Push rather than directly.
func (h *kdBranchHeap) Push(x any) { *h = append(*h, x.(kdBranch)) }

// Pop removes and returns the last subtree in the underlying slice. It implements
// container/heap.Interface and is invoked through heap.Pop rather than directly.
func (h *kdBranchHeap) Pop() any {
	old := *h
	n := len(old)
	it := old[n-1]
	*h = old[:n-1]
	return it
}

// KnnSearch returns the k nearest neighbours of query, searching every tree
// through one shared priority queue. With MaxChecks == 0 the result is exact.
func (f *KDForestIndex) KnnSearch(query []float64, k int) []Neighbor {
	if k <= 0 || len(f.trees) == 0 {
		return nil
	}
	res := &knnSet{k: k}
	seen := make([]bool, len(f.data))
	pq := &kdBranchHeap{}
	for _, root := range f.trees {
		*pq = append(*pq, kdBranch{node: root, mindsq: 0})
	}
	heap.Init(pq)

	checks := 0
	for pq.Len() > 0 {
		if f.MaxChecks > 0 && checks >= f.MaxChecks {
			break
		}
		br := heap.Pop(pq).(kdBranch)
		if res.full() {
			w := res.worst()
			// The queue is ordered by lower bound, so once the closest pending
			// branch cannot beat the current worst neighbour, none can.
			if br.mindsq > w*w {
				break
			}
		}
		f.descend(br.node, query, br.mindsq, res, seen, pq, &checks)
	}
	return res.n
}

// descend walks from node to a leaf along the near side of each split, pushing
// the far side onto the queue with a lower bound, then scans the leaf. lb is a
// lower bound on the distance from the query to the region entered here.
func (f *KDForestIndex) descend(node *kdRandNode, query []float64, lb float64, res *knnSet, seen []bool, pq *kdBranchHeap, checks *int) {
	for !node.leaf {
		if f.MaxChecks > 0 && *checks >= f.MaxChecks {
			return
		}
		diff := query[node.dim] - node.split
		near, far := node.left, node.right
		if diff >= 0 {
			near, far = node.right, node.left
		}
		farLB := lb
		if plane := diff * diff; plane > farLB {
			farLB = plane
		}
		heap.Push(pq, kdBranch{node: far, mindsq: farLB})
		node = near
	}
	for _, id := range node.indices {
		if seen[id] {
			continue
		}
		seen[id] = true
		res.add(Neighbor{Index: id, Distance: DistL2(query, f.data[id])})
		*checks++
	}
}

// RadiusSearch returns every point within radius of query, sorted ascending by
// distance. It ignores MaxChecks and is always exact.
func (f *KDForestIndex) RadiusSearch(query []float64, radius float64) []Neighbor {
	if len(f.trees) == 0 {
		return nil
	}
	seen := make([]bool, len(f.data))
	var out []Neighbor
	for _, root := range f.trees {
		f.radius(root, query, radius, seen, &out)
	}
	sortNeighbors(out)
	return out
}

func (f *KDForestIndex) radius(node *kdRandNode, query []float64, radius float64, seen []bool, out *[]Neighbor) {
	if node.leaf {
		for _, id := range node.indices {
			if seen[id] {
				continue
			}
			seen[id] = true
			d := DistL2(query, f.data[id])
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
	f.radius(near, query, radius, seen, out)
	if diff*diff <= radius*radius {
		f.radius(far, query, radius, seen, out)
	}
}
