package hfs

import "sort"

// unionFind is a disjoint-set forest with union-by-size and path compression. It
// backs both the superpixel connectivity pass and the graph-based region merges.
type unionFind struct {
	parent []int
	size   []int
}

// newUnionFind returns a forest of n singleton sets labelled 0..n-1.
func newUnionFind(n int) *unionFind {
	u := &unionFind{parent: make([]int, n), size: make([]int, n)}
	for i := 0; i < n; i++ {
		u.parent[i] = i
		u.size[i] = 1
	}
	return u
}

// find returns the representative of x's set, compressing the path on the way.
func (u *unionFind) find(x int) int {
	for u.parent[x] != x {
		u.parent[x] = u.parent[u.parent[x]]
		x = u.parent[x]
	}
	return x
}

// union merges the sets containing a and b and returns the new representative.
// The larger set absorbs the smaller; ties keep the lower index as the root so
// the result is independent of argument order.
func (u *unionFind) union(a, b int) int {
	ra, rb := u.find(a), u.find(b)
	if ra == rb {
		return ra
	}
	if u.size[ra] < u.size[rb] || (u.size[ra] == u.size[rb] && rb < ra) {
		ra, rb = rb, ra
	}
	u.parent[rb] = ra
	u.size[ra] += u.size[rb]
	return ra
}

// edge is a weighted connection between two graph nodes (superpixels or regions).
type edge struct {
	a, b int
	w    float64
}

// relabelConsecutive maps an arbitrary per-element labelling (for example
// union-find roots) onto the dense range 0..count-1, preserving first-appearance
// order so the result is deterministic. It returns the new labels and the count.
func relabelConsecutive(raw []int) (labels []int, count int) {
	remap := make(map[int]int, len(raw))
	labels = make([]int, len(raw))
	for i, r := range raw {
		id, ok := remap[r]
		if !ok {
			id = count
			remap[r] = id
			count++
		}
		labels[i] = id
	}
	return labels, count
}

// egbMerge runs the Felzenszwalb & Huttenlocher minimum-spanning-forest merge
// over a graph of nodeCount nodes and the given weighted edges. Two components
// are joined when the connecting edge weight does not exceed the minimum internal
// variation of either endpoint, where a component's internal variation is its
// largest interior edge plus threshold/size. A larger threshold yields larger
// regions. Edges are processed in nondecreasing weight order (stable, so equal
// weights keep their input order) and the returned forest is deterministic.
func egbMerge(nodeCount int, edges []edge, threshold float64) *unionFind {
	sorted := make([]edge, len(edges))
	copy(sorted, edges)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].w < sorted[j].w })

	uf := newUnionFind(nodeCount)
	internal := make([]float64, nodeCount) // largest interior edge per root
	for _, e := range sorted {
		ra, rb := uf.find(e.a), uf.find(e.b)
		if ra == rb {
			continue
		}
		limitA := internal[ra] + threshold/float64(uf.size[ra])
		limitB := internal[rb] + threshold/float64(uf.size[rb])
		if e.w <= limitA && e.w <= limitB {
			root := uf.union(ra, rb)
			if e.w > internal[root] {
				internal[root] = e.w
			}
		}
	}
	return uf
}

// absorbSmall merges every component smaller than minSize (measured in the
// weights carried by nodeWeight, i.e. superpixel pixel counts) into the adjacent
// component joined by the lightest edge, repeating until no undersized component
// has a lighter alternative. It mutates uf in place. Passing minSize <= 1 is a
// no-op.
func absorbSmall(uf *unionFind, edges []edge, nodeWeight []int, minSize int) {
	if minSize <= 1 {
		return
	}
	// Accumulate the true pixel weight of each root so setSize (node count) is
	// not confused with pixel area.
	weight := make([]int, len(uf.parent))
	for i, w := range nodeWeight {
		weight[uf.find(i)] += w
	}
	sorted := make([]edge, len(edges))
	copy(sorted, edges)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].w < sorted[j].w })

	// Repeatedly sweep the edges lightest-first, merging any edge that touches an
	// undersized component, until a full sweep changes nothing.
	for {
		changed := false
		for _, e := range sorted {
			ra, rb := uf.find(e.a), uf.find(e.b)
			if ra == rb {
				continue
			}
			if weight[ra] < minSize || weight[rb] < minSize {
				root := uf.union(ra, rb)
				other := ra
				if root == ra {
					other = rb
				}
				weight[root] += weight[other]
				changed = true
			}
		}
		if !changed {
			break
		}
	}
}
