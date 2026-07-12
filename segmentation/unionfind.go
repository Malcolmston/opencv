package segmentation

// unionFind is a disjoint-set forest with union-by-size and path compression,
// used by the graph-based segmenters to merge pixels or regions into components.
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
// the result is deterministic.
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

// setSize returns the number of elements in the set containing x.
func (u *unionFind) setSize(x int) int {
	return u.size[u.find(x)]
}
