package connected

// connectedUnionFind is a disjoint-set forest over provisional component
// labels, using path halving and union by smaller root. Index 0 is reserved so
// that a zero provisional label always means "background / unassigned".
type connectedUnionFind struct {
	parent []int
}

// connectedNewUnionFind returns an empty forest with the background slot
// reserved.
func connectedNewUnionFind() *connectedUnionFind {
	return &connectedUnionFind{parent: []int{0}}
}

// makeSet creates a new singleton label and returns its identifier.
func (u *connectedUnionFind) makeSet() int {
	id := len(u.parent)
	u.parent = append(u.parent, id)
	return id
}

// find returns the representative (root) of x, compressing the path with
// halving as it climbs.
func (u *connectedUnionFind) find(x int) int {
	for u.parent[x] != x {
		u.parent[x] = u.parent[u.parent[x]]
		x = u.parent[x]
	}
	return x
}

// union merges the sets containing a and b, keeping the numerically smaller
// root as the representative so labelling stays deterministic.
func (u *connectedUnionFind) union(a, b int) {
	ra, rb := u.find(a), u.find(b)
	if ra == rb {
		return
	}
	if ra < rb {
		u.parent[rb] = ra
	} else {
		u.parent[ra] = rb
	}
}
