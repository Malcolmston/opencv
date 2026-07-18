package segment2

// FlowGraph is a directed graph with real-valued edge capacities used for
// max-flow / min-cut computations. Vertices are integers in [0, N). Construct
// one with [NewFlowGraph] and add capacities with [FlowGraph.AddEdge].
type FlowGraph struct {
	n      int
	source int
	sink   int
	// head[v] indexes the first edge of vertex v in the edge arrays (-1 if none).
	head []int
	to   []int
	next []int
	cap  []float64
}

// NewFlowGraph creates an empty flow graph with n vertices and the given source
// and sink. It panics if n <= 0 or the terminals are out of range.
func NewFlowGraph(n, source, sink int) *FlowGraph {
	if n <= 0 || source < 0 || source >= n || sink < 0 || sink >= n {
		panic("segment2: NewFlowGraph invalid arguments")
	}
	g := &FlowGraph{n: n, source: source, sink: sink, head: make([]int, n)}
	for i := range g.head {
		g.head[i] = -1
	}
	return g
}

// AddEdge adds a directed edge from u to v with the given forward capacity and a
// reverse edge of capacity revCap (use 0 for a purely directed edge, or an equal
// value for an undirected edge). Capacities must be non-negative. It panics on
// out-of-range vertices or negative capacity.
func (g *FlowGraph) AddEdge(u, v int, capacity, revCap float64) {
	if u < 0 || u >= g.n || v < 0 || v >= g.n {
		panic("segment2: AddEdge vertex out of range")
	}
	if capacity < 0 || revCap < 0 {
		panic("segment2: AddEdge negative capacity")
	}
	g.to = append(g.to, v)
	g.cap = append(g.cap, capacity)
	g.next = append(g.next, g.head[u])
	g.head[u] = len(g.to) - 1

	g.to = append(g.to, u)
	g.cap = append(g.cap, revCap)
	g.next = append(g.next, g.head[v])
	g.head[v] = len(g.to) - 1
}

// MaxFlow computes the maximum flow from source to sink with Dinic's algorithm
// and returns the flow value together with the min-cut: a boolean slice of
// length N in which true marks the vertices on the source side of a minimum cut
// (those still reachable from the source in the residual graph). The graph's
// residual capacities are consumed by the call.
func (g *FlowGraph) MaxFlow() (float64, []bool) {
	const eps = 1e-12
	level := make([]int, g.n)
	iter := make([]int, g.n)
	queue := make([]int, 0, g.n)

	bfs := func() bool {
		for i := range level {
			level[i] = -1
		}
		queue = queue[:0]
		level[g.source] = 0
		queue = append(queue, g.source)
		for qi := 0; qi < len(queue); qi++ {
			u := queue[qi]
			for e := g.head[u]; e != -1; e = g.next[e] {
				if g.cap[e] > eps && level[g.to[e]] < 0 {
					level[g.to[e]] = level[u] + 1
					queue = append(queue, g.to[e])
				}
			}
		}
		return level[g.sink] >= 0
	}

	var dfs func(u int, f float64) float64
	dfs = func(u int, f float64) float64 {
		if u == g.sink {
			return f
		}
		for ; iter[u] != -1; iter[u] = g.next[iter[u]] {
			e := iter[u]
			v := g.to[e]
			if g.cap[e] > eps && level[v] == level[u]+1 {
				d := f
				if g.cap[e] < d {
					d = g.cap[e]
				}
				pushed := dfs(v, d)
				if pushed > eps {
					g.cap[e] -= pushed
					g.cap[e^1] += pushed
					return pushed
				}
			}
		}
		return 0
	}

	var flow float64
	const inf = 1e18
	for bfs() {
		copy(iter, g.head)
		for {
			f := dfs(g.source, inf)
			if f <= eps {
				break
			}
			flow += f
		}
	}

	// Source-side of the min cut: vertices reachable from source in residual.
	reach := make([]bool, g.n)
	queue = queue[:0]
	reach[g.source] = true
	queue = append(queue, g.source)
	for qi := 0; qi < len(queue); qi++ {
		u := queue[qi]
		for e := g.head[u]; e != -1; e = g.next[e] {
			if g.cap[e] > eps && !reach[g.to[e]] {
				reach[g.to[e]] = true
				queue = append(queue, g.to[e])
			}
		}
	}
	return flow, reach
}
