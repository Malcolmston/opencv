package cudalegacy

import (
	cv "github.com/malcolmston/opencv"
)

// maxFlow is a Dinic max-flow solver over a directed graph with float64 edge
// capacities. Nodes are integers 0..n-1. Each undirected or directed edge is
// stored as a forward/backward pair in parallel slices.
type maxFlow struct {
	n    int
	head []int
	to   []int
	cap  []float64
	next []int
}

func newMaxFlow(n int) *maxFlow {
	head := make([]int, n)
	for i := range head {
		head[i] = -1
	}
	return &maxFlow{n: n, head: head}
}

// addEdge adds a directed edge u->v of capacity cap and a reverse edge v->u of
// capacity rcap (rcap = cap makes the pair undirected).
func (g *maxFlow) addEdge(u, v int, capUV, capVU float64) {
	g.to = append(g.to, v)
	g.cap = append(g.cap, capUV)
	g.next = append(g.next, g.head[u])
	g.head[u] = len(g.to) - 1

	g.to = append(g.to, u)
	g.cap = append(g.cap, capVU)
	g.next = append(g.next, g.head[v])
	g.head[v] = len(g.to) - 1
}

func (g *maxFlow) bfs(s, t int, level []int) bool {
	for i := range level {
		level[i] = -1
	}
	level[s] = 0
	queue := []int{s}
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		for e := g.head[u]; e != -1; e = g.next[e] {
			if g.cap[e] > 1e-12 && level[g.to[e]] < 0 {
				level[g.to[e]] = level[u] + 1
				queue = append(queue, g.to[e])
			}
		}
	}
	return level[t] >= 0
}

func (g *maxFlow) dfs(u, t int, f float64, level, it []int) float64 {
	if u == t {
		return f
	}
	for ; it[u] != -1; it[u] = g.next[it[u]] {
		e := it[u]
		v := g.to[e]
		if g.cap[e] > 1e-12 && level[v] == level[u]+1 {
			d := f
			if g.cap[e] < d {
				d = g.cap[e]
			}
			pushed := g.dfs(v, t, d, level, it)
			if pushed > 0 {
				g.cap[e] -= pushed
				g.cap[e^1] += pushed
				return pushed
			}
		}
	}
	return 0
}

// run computes the maximum flow from s to t and returns the boolean set of nodes
// still reachable from s in the residual graph (the source side of the minimum
// cut).
func (g *maxFlow) run(s, t int) []bool {
	level := make([]int, g.n)
	it := make([]int, g.n)
	for g.bfs(s, t, level) {
		copy(it, g.head)
		for {
			f := g.dfs(s, t, inf, level, it)
			if f <= 1e-12 {
				break
			}
		}
	}
	// Source side = nodes reachable from s in the residual graph.
	reach := make([]bool, g.n)
	reach[s] = true
	queue := []int{s}
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		for e := g.head[u]; e != -1; e = g.next[e] {
			if g.cap[e] > 1e-12 && !reach[g.to[e]] {
				reach[g.to[e]] = true
				queue = append(queue, g.to[e])
			}
		}
	}
	return reach
}

// GraphCut is a CPU-backed mirror of the binary segmentation performed by
// OpenCV's cv::cuda::graphcut. It solves the classic two-terminal min-cut /
// max-flow labelling on the pixel grid: every pixel is a node connected to a
// virtual source with capacity sourceCap and to a virtual sink with capacity
// sinkCap, and to each 4-connected neighbour with capacity lambda (the pairwise
// smoothness that penalises label changes between adjacent pixels).
//
// The returned single-channel [GpuMat] labels each pixel [SourceLabel] (255,
// the source side of the cut) or [SinkLabel] (0, the sink side). sourceCap and
// sinkCap must be identically-sized [cv.FloatMat] planes; negative capacities
// are clamped to 0. lambda is clamped to a non-negative value. It panics on nil
// or mismatched capacity planes. The stream is a no-op.
func GraphCut(sourceCap, sinkCap *cv.FloatMat, lambda float64, stream *Stream) *GpuMat {
	_ = stream
	if sourceCap == nil || sinkCap == nil {
		panic("cudalegacy: GraphCut given nil capacity planes")
	}
	if sourceCap.Rows != sinkCap.Rows || sourceCap.Cols != sinkCap.Cols {
		panic("cudalegacy: GraphCut source and sink capacity sizes differ")
	}
	if lambda < 0 {
		lambda = 0
	}
	rows, cols := sourceCap.Rows, sourceCap.Cols
	n := rows * cols
	s := n
	t := n + 1
	g := newMaxFlow(n + 2)

	clampNN := func(v float64) float64 {
		if v < 0 {
			return 0
		}
		return v
	}
	for i := 0; i < n; i++ {
		g.addEdge(s, i, clampNN(sourceCap.Data[i]), 0)
		g.addEdge(i, t, clampNN(sinkCap.Data[i]), 0)
	}
	if lambda > 0 {
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				idx := y*cols + x
				if x+1 < cols {
					g.addEdge(idx, idx+1, lambda, lambda)
				}
				if y+1 < rows {
					g.addEdge(idx, idx+cols, lambda, lambda)
				}
			}
		}
	}

	reach := g.run(s, t)
	out := cv.NewMat(rows, cols, 1)
	for i := 0; i < n; i++ {
		if reach[i] {
			out.Data[i] = SourceLabel
		} else {
			out.Data[i] = SinkLabel
		}
	}
	return GpuMatFromMat(out)
}

// Labels assigned by [GraphCut] to the two sides of the minimum cut.
const (
	// SourceLabel marks pixels on the source side of the cut (foreground).
	SourceLabel uint8 = 255
	// SinkLabel marks pixels on the sink side of the cut (background).
	SinkLabel uint8 = 0
)
