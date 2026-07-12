package imgprocx

import "math"

// mcmfEdge is a directed residual-graph edge for the min-cost max-flow solver
// behind [EMD]: to is the head node, cap the remaining capacity and cost the
// per-unit cost. Each edge is paired with its reverse at the adjacent index.
type mcmfEdge struct {
	to   int
	cap  float64
	cost float64
}

// EMD returns the Earth Mover's Distance between two weighted signatures,
// mirroring cv2.EMD. supply and demand hold the non-negative weights of the two
// distributions and cost[i][j] is the ground distance between supply cluster i
// and demand cluster j. It computes the minimum-cost way to transform one
// distribution into the other and returns that cost divided by the total flow
// moved — the same normalisation OpenCV uses.
//
// The transportation problem is solved exactly as a min-cost max-flow: a source
// feeds each supply node (capacity = its weight), every supply node connects to
// every demand node (cost = the ground distance), and each demand node drains to
// a sink (capacity = its weight). When the total supply and demand differ only
// the smaller total is moved, matching OpenCV's handling of unequal masses. It
// panics if cost is not a len(supply)×len(demand) matrix or any weight is
// negative.
func EMD(supply, demand []float64, cost [][]float64) float64 {
	m := len(supply)
	n := len(demand)
	if len(cost) != m {
		panic("imgprocx: EMD cost must have one row per supply cluster")
	}
	for i := range cost {
		if len(cost[i]) != n {
			panic("imgprocx: EMD cost must have one column per demand cluster")
		}
	}
	var totalS, totalD float64
	for _, s := range supply {
		if s < 0 {
			panic("imgprocx: EMD requires non-negative supply weights")
		}
		totalS += s
	}
	for _, d := range demand {
		if d < 0 {
			panic("imgprocx: EMD requires non-negative demand weights")
		}
		totalD += d
	}
	if totalS == 0 || totalD == 0 {
		return 0
	}

	// Node layout: 0 = source, 1..m = supply, m+1..m+n = demand, m+n+1 = sink.
	source := 0
	sink := m + n + 1
	numNodes := m + n + 2
	graph := make([][]int, numNodes) // adjacency: node -> edge indices
	var edges []mcmfEdge

	addEdge := func(from, to int, capacity, c float64) {
		graph[from] = append(graph[from], len(edges))
		edges = append(edges, mcmfEdge{to: to, cap: capacity, cost: c})
		graph[to] = append(graph[to], len(edges))
		edges = append(edges, mcmfEdge{to: from, cap: 0, cost: -c})
	}

	inf := totalS + totalD
	for i := 0; i < m; i++ {
		addEdge(source, 1+i, supply[i], 0)
	}
	for j := 0; j < n; j++ {
		addEdge(1+m+j, sink, demand[j], 0)
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			addEdge(1+i, 1+m+j, inf, cost[i][j])
		}
	}

	var totalCost, totalFlow float64
	for {
		// Bellman-Ford (SPFA) shortest path by cost from source to sink.
		dist := make([]float64, numNodes)
		inQueue := make([]bool, numNodes)
		prevEdge := make([]int, numNodes)
		for i := range dist {
			dist[i] = math.Inf(1)
			prevEdge[i] = -1
		}
		dist[source] = 0
		queue := []int{source}
		inQueue[source] = true
		for len(queue) > 0 {
			u := queue[0]
			queue = queue[1:]
			inQueue[u] = false
			for _, ei := range graph[u] {
				e := edges[ei]
				if e.cap > 1e-15 && dist[u]+e.cost < dist[e.to]-1e-15 {
					dist[e.to] = dist[u] + e.cost
					prevEdge[e.to] = ei
					if !inQueue[e.to] {
						queue = append(queue, e.to)
						inQueue[e.to] = true
					}
				}
			}
		}
		if math.IsInf(dist[sink], 1) {
			break // no augmenting path remains
		}
		// Bottleneck along the discovered path.
		bottleneck := math.Inf(1)
		for v := sink; v != source; {
			ei := prevEdge[v]
			if edges[ei].cap < bottleneck {
				bottleneck = edges[ei].cap
			}
			v = edges[ei^1].to
		}
		if bottleneck <= 1e-15 {
			break
		}
		for v := sink; v != source; {
			ei := prevEdge[v]
			edges[ei].cap -= bottleneck
			edges[ei^1].cap += bottleneck
			v = edges[ei^1].to
		}
		totalFlow += bottleneck
		totalCost += bottleneck * dist[sink]
	}

	if totalFlow <= 0 {
		return 0
	}
	return totalCost / totalFlow
}
