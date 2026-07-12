package saliency

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// GMRSaliency implements the Graph-based Manifold Ranking salient object
// detector of Yang, Zhang, Lu, Ruan & Yang, "Saliency Detection via Graph-Based
// Manifold Ranking" (CVPR 2013).
//
// The image is partitioned into compact regions (super-pixels), each a graph
// node whose feature is its mean CIE L*a*b* colour and whose edges connect
// spatially adjacent regions and all image-border regions to one another. A
// closed-form manifold-ranking function propagates label information over this
// graph:
//
//	f* = (D − αW)⁻¹ y
//
// Stage one ranks every node against the four image borders in turn, treating
// each border as a set of background queries; the products of the four
// complementary (1 − rank) maps form a background-based saliency estimate.
// Stage two binarises that estimate to obtain foreground queries and re-runs
// the ranking, yielding the final map. Regions that rank far from the borders —
// a distinct central object — end up bright.
//
// Regular grid regions stand in for SLIC super-pixels and the linear system is
// solved by Gauss-Seidel iteration rather than explicit inversion, keeping the
// detector dependency-free and deterministic while preserving the two-stage
// ranking behaviour.
//
// Construct one with [NewGMRSaliency]. It satisfies [StaticSaliency].
type GMRSaliency struct {
	// Grid is the number of regions per side (Grid×Grid nodes). The default is
	// 10.
	Grid int
	// Alpha is the manifold-ranking balance parameter in (0,1). The default is
	// 0.99.
	Alpha float64
	// Sigma controls the colour affinity falloff (Lab normalised to [0,1]). The
	// default is 0.1.
	Sigma float64
	// Iterations is the number of Gauss-Seidel sweeps per solve. The default is
	// 60.
	Iterations int
}

// NewGMRSaliency returns a detector with a 10×10 region grid and the paper's
// default ranking parameters.
func NewGMRSaliency() *GMRSaliency {
	return &GMRSaliency{Grid: 10, Alpha: 0.99, Sigma: 0.1, Iterations: 60}
}

// gmrGraph holds the region graph: per-node mean colours, degree and adjacency
// affinity lists.
type gmrGraph struct {
	n         int
	degree    []float64
	neighbors [][]int
	weights   [][]float64
	isBorder  []bool
	side      [][]int // node indices on each of the 4 borders
}

// ComputeSaliency returns the manifold-ranking saliency map of img: a
// single-channel [cv.Mat] the same size as img, normalised to [0,255]. It
// panics if img is nil or empty.
func (s *GMRSaliency) ComputeSaliency(img *cv.Mat) *cv.Mat {
	l, a, b := labPlanes(img)
	rows, cols := l.rows, l.cols

	grid := s.Grid
	if grid < 2 {
		grid = 10
	}
	if grid > rows {
		grid = rows
	}
	if grid > cols {
		grid = cols
	}
	alpha := s.Alpha
	if !(alpha > 0 && alpha < 1) {
		alpha = 0.99
	}
	sigma := s.Sigma
	if sigma <= 0 {
		sigma = 0.1
	}
	iters := s.Iterations
	if iters < 1 {
		iters = 60
	}

	// Assign each pixel to a region and accumulate mean colours.
	label := make([]int, rows*cols)
	nNodes := grid * grid
	sumL := make([]float64, nNodes)
	sumA := make([]float64, nNodes)
	sumB := make([]float64, nNodes)
	cnt := make([]float64, nNodes)
	regionOf := func(y, x int) (int, int) {
		gy := y * grid / rows
		gx := x * grid / cols
		if gy >= grid {
			gy = grid - 1
		}
		if gx >= grid {
			gx = grid - 1
		}
		return gy, gx
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			gy, gx := regionOf(y, x)
			id := gy*grid + gx
			i := y*cols + x
			label[i] = id
			sumL[id] += l.data[i]
			sumA[id] += a.data[i]
			sumB[id] += b.data[i]
			cnt[id]++
		}
	}
	meanL := make([]float64, nNodes)
	meanA := make([]float64, nNodes)
	meanB := make([]float64, nNodes)
	for id := 0; id < nNodes; id++ {
		if cnt[id] > 0 {
			meanL[id] = sumL[id] / cnt[id] / 255
			meanA[id] = sumA[id] / cnt[id] / 255
			meanB[id] = sumB[id] / cnt[id] / 255
		}
	}

	g := buildGMRGraph(grid, meanL, meanA, meanB, sigma)

	// Stage 1: rank against each border, combine complementary maps.
	bg := make([]float64, nNodes)
	for i := range bg {
		bg[i] = 1
	}
	for side := 0; side < 4; side++ {
		y := make([]float64, nNodes)
		for _, node := range g.side[side] {
			y[node] = 1
		}
		f := manifoldRank(g, y, alpha, iters)
		normalizeInPlace(f)
		for i := range bg {
			bg[i] *= 1 - f[i]
		}
	}
	normalizeInPlace(bg)

	// Stage 2: foreground queries from thresholding the background estimate.
	mean := 0.0
	for _, v := range bg {
		mean += v
	}
	mean /= float64(nNodes)
	yq := make([]float64, nNodes)
	any := false
	for i, v := range bg {
		if v > mean {
			yq[i] = 1
			any = true
		}
	}
	final := bg
	if any {
		f := manifoldRank(g, yq, alpha, iters)
		normalizeInPlace(f)
		final = f
	}

	out := newPlane(rows, cols)
	for i := range out.data {
		out.data[i] = final[label[i]]
	}
	out = meanBlur(out, 2)
	return out.normalizedMat()
}

// buildGMRGraph constructs the region adjacency graph with colour affinities.
func buildGMRGraph(grid int, ml, ma, mb []float64, sigma float64) *gmrGraph {
	n := grid * grid
	g := &gmrGraph{
		n:         n,
		degree:    make([]float64, n),
		neighbors: make([][]int, n),
		weights:   make([][]float64, n),
		isBorder:  make([]bool, n),
		side:      make([][]int, 4),
	}
	twoSig2 := 2 * sigma * sigma
	affinity := func(i, j int) float64 {
		dl := ml[i] - ml[j]
		da := ma[i] - ma[j]
		db := mb[i] - mb[j]
		return math.Exp(-(dl*dl + da*da + db*db) / twoSig2)
	}
	addEdge := func(i, j int) {
		w := affinity(i, j)
		g.neighbors[i] = append(g.neighbors[i], j)
		g.weights[i] = append(g.weights[i], w)
		g.degree[i] += w
	}
	seen := make(map[[2]int]bool)
	connect := func(i, j int) {
		if i == j {
			return
		}
		key := [2]int{i, j}
		if i > j {
			key = [2]int{j, i}
		}
		if seen[key] {
			return
		}
		seen[key] = true
		addEdge(i, j)
		addEdge(j, i)
	}
	for gy := 0; gy < grid; gy++ {
		for gx := 0; gx < grid; gx++ {
			id := gy*grid + gx
			// 8-connected spatial neighbours.
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					ny, nx := gy+dy, gx+dx
					if ny < 0 || ny >= grid || nx < 0 || nx >= grid {
						continue
					}
					connect(id, ny*grid+nx)
				}
			}
			if gy == 0 {
				g.isBorder[id] = true
				g.side[0] = append(g.side[0], id)
			}
			if gy == grid-1 {
				g.isBorder[id] = true
				g.side[1] = append(g.side[1], id)
			}
			if gx == 0 {
				g.isBorder[id] = true
				g.side[2] = append(g.side[2], id)
			}
			if gx == grid-1 {
				g.isBorder[id] = true
				g.side[3] = append(g.side[3], id)
			}
		}
	}
	// Closed-loop constraint: connect all border regions to one another.
	var border []int
	for i := 0; i < n; i++ {
		if g.isBorder[i] {
			border = append(border, i)
		}
	}
	for i := 0; i < len(border); i++ {
		for j := i + 1; j < len(border); j++ {
			connect(border[i], border[j])
		}
	}
	return g
}

// manifoldRank solves f = (D - alpha*W)^{-1} y by Gauss-Seidel iteration:
// f_i <- (y_i + alpha * sum_j w_ij f_j) / D_ii.
func manifoldRank(g *gmrGraph, y []float64, alpha float64, iters int) []float64 {
	f := make([]float64, g.n)
	copy(f, y)
	for it := 0; it < iters; it++ {
		for i := 0; i < g.n; i++ {
			d := g.degree[i]
			if d <= 0 {
				f[i] = y[i]
				continue
			}
			var acc float64
			nbr := g.neighbors[i]
			w := g.weights[i]
			for k, j := range nbr {
				acc += w[k] * f[j]
			}
			f[i] = (y[i] + alpha*acc) / d
		}
	}
	return f
}

// normalizeInPlace min-max normalises v to [0,1]; a flat vector becomes zeros.
func normalizeInPlace(v []float64) {
	mn, mx := math.Inf(1), math.Inf(-1)
	for _, x := range v {
		if x < mn {
			mn = x
		}
		if x > mx {
			mx = x
		}
	}
	if !(mx > mn) {
		for i := range v {
			v[i] = 0
		}
		return
	}
	inv := 1 / (mx - mn)
	for i := range v {
		v[i] = (v[i] - mn) * inv
	}
}
