package stitching

import (
	"image"
	"math"

	cv "github.com/malcolmston/opencv"
)

// SeamFinder decides, in the regions where two panorama images overlap, which
// image should supply each pixel. It rewrites the coverage masks so that the
// overlap is partitioned along a seam that is as invisible as possible — one that
// runs where the two images already agree — instead of cross-fading everywhere.
// A good seam hides moving objects and parallax that blending alone would ghost.
//
// Find is given the warped images, their canvas corners and their masks (all
// index-aligned); it zeroes mask pixels on the losing side of every seam so the
// masks become disjoint over each overlap. Implementations are [NoSeamFinder],
// [VoronoiSeamFinder], [DpSeamFinder] and [GraphCutSeamFinder], selected for a
// [Pipeline] with [Pipeline.SetSeamFinder].
type SeamFinder interface {
	// Find partitions every pairwise overlap by editing masks in place.
	Find(images []*cv.Mat, corners []image.Point, masks []*cv.FloatMat)
}

// overlapRect returns the canvas-space intersection rectangle [x0,x1)×[y0,y1) of
// image a (at ca) and image b (at cb), and whether it is non-empty.
func overlapRect(a, b *cv.Mat, ca, cb image.Point) (x0, y0, x1, y1 int, ok bool) {
	x0 = maxInt(ca.X, cb.X)
	y0 = maxInt(ca.Y, cb.Y)
	x1 = minInt(ca.X+a.Cols, cb.X+b.Cols)
	y1 = minInt(ca.Y+a.Rows, cb.Y+b.Rows)
	return x0, y0, x1, y1, x1 > x0 && y1 > y0
}

// assignPair applies a per-pixel labelling over the valid overlap of images i and
// j. For each overlapping pixel valid in both masks, keepA reports whether image
// i keeps it; the other image's mask is zeroed there so the masks stay disjoint.
func assignPair(i, j int, images []*cv.Mat, corners []image.Point, masks []*cv.FloatMat, keepA func(lx, ly int) bool) {
	a, b := images[i], images[j]
	ca, cb := corners[i], corners[j]
	x0, y0, x1, y1, ok := overlapRect(a, b, ca, cb)
	if !ok {
		return
	}
	for gy := y0; gy < y1; gy++ {
		for gx := x0; gx < x1; gx++ {
			pa := (gy-ca.Y)*a.Cols + (gx - ca.X)
			pb := (gy-cb.Y)*b.Cols + (gx - cb.X)
			if masks[i].Data[pa] <= 0 || masks[j].Data[pb] <= 0 {
				continue
			}
			if keepA(gx-x0, gy-y0) {
				masks[j].Data[pb] = 0
			} else {
				masks[i].Data[pa] = 0
			}
		}
	}
}

// overlapDiff builds the per-pixel absolute intensity difference over the overlap
// rectangle of images i and j. Pixels not valid in both masks are marked with a
// negative sentinel in valid.
func overlapDiff(i, j int, images []*cv.Mat, corners []image.Point, masks []*cv.FloatMat, x0, y0, w, h int) (diff []float64, valid []bool) {
	a, b := images[i], images[j]
	ca, cb := corners[i], corners[j]
	diff = make([]float64, w*h)
	valid = make([]bool, w*h)
	for ly := 0; ly < h; ly++ {
		for lx := 0; lx < w; lx++ {
			gx := x0 + lx
			gy := y0 + ly
			pa := (gy-ca.Y)*a.Cols + (gx - ca.X)
			pb := (gy-cb.Y)*b.Cols + (gx - cb.X)
			k := ly*w + lx
			if masks[i].Data[pa] <= 0 || masks[j].Data[pb] <= 0 {
				valid[k] = false
				diff[k] = 0
				continue
			}
			valid[k] = true
			diff[k] = math.Abs(pixelIntensity(a, pa) - pixelIntensity(b, pb))
		}
	}
	return diff, valid
}

// NoSeamFinder leaves the masks untouched, so overlaps are resolved entirely by
// the blender. Use it to disable seam finding.
type NoSeamFinder struct{}

// Find does nothing.
func (NoSeamFinder) Find([]*cv.Mat, []image.Point, []*cv.FloatMat) {}

// VoronoiSeamFinder assigns each overlap pixel to the image whose centre is
// nearest, producing a straight Voronoi boundary. It ignores image content, so
// it is the fastest option but does not hide moving objects or misalignment.
type VoronoiSeamFinder struct{}

// Find partitions overlaps by nearest image centre.
func (VoronoiSeamFinder) Find(images []*cv.Mat, corners []image.Point, masks []*cv.FloatMat) {
	n := len(images)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			acx := float64(corners[i].X) + float64(images[i].Cols)/2
			acy := float64(corners[i].Y) + float64(images[i].Rows)/2
			bcx := float64(corners[j].X) + float64(images[j].Cols)/2
			bcy := float64(corners[j].Y) + float64(images[j].Rows)/2
			x0, y0 := maxInt(corners[i].X, corners[j].X), maxInt(corners[i].Y, corners[j].Y)
			assignPair(i, j, images, corners, masks, func(lx, ly int) bool {
				gx := float64(x0 + lx)
				gy := float64(y0 + ly)
				da := (gx-acx)*(gx-acx) + (gy-acy)*(gy-acy)
				db := (gx-bcx)*(gx-bcx) + (gy-bcy)*(gy-bcy)
				return da <= db
			})
		}
	}
}

// DpSeamFinder places a minimum-cost seam through each overlap with dynamic
// programming: the seam follows the path of least accumulated colour difference
// across the overlap, so it threads through regions where the two images already
// match. It is much cheaper than [GraphCutSeamFinder] and usually nearly as good
// for simple two-image overlaps.
type DpSeamFinder struct {
	lastCost float64
}

// SeamCost returns the accumulated colour-difference cost of the seam found by
// the most recent Find call; a low value means the images agree along the seam.
func (dp *DpSeamFinder) SeamCost() float64 { return dp.lastCost }

// Find routes a minimum-cost seam through every overlap.
func (dp *DpSeamFinder) Find(images []*cv.Mat, corners []image.Point, masks []*cv.FloatMat) {
	dp.lastCost = 0
	n := len(images)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			x0, y0, x1, y1, ok := overlapRect(images[i], images[j], corners[i], corners[j])
			if !ok {
				continue
			}
			w, h := x1-x0, y1-y0
			diff, valid := overlapDiff(i, j, images, corners, masks, x0, y0, w, h)
			// Vertical seam (one column per row) when the overlap is tall, else a
			// horizontal seam (one row per column).
			vertical := h >= w
			// Orient which image is the "low" side (left or top).
			aLow := true
			if vertical {
				aLow = corners[i].X <= corners[j].X
			} else {
				aLow = corners[i].Y <= corners[j].Y
			}
			seam, cost := minSeam(diff, valid, w, h, vertical)
			dp.lastCost += cost
			assignPair(i, j, images, corners, masks, func(lx, ly int) bool {
				var onLowSide bool
				if vertical {
					onLowSide = lx <= seam[ly]
				} else {
					onLowSide = ly <= seam[lx]
				}
				// The low-side image keeps the low side of the seam.
				if aLow {
					return onLowSide
				}
				return !onLowSide
			})
		}
	}
}

// minSeam finds a minimum-cost monotone seam across a w×h cost field. When
// vertical is true it returns one column index per row (length h); otherwise one
// row index per column (length w). Invalid cells are given a large cost so the
// seam avoids them. It also returns the seam's total cost.
func minSeam(cost []float64, valid []bool, w, h int, vertical bool) ([]int, float64) {
	const big = 1e12
	at := func(x, y int) float64 {
		if !valid[y*w+x] {
			return big
		}
		return cost[y*w+x]
	}
	if vertical {
		// Accumulate top-to-bottom; each row picks a column.
		m := make([]float64, w*h)
		for x := 0; x < w; x++ {
			m[x] = at(x, 0)
		}
		for y := 1; y < h; y++ {
			for x := 0; x < w; x++ {
				best := m[(y-1)*w+x]
				if x > 0 && m[(y-1)*w+x-1] < best {
					best = m[(y-1)*w+x-1]
				}
				if x < w-1 && m[(y-1)*w+x+1] < best {
					best = m[(y-1)*w+x+1]
				}
				m[y*w+x] = at(x, y) + best
			}
		}
		// Backtrack from the minimum in the last row.
		endX := 0
		for x := 1; x < w; x++ {
			if m[(h-1)*w+x] < m[(h-1)*w+endX] {
				endX = x
			}
		}
		total := m[(h-1)*w+endX]
		seam := make([]int, h)
		x := endX
		for y := h - 1; y >= 0; y-- {
			seam[y] = x
			if y == 0 {
				break
			}
			best := x
			bv := m[(y-1)*w+x]
			if x > 0 && m[(y-1)*w+x-1] < bv {
				bv = m[(y-1)*w+x-1]
				best = x - 1
			}
			if x < w-1 && m[(y-1)*w+x+1] < bv {
				best = x + 1
			}
			x = best
		}
		return seam, total
	}
	// Horizontal seam: accumulate left-to-right; each column picks a row.
	m := make([]float64, w*h)
	for y := 0; y < h; y++ {
		m[y*w] = at(0, y)
	}
	for x := 1; x < w; x++ {
		for y := 0; y < h; y++ {
			best := m[y*w+x-1]
			if y > 0 && m[(y-1)*w+x-1] < best {
				best = m[(y-1)*w+x-1]
			}
			if y < h-1 && m[(y+1)*w+x-1] < best {
				best = m[(y+1)*w+x-1]
			}
			m[y*w+x] = at(x, y) + best
		}
	}
	endY := 0
	for y := 1; y < h; y++ {
		if m[y*w+w-1] < m[endY*w+w-1] {
			endY = y
		}
	}
	total := m[endY*w+w-1]
	seam := make([]int, w)
	y := endY
	for x := w - 1; x >= 0; x-- {
		seam[x] = y
		if x == 0 {
			break
		}
		best := y
		bv := m[y*w+x-1]
		if y > 0 && m[(y-1)*w+x-1] < bv {
			bv = m[(y-1)*w+x-1]
			best = y - 1
		}
		if y < h-1 && m[(y+1)*w+x-1] < bv {
			best = y + 1
		}
		y = best
	}
	return seam, total
}

// GraphCutSeamFinder finds the globally optimal seam through each overlap as the
// minimum cut of a graph whose edge weights are the colour difference between the
// two images. Unlike the dynamic-programming seam, the cut is not restricted to a
// monotone path, so it can route around 2-D obstacles such as a person who moved
// between shots. The cut is computed with a Dinic max-flow.
type GraphCutSeamFinder struct {
	lastCost float64
}

// CutCost returns the total edge weight severed by the most recent Find; a low
// value means the seam runs where the images agree.
func (gc *GraphCutSeamFinder) CutCost() float64 { return gc.lastCost }

// Find computes a minimum-cut seam through every overlap.
func (gc *GraphCutSeamFinder) Find(images []*cv.Mat, corners []image.Point, masks []*cv.FloatMat) {
	gc.lastCost = 0
	n := len(images)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			x0, y0, x1, y1, ok := overlapRect(images[i], images[j], corners[i], corners[j])
			if !ok {
				continue
			}
			w, h := x1-x0, y1-y0
			diff, valid := overlapDiff(i, j, images, corners, masks, x0, y0, w, h)
			vertical := h >= w
			aLow := true
			if vertical {
				aLow = corners[i].X <= corners[j].X
			} else {
				aLow = corners[i].Y <= corners[j].Y
			}
			label, cost := graphCutLabels(diff, valid, w, h, vertical)
			gc.lastCost += cost
			assignPair(i, j, images, corners, masks, func(lx, ly int) bool {
				lowSide := label[ly*w+lx] // true => source (low) side
				if aLow {
					return lowSide
				}
				return !lowSide
			})
		}
	}
}

// graphCutLabels labels each valid overlap cell as source-side (true) or
// sink-side (false) by a minimum cut. Terminal seeds are placed on the low and
// high edges of the overlap (left/right for a vertical seam, top/bottom for a
// horizontal one). Neighbour edges carry the sum of the two cells' colour
// differences, so the cut cost is the seam's visibility. Returns the labels and
// the cut cost.
func graphCutLabels(diff []float64, valid []bool, w, h int, vertical bool) ([]bool, float64) {
	const inf = 1e15
	nNodes := w*h + 2
	src := w * h
	sink := w*h + 1
	mf := newMaxflow(nNodes)
	idx := func(x, y int) int { return y*w + x }

	// Neighbour (smoothness) edges between adjacent valid cells.
	addEdge := func(x0, y0, x1, y1 int) {
		if !valid[idx(x0, y0)] || !valid[idx(x1, y1)] {
			return
		}
		cap := diff[idx(x0, y0)] + diff[idx(x1, y1)] + 1e-3
		mf.addEdge(idx(x0, y0), idx(x1, y1), cap, cap)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if !valid[idx(x, y)] {
				continue
			}
			if x+1 < w {
				addEdge(x, y, x+1, y)
			}
			if y+1 < h {
				addEdge(x, y, x, y+1)
			}
		}
	}

	// Terminal seeds on the two opposing edges of the overlap.
	seedSourceSink := func(x, y int) (isSource, isSink bool) {
		if vertical {
			return x == 0, x == w-1
		}
		return y == 0, y == h-1
	}
	seeded := false
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if !valid[idx(x, y)] {
				continue
			}
			s, t := seedSourceSink(x, y)
			if s {
				mf.addEdge(src, idx(x, y), inf, 0)
				seeded = true
			}
			if t {
				mf.addEdge(idx(x, y), sink, inf, 0)
				seeded = true
			}
		}
	}

	label := make([]bool, w*h)
	if !seeded {
		// No terminal contact (degenerate overlap): assign everything to source.
		for k := range label {
			label[k] = true
		}
		return label, 0
	}
	cost := mf.maxflow(src, sink)
	reach := mf.minCutReachable(src)
	for k := 0; k < w*h; k++ {
		if valid[k] {
			label[k] = reach[k]
		} else {
			label[k] = true
		}
	}
	return label, cost
}

// maxflow is a Dinic max-flow solver over a fixed node set. It is used by
// [GraphCutSeamFinder] to compute the minimum-cut seam.
type maxflow struct {
	to    []float64Edge
	level []int
	iter  []int
	adj   [][]int
}

// float64Edge is one directed residual edge.
type float64Edge struct {
	dst int
	cap float64
}

// newMaxflow allocates a solver for n nodes.
func newMaxflow(n int) *maxflow {
	return &maxflow{
		level: make([]int, n),
		iter:  make([]int, n),
		adj:   make([][]int, n),
	}
}

// addEdge adds a directed edge u→v of capacity capUV and its reverse v→u of
// capacity capVU (use 0 for a purely directed edge).
func (m *maxflow) addEdge(u, v int, capUV, capVU float64) {
	m.adj[u] = append(m.adj[u], len(m.to))
	m.to = append(m.to, float64Edge{dst: v, cap: capUV})
	m.adj[v] = append(m.adj[v], len(m.to))
	m.to = append(m.to, float64Edge{dst: u, cap: capVU})
}

// bfs builds the level graph from s, returning whether t is reachable.
func (m *maxflow) bfs(s, t int) bool {
	for i := range m.level {
		m.level[i] = -1
	}
	queue := []int{s}
	m.level[s] = 0
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		for _, e := range m.adj[u] {
			ed := m.to[e]
			if ed.cap > 1e-12 && m.level[ed.dst] < 0 {
				m.level[ed.dst] = m.level[u] + 1
				queue = append(queue, ed.dst)
			}
		}
	}
	return m.level[t] >= 0
}

// dfs pushes blocking flow along the level graph.
func (m *maxflow) dfs(u, t int, f float64) float64 {
	if u == t {
		return f
	}
	for ; m.iter[u] < len(m.adj[u]); m.iter[u]++ {
		e := m.adj[u][m.iter[u]]
		ed := &m.to[e]
		if ed.cap > 1e-12 && m.level[ed.dst] == m.level[u]+1 {
			d := m.dfs(ed.dst, t, math.Min(f, ed.cap))
			if d > 1e-12 {
				ed.cap -= d
				m.to[e^1].cap += d
				return d
			}
		}
	}
	return 0
}

// maxflow returns the maximum flow (equivalently, the minimum cut cost) from s
// to t.
func (m *maxflow) maxflow(s, t int) float64 {
	var flow float64
	for m.bfs(s, t) {
		for i := range m.iter {
			m.iter[i] = 0
		}
		for {
			f := m.dfs(s, t, math.Inf(1))
			if f <= 1e-12 {
				break
			}
			flow += f
		}
	}
	return flow
}

// minCutReachable returns, for each node, whether it is reachable from s in the
// residual graph — i.e. it lies on the source side of the minimum cut.
func (m *maxflow) minCutReachable(s int) []bool {
	reach := make([]bool, len(m.adj))
	queue := []int{s}
	reach[s] = true
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		for _, e := range m.adj[u] {
			ed := m.to[e]
			if ed.cap > 1e-12 && !reach[ed.dst] {
				reach[ed.dst] = true
				queue = append(queue, ed.dst)
			}
		}
	}
	return reach
}
