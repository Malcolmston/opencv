package text

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Region is one detected extremal region: a connected component of an intensity
// level set that is maximally stable across a range of thresholds.
type Region struct {
	// Rect is the axis-aligned bounding box of the region.
	Rect cv.Rect
	// Points holds every pixel of the region, in row-major (top-to-bottom,
	// left-to-right) order.
	Points []cv.Point
	// Level is the intensity threshold at which the region is maximally stable.
	// For a bright region (detected on the inverted image) it is the threshold
	// on the inverted intensity.
	Level int
	// Area is the number of pixels in the region (len(Points)).
	Area int
	// Variation is the region's MSER stability score: the relative area growth
	// over the delta threshold band. Smaller is more stable.
	Variation float64
	// Bright reports whether the region was detected as a bright blob on a
	// darker background (MSER-) rather than a dark blob on a lighter one (MSER+).
	Bright bool
}

// DetectRegionsMSER extracts Maximally Stable Extremal Regions from img and
// returns their bounding boxes. It runs the classic MSER over the component tree
// of the image's intensity level sets, keeping a region whenever its area stays
// nearly constant (its relative growth over the delta threshold band is a local
// minimum below maxVariation) and its pixel area lies in [minArea, maxArea].
//
// Both polarities are searched: dark regions on a lighter background and, by
// inverting the image, bright regions on a darker one. delta is the intensity
// step over which stability is measured (values below 1 are treated as 1).
// maxArea <= 0 is treated as "no upper bound" (the whole image). img may be
// single-channel or three-channel; colour input is reduced to grayscale first.
//
// The returned boxes are sorted top-to-bottom then left-to-right and de-duplicated
// so that near-identical regions from overlapping thresholds or opposite
// polarities appear once.
func DetectRegionsMSER(img *cv.Mat, delta, minArea, maxArea int, maxVariation float64) []cv.Rect {
	regions := MSERRegions(img, delta, minArea, maxArea, maxVariation)
	boxes := make([]cv.Rect, len(regions))
	for i, r := range regions {
		boxes[i] = r.Rect
	}
	return boxes
}

// MSERRegions is like [DetectRegionsMSER] but returns the full [Region] records,
// including the pixel set of each region and the threshold and variation at
// which it was found. See [DetectRegionsMSER] for the meaning of the parameters.
func MSERRegions(img *cv.Mat, delta, minArea, maxArea int, maxVariation float64) []Region {
	gray := toGray(img)
	rows, cols := gray.Rows, gray.Cols
	n := rows * cols
	if delta < 1 {
		delta = 1
	}
	if minArea < 1 {
		minArea = 1
	}
	if maxArea <= 0 || maxArea > n {
		maxArea = n
	}

	// MSER+ : dark regions on a lighter background (lower level sets).
	dark := gray.Data
	// MSER- : bright regions on a darker background — the lower level sets of the
	// inverted image.
	inv := make([]uint8, n)
	for i, v := range gray.Data {
		inv[i] = 255 - v
	}

	var out []Region
	out = append(out, extractMSER(dark, rows, cols, delta, minArea, maxArea, maxVariation, false)...)
	out = append(out, extractMSER(inv, rows, cols, delta, minArea, maxArea, maxVariation, true)...)

	return dedupeRegions(out)
}

// mserNode is one extremal region observed at one threshold level while sweeping
// the intensity level sets from dark to bright.
type mserNode struct {
	level     int
	area      int
	rep       int // a representative (member) pixel index, always the component's min index
	minX      int
	minY      int
	maxX      int
	maxY      int
	parent    int // index into the node slice of the containing region one level up, or -1
	variation float64
}

// extractMSER runs the MSER selection over the lower level sets of vals (a region
// is the set {p : vals[p] <= t} restricted to one connected component). bright
// is recorded on every emitted region so callers can tell the polarity apart.
func extractMSER(vals []uint8, rows, cols, delta, minArea, maxArea int, maxVariation float64, bright bool) []Region {
	n := rows * cols

	// Bucket pixel indices by intensity so each threshold step adds one level.
	var buckets [256][]int
	for i, v := range vals {
		buckets[v] = append(buckets[v], i)
	}

	uf := newIntUnionFind(n)
	added := make([]bool, n)
	addedCount := 0

	var nodes []mserNode
	// prevMap maps the representative pixel of each region observed at the
	// previously snapshotted level to its node index, so we can link a region to
	// its container one level up.
	prevMap := map[int]int{}

	// acc accumulates area and bounding box for one component during a snapshot.
	type acc struct {
		area                   int
		minX, minY, maxX, maxY int
	}

	for t := 0; t < 256; t++ {
		for _, p := range buckets[t] {
			added[p] = true
			addedCount++
			x, y := p%cols, p/cols
			// Union with already-added 8-connected neighbours.
			for dy := -1; dy <= 1; dy++ {
				ny := y + dy
				if ny < 0 || ny >= rows {
					continue
				}
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					nx := x + dx
					if nx < 0 || nx >= cols {
						continue
					}
					q := ny*cols + nx
					if added[q] {
						uf.union(p, q)
					}
				}
			}
		}
		if addedCount == 0 {
			continue // nothing exists yet at this threshold
		}

		// Snapshot: accumulate area and bbox per component root.
		accs := map[int]*acc{}
		for p := 0; p < n; p++ {
			if !added[p] {
				continue
			}
			r := uf.find(p)
			x, y := p%cols, p/cols
			a := accs[r]
			if a == nil {
				a = &acc{area: 0, minX: x, minY: y, maxX: x, maxY: y}
				accs[r] = a
			}
			a.area++
			if x < a.minX {
				a.minX = x
			}
			if x > a.maxX {
				a.maxX = x
			}
			if y < a.minY {
				a.minY = y
			}
			if y > a.maxY {
				a.maxY = y
			}
		}

		// Create nodes in ascending root order for deterministic indexing.
		roots := make([]int, 0, len(accs))
		for r := range accs {
			roots = append(roots, r)
		}
		sort.Ints(roots)
		curMap := make(map[int]int, len(roots))
		for _, r := range roots {
			a := accs[r]
			idx := len(nodes)
			nodes = append(nodes, mserNode{
				level: t, area: a.area, rep: r,
				minX: a.minX, minY: a.minY, maxX: a.maxX, maxY: a.maxY,
				parent: -1,
			})
			curMap[r] = idx
		}

		// Link each region from the previous snapshot to the region that now
		// contains it (its parent one level up).
		prevReps := make([]int, 0, len(prevMap))
		for rep := range prevMap {
			prevReps = append(prevReps, rep)
		}
		sort.Ints(prevReps)
		for _, rep := range prevReps {
			nodes[prevMap[rep]].parent = curMap[uf.find(rep)]
		}
		prevMap = curMap
	}

	if len(nodes) == 0 {
		return nil
	}

	// Stability: variation is the relative area growth over the delta band.
	for i := range nodes {
		target := nodes[i].level + delta
		j := i
		for nodes[j].parent != -1 && nodes[j].level < target {
			j = nodes[j].parent
		}
		nodes[i].variation = float64(nodes[j].area-nodes[i].area) / float64(nodes[i].area)
	}

	// Track, per node, the variation of its main child (the child continuing the
	// branch, i.e. the one with the largest area) so we can test for a local
	// minimum of variation along the branch.
	bestChildArea := make([]int, len(nodes))
	mainChildVar := make([]float64, len(nodes))
	hasChild := make([]bool, len(nodes))
	for i := range nodes {
		p := nodes[i].parent
		if p < 0 {
			continue
		}
		if !hasChild[p] || nodes[i].area > bestChildArea[p] {
			hasChild[p] = true
			bestChildArea[p] = nodes[i].area
			mainChildVar[p] = nodes[i].variation
		}
	}

	const eps = 1e-9
	var picked []mserNode
	for i := range nodes {
		nd := nodes[i]
		if nd.area < minArea || nd.area > maxArea {
			continue
		}
		if nd.variation > maxVariation {
			continue
		}
		// Local minimum of variation: no worse than parent and main child.
		if nd.parent >= 0 && nd.variation > nodes[nd.parent].variation+eps {
			continue
		}
		if hasChild[i] && nd.variation > mainChildVar[i]+eps {
			continue
		}
		picked = append(picked, nd)
	}

	// Reconstruct the exact pixel set of each picked region and emit it.
	regions := make([]Region, 0, len(picked))
	for _, nd := range picked {
		pts := floodComponent(vals, rows, cols, nd.rep, nd.level)
		regions = append(regions, Region{
			Rect: cv.Rect{
				X: nd.minX, Y: nd.minY,
				Width: nd.maxX - nd.minX + 1, Height: nd.maxY - nd.minY + 1,
			},
			Points:    pts,
			Level:     nd.level,
			Area:      len(pts),
			Variation: nd.variation,
			Bright:    bright,
		})
	}
	return regions
}

// floodComponent returns every pixel of the connected component (8-connected) of
// the level set {p : vals[p] <= level} that contains the seed pixel, in
// row-major order.
func floodComponent(vals []uint8, rows, cols, seed, level int) []cv.Point {
	visited := make([]bool, rows*cols)
	stack := []int{seed}
	visited[seed] = true
	var comp []int
	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		comp = append(comp, p)
		x, y := p%cols, p/cols
		for dy := -1; dy <= 1; dy++ {
			ny := y + dy
			if ny < 0 || ny >= rows {
				continue
			}
			for dx := -1; dx <= 1; dx++ {
				if dx == 0 && dy == 0 {
					continue
				}
				nx := x + dx
				if nx < 0 || nx >= cols {
					continue
				}
				q := ny*cols + nx
				if !visited[q] && int(vals[q]) <= level {
					visited[q] = true
					stack = append(stack, q)
				}
			}
		}
	}
	sort.Ints(comp)
	pts := make([]cv.Point, len(comp))
	for i, p := range comp {
		pts[i] = cv.Point{X: p % cols, Y: p / cols}
	}
	return pts
}

// dedupeRegions removes near-duplicate regions (bounding-box IoU >= 0.8),
// keeping the more stable one, and returns the survivors sorted top-to-bottom
// then left-to-right.
func dedupeRegions(in []Region) []Region {
	// Sort by increasing variation so the most stable region of a duplicate set
	// is considered first and kept.
	order := make([]int, len(in))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		ra, rb := in[order[a]], in[order[b]]
		if ra.Variation != rb.Variation {
			return ra.Variation < rb.Variation
		}
		if ra.Area != rb.Area {
			return ra.Area < rb.Area
		}
		return order[a] < order[b]
	})

	var kept []Region
	for _, idx := range order {
		r := in[idx]
		dup := false
		for _, k := range kept {
			if rectIoU(r.Rect, k.Rect) >= 0.8 {
				dup = true
				break
			}
		}
		if !dup {
			kept = append(kept, r)
		}
	}

	sort.SliceStable(kept, func(a, b int) bool {
		ra, rb := kept[a].Rect, kept[b].Rect
		if ra.Y != rb.Y {
			return ra.Y < rb.Y
		}
		if ra.X != rb.X {
			return ra.X < rb.X
		}
		return rectArea(ra) < rectArea(rb)
	})
	return kept
}
