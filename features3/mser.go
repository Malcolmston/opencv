package features3

import (
	"image"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// MSERRegion is a Maximally Stable Extremal Region: a connected set of pixels
// that keeps a nearly constant area across a range of intensity thresholds, as
// returned by [MSERRegions].
type MSERRegion struct {
	// Points holds the pixels belonging to the region.
	Points []cv.Point
	// Level is the intensity threshold (0–255) at which the region was measured.
	Level int
	// Dark reports whether the region is a dark region on a lighter background
	// (true) or a bright region on a darker background (false).
	Dark bool
	// Variation is the MSER growth rate at the region's level; smaller is more
	// stable.
	Variation float64
}

// Area returns the number of pixels in the region.
func (r MSERRegion) Area() int {
	return len(r.Points)
}

// Centroid returns the mean location of the region's pixels. It panics on an
// empty region.
func (r MSERRegion) Centroid() cv.Point2f {
	if len(r.Points) == 0 {
		panic("features3: Centroid on empty region")
	}
	var sx, sy float64
	for _, p := range r.Points {
		sx += float64(p.X)
		sy += float64(p.Y)
	}
	n := float64(len(r.Points))
	return cv.Point2f{X: sx / n, Y: sy / n}
}

// BoundingRect returns the axis-aligned bounding box of the region as an
// image.Rectangle (Max exclusive). It panics on an empty region.
func (r MSERRegion) BoundingRect() image.Rectangle {
	if len(r.Points) == 0 {
		panic("features3: BoundingRect on empty region")
	}
	minX, minY := r.Points[0].X, r.Points[0].Y
	maxX, maxY := minX, minY
	for _, p := range r.Points {
		if p.X < minX {
			minX = p.X
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}
	return image.Rect(minX, minY, maxX+1, maxY+1)
}

// features3mserNode is a node of the extremal-region component tree: a component
// at a given intensity level with its area and a representative pixel.
type features3mserNode struct {
	level  int
	area   int
	rep    int
	parent int
}

var features3mser8 = [8][2]int{
	{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1},
}

// features3mserDark extracts maximally stable extremal regions of the "dark on
// light" polarity (thresholding values <= level) from an intensity array by
// building the component tree incrementally over ascending intensity and
// selecting nodes whose growth variation is a local minimum below maxVariation.
func features3mserDark(values []int, rows, cols, delta, minArea, maxArea int, maxVariation float64) []features3mserResult {
	n := rows * cols
	parent := make([]int, n)
	size := make([]int, n)
	nodeOf := make([]int, n)
	active := make([]bool, n)
	for i := range parent {
		parent[i] = -1
	}
	var nodes []features3mserNode

	var find func(int) int
	find = func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}

	// Bucket pixels by intensity (counting sort).
	var buckets [256][]int
	for i := 0; i < n; i++ {
		buckets[values[i]] = append(buckets[values[i]], i)
	}

	union := func(a, b, level int) {
		ra, rb := find(a), find(b)
		if ra == rb {
			return
		}
		if size[ra] < size[rb] {
			ra, rb = rb, ra
		}
		newArea := size[ra] + size[rb]
		merged := features3mserNode{level: level, area: newArea, rep: nodes[nodeOf[ra]].rep, parent: -1}
		nodes = append(nodes, merged)
		mi := len(nodes) - 1
		nodes[nodeOf[ra]].parent = mi
		nodes[nodeOf[rb]].parent = mi
		parent[rb] = ra
		size[ra] = newArea
		nodeOf[ra] = mi
	}

	for g := 0; g < 256; g++ {
		for _, p := range buckets[g] {
			parent[p] = p
			size[p] = 1
			active[p] = true
			nodes = append(nodes, features3mserNode{level: g, area: 1, rep: p, parent: -1})
			nodeOf[p] = len(nodes) - 1
		}
		for _, p := range buckets[g] {
			px := p % cols
			py := p / cols
			for _, o := range features3mser8 {
				nx, ny := px+o[0], py+o[1]
				if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
					continue
				}
				q := ny*cols + nx
				if active[q] {
					union(p, q, g)
				}
			}
		}
	}

	// The region represented by a node persists (unchanged) until a strictly
	// higher-level node exists on its upward path. The area at threshold t is the
	// area of the last node on that path whose level is <= t.
	ancestorArea := func(idx, t int) int {
		cur := idx
		for nodes[cur].parent != -1 && nodes[nodes[cur].parent].level <= t {
			cur = nodes[cur].parent
		}
		return nodes[cur].area
	}
	variation := make([]float64, len(nodes))
	for i := range nodes {
		ap := ancestorArea(i, nodes[i].level+delta)
		variation[i] = float64(ap-nodes[i].area) / float64(nodes[i].area)
	}
	// A node is maximally stable when its variation is a local minimum along its
	// branch (no greater than its parent's) and below maxVariation, with area in
	// range.
	var selected []features3mserResult
	for i := range nodes {
		nd := nodes[i]
		if nd.area < minArea || (maxArea > 0 && nd.area > maxArea) {
			continue
		}
		if variation[i] > maxVariation {
			continue
		}
		if nd.parent != -1 && variation[i] > variation[nd.parent] {
			continue
		}
		selected = append(selected, features3mserResult{level: nd.level, area: nd.area, rep: nd.rep, variation: variation[i]})
	}
	// Deduplicate near-identical regions sharing a representative, keeping the
	// most stable (lowest variation, then smallest area).
	sort.SliceStable(selected, func(a, b int) bool {
		if selected[a].rep != selected[b].rep {
			return selected[a].rep < selected[b].rep
		}
		if selected[a].variation != selected[b].variation {
			return selected[a].variation < selected[b].variation
		}
		return selected[a].area < selected[b].area
	})
	var dedup []features3mserResult
	for _, s := range selected {
		if len(dedup) > 0 {
			last := dedup[len(dedup)-1]
			if last.rep == s.rep && absInt(last.area-s.area)*20 < last.area+1 {
				continue
			}
		}
		dedup = append(dedup, s)
	}
	return dedup
}

// features3mserResult is a selected maximally stable region descriptor.
type features3mserResult struct {
	level     int
	area      int
	rep       int
	variation float64
}

// absInt returns the absolute value of an int.
func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// features3flood returns the pixels of the connected component containing rep in
// the mask defined by the predicate over values, using 8-connectivity.
func features3flood(values []int, rows, cols, rep int, dark bool, level int) []cv.Point {
	inMask := func(v int) bool {
		if dark {
			return v <= level
		}
		return v >= level
	}
	if !inMask(values[rep]) {
		return nil
	}
	visited := make([]bool, rows*cols)
	stack := []int{rep}
	visited[rep] = true
	var pts []cv.Point
	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		px := p % cols
		py := p / cols
		pts = append(pts, cv.Point{X: px, Y: py})
		for _, o := range features3mser8 {
			nx, ny := px+o[0], py+o[1]
			if nx < 0 || nx >= cols || ny < 0 || ny >= rows {
				continue
			}
			q := ny*cols + nx
			if !visited[q] && inMask(values[q]) {
				visited[q] = true
				stack = append(stack, q)
			}
		}
	}
	return pts
}

// MSERRegions extracts Maximally Stable Extremal Regions from an image, the Go
// analogue of OpenCV's cv::MSER. It sweeps intensity thresholds building the
// component tree of extremal regions for both polarities (dark regions on a
// light background and bright regions on a dark background) and returns those
// whose area stays stable across a delta-level band. Only regions with area in
// [minArea, maxArea] (maxArea <= 0 means no upper limit) and growth variation
// below maxVariation are kept. Colour input is converted to grayscale first.
// Results are sorted by descending area.
func MSERRegions(img *cv.Mat, delta, minArea, maxArea int, maxVariation float64) []MSERRegion {
	if delta < 1 {
		delta = 5
	}
	if maxVariation <= 0 {
		maxVariation = 0.25
	}
	g := features3ToGray(img)
	rows, cols := g.Rows, g.Cols
	values := make([]int, rows*cols)
	inv := make([]int, rows*cols)
	for i, v := range g.Data {
		iv := int(v)
		if iv < 0 {
			iv = 0
		} else if iv > 255 {
			iv = 255
		}
		values[i] = iv
		inv[i] = 255 - iv
	}
	var regions []MSERRegion
	// Dark regions (threshold values <= level).
	for _, nd := range features3mserDark(values, rows, cols, delta, minArea, maxArea, maxVariation) {
		pts := features3flood(values, rows, cols, nd.rep, true, nd.level)
		if len(pts) == 0 {
			continue
		}
		regions = append(regions, MSERRegion{Points: pts, Level: nd.level, Dark: true, Variation: nd.variation})
	}
	// Bright regions (threshold values >= level) via the inverted image.
	for _, nd := range features3mserDark(inv, rows, cols, delta, minArea, maxArea, maxVariation) {
		origLevel := 255 - nd.level
		pts := features3flood(values, rows, cols, nd.rep, false, origLevel)
		if len(pts) == 0 {
			continue
		}
		regions = append(regions, MSERRegion{Points: pts, Level: origLevel, Dark: false, Variation: nd.variation})
	}
	sort.SliceStable(regions, func(a, b int) bool {
		if len(regions[a].Points) != len(regions[b].Points) {
			return len(regions[a].Points) > len(regions[b].Points)
		}
		return regions[a].Level < regions[b].Level
	})
	return regions
}
