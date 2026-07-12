package text

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// GroupingOrientation selects how [ERGrouping] links character boxes into text
// lines, mirroring OpenCV's erGrouping ORIENTATION_* modes.
type GroupingOrientation int

const (
	// OrientationHoriz groups only near-horizontal text: two boxes join when they
	// share a similar height, their vertical centres overlap, and the horizontal
	// gap between them is small. This is OpenCV's exhaustive-search mode restricted
	// to horizontal reading order.
	OrientationHoriz GroupingOrientation = iota
	// OrientationAny additionally admits slanted lines by allowing the vertical
	// centre offset to grow with the horizontal separation, so a diagonal run of
	// characters is still collected into one line.
	OrientationAny
)

// ERGrouping clusters filtered character boxes into text lines. It performs an
// exhaustive pairwise linkage: every pair of boxes is tested against the
// orientation's collinearity predicate and compatible boxes are merged with a
// deterministic union-find. Each returned line is sorted left-to-right and the
// lines are ordered top-to-bottom.
//
// This is the geometric core of OpenCV's erGrouping in ORIENTATION_HORIZ (and a
// slope-tolerant ORIENTATION_ANY); the learned pair/triplet feature classifier of
// the original is replaced by the documented thresholds below.
func ERGrouping(boxes []cv.Rect, orientation GroupingOrientation) [][]cv.Rect {
	n := len(boxes)
	if n == 0 {
		return nil
	}
	uf := newIntUnionFind(n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if erGroupCompatible(boxes[i], boxes[j], orientation) {
				uf.union(i, j)
			}
		}
	}
	return collectGroups(boxes, uf, n)
}

// erGroupCompatible reports whether two boxes can share a text line under the
// given orientation. Both modes require similar heights and bounded horizontal
// spacing; they differ in how much vertical centre offset they tolerate.
func erGroupCompatible(a, b cv.Rect, orientation GroupingOrientation) bool {
	if a.Height <= 0 || b.Height <= 0 || a.Width <= 0 || b.Width <= 0 {
		return false
	}
	hi, lo := a.Height, b.Height
	if hi < lo {
		hi, lo = lo, hi
	}
	// Height similarity: characters on a line share a cap height.
	if float64(hi)/float64(lo) > 1.7 {
		return false
	}

	cya := float64(a.Y) + float64(a.Height)/2
	cyb := float64(b.Y) + float64(b.Height)/2
	dcy := math.Abs(cya - cyb)

	gap := horizontalGap(a, b)
	widest := a.Width
	if b.Width > widest {
		widest = b.Width
	}
	// Horizontal proximity: at most ~1.6 box widths of empty space.
	if float64(gap) > 1.6*float64(widest) {
		return false
	}

	switch orientation {
	case OrientationHoriz:
		return dcy <= 0.4*float64(hi)
	case OrientationAny:
		// Allow the baseline to drift with horizontal distance (a bounded slope).
		allowed := 0.4*float64(hi) + 0.6*float64(gap)
		return dcy <= allowed
	default:
		return false
	}
}

// ERGroupingBBox is the simplified, bounding-box-only grouping OpenCV offers as a
// fast alternative to the feature-based grouper. Every box is dilated
// horizontally by a fraction of its width and vertically by a fraction of its
// height; boxes whose dilated rectangles overlap are merged. It ignores height
// similarity, so it is more permissive than [ERGrouping] but needs no per-region
// features.
func ERGroupingBBox(boxes []cv.Rect) [][]cv.Rect {
	n := len(boxes)
	if n == 0 {
		return nil
	}
	// Pre-dilate once for a deterministic, order-independent overlap test.
	dil := make([]cv.Rect, n)
	for i, b := range boxes {
		dx := b.Width // one box width of horizontal reach on each side
		dy := b.Height / 4
		dil[i] = cv.Rect{
			X:      b.X - dx,
			Y:      b.Y - dy,
			Width:  b.Width + 2*dx,
			Height: b.Height + 2*dy,
		}
	}
	uf := newIntUnionFind(n)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if rectsIntersect(dil[i], dil[j]) {
				uf.union(i, j)
			}
		}
	}
	return collectGroups(boxes, uf, n)
}

// collectGroups turns a union-find partition of boxes into deterministically
// ordered text lines (each sorted left-to-right, lines ordered top-to-bottom).
func collectGroups(boxes []cv.Rect, uf *intUnionFind, n int) [][]cv.Rect {
	members := map[int][]int{}
	var roots []int
	for i := 0; i < n; i++ {
		r := uf.find(i)
		if _, ok := members[r]; !ok {
			roots = append(roots, r)
		}
		members[r] = append(members[r], i)
	}
	var lines [][]cv.Rect
	for _, r := range roots {
		idxs := members[r]
		line := make([]cv.Rect, len(idxs))
		for k, idx := range idxs {
			line[k] = boxes[idx]
		}
		sort.SliceStable(line, func(a, b int) bool {
			if line[a].X != line[b].X {
				return line[a].X < line[b].X
			}
			return line[a].Y < line[b].Y
		})
		lines = append(lines, line)
	}
	sort.SliceStable(lines, func(a, b int) bool {
		ca, cb := meanCenterY(lines[a]), meanCenterY(lines[b])
		if ca != cb {
			return ca < cb
		}
		return lines[a][0].X < lines[b][0].X
	})
	return lines
}

// LineBoundingBox returns the smallest axis-aligned rectangle covering every box
// in a group. It panics on an empty group.
func LineBoundingBox(group []cv.Rect) cv.Rect {
	if len(group) == 0 {
		panic("text: LineBoundingBox on empty group")
	}
	minX, minY := group[0].X, group[0].Y
	maxX, maxY := group[0].X+group[0].Width, group[0].Y+group[0].Height
	for _, b := range group[1:] {
		minX = minInt(minX, b.X)
		minY = minInt(minY, b.Y)
		maxX = maxInt(maxX, b.X+b.Width)
		maxY = maxInt(maxY, b.Y+b.Height)
	}
	return cv.Rect{X: minX, Y: minY, Width: maxX - minX, Height: maxY - minY}
}

// rectsIntersect reports whether two rectangles share any area.
func rectsIntersect(a, b cv.Rect) bool {
	return a.X < b.X+b.Width && b.X < a.X+a.Width &&
		a.Y < b.Y+b.Height && b.Y < a.Y+a.Height
}
