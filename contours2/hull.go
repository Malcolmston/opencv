package contours2

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// ConvexHull returns the vertices of the convex hull of a set of points,
// computed with Andrew's monotone-chain algorithm (an O(n log n) variant that
// yields the same hull as OpenCV's Sklansky routine but is robust for arbitrary
// point sets). When clockwise is false the hull is returned in counter-clockwise
// order in the mathematical sense; when true the order is reversed. Collinear
// interior points are dropped. It panics on an empty point set.
func ConvexHull(pts []cv.Point, clockwise bool) []cv.Point {
	idx := contours2hullIndices(pts)
	out := make([]cv.Point, len(idx))
	for i, id := range idx {
		out[i] = pts[id]
	}
	if clockwise {
		contours2reverse(out)
	}
	return out
}

// ConvexHullIndices returns the indices into pts of the convex-hull vertices,
// sorted in ascending index (contour-traversal) order so that consecutive
// entries bound an arc of the original contour. This ordering is what
// [ConvexityDefects] expects. It panics on an empty point set.
func ConvexHullIndices(pts []cv.Point) []int {
	idx := contours2hullIndices(pts)
	sort.Ints(idx)
	return idx
}

// contours2hullIndices computes the convex hull as a slice of indices into pts,
// in counter-clockwise order, using the monotone-chain algorithm.
func contours2hullIndices(pts []cv.Point) []int {
	n := len(pts)
	if n == 0 {
		panic("contours2: ConvexHull on empty point set")
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		pa, pb := pts[order[a]], pts[order[b]]
		if pa.X != pb.X {
			return pa.X < pb.X
		}
		return pa.Y < pb.Y
	})
	// Deduplicate identical coordinates.
	uniq := order[:0]
	for i, id := range order {
		if i > 0 && pts[id] == pts[order[i-1]] {
			continue
		}
		uniq = append(uniq, id)
	}
	m := len(uniq)
	if m < 3 {
		out := make([]int, m)
		copy(out, uniq)
		return out
	}

	hull := make([]int, 0, 2*m)
	// Lower hull.
	for _, id := range uniq {
		for len(hull) >= 2 && contours2crossi(pts[hull[len(hull)-2]], pts[hull[len(hull)-1]], pts[id]) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, id)
	}
	// Upper hull.
	lower := len(hull) + 1
	for i := m - 2; i >= 0; i-- {
		id := uniq[i]
		for len(hull) >= lower && contours2crossi(pts[hull[len(hull)-2]], pts[hull[len(hull)-1]], pts[id]) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, id)
	}
	return hull[:len(hull)-1]
}

// contours2reverse reverses a slice of points in place.
func contours2reverse(pts []cv.Point) {
	for i, j := 0, len(pts)-1; i < j; i, j = i+1, j-1 {
		pts[i], pts[j] = pts[j], pts[i]
	}
}

// IsContourConvex reports whether a polygon is convex, i.e. every turn between
// consecutive edges has the same orientation. Fewer than three vertices are
// treated as non-convex. This mirrors OpenCV's isContourConvex.
func IsContourConvex(contour []cv.Point) bool {
	n := len(contour)
	if n < 3 {
		return false
	}
	var sign int
	for i := 0; i < n; i++ {
		a := contour[i]
		b := contour[(i+1)%n]
		c := contour[(i+2)%n]
		cross := (b.X-a.X)*(c.Y-b.Y) - (b.Y-a.Y)*(c.X-b.X)
		if cross != 0 {
			s := 1
			if cross < 0 {
				s = -1
			}
			if sign == 0 {
				sign = s
			} else if s != sign {
				return false
			}
		}
	}
	return true
}

// ConvexityDefects finds the concavities of a contour relative to its convex
// hull. hullIndices must be indices into contour as returned by
// [ConvexHullIndices]; if it is nil the hull is computed internally. For each
// hull edge the contour arc between its endpoints is scanned and the point
// farthest from the edge is reported as a [ConvexityDefect] when its depth is
// positive. This mirrors OpenCV's convexityDefects.
func ConvexityDefects(contour []cv.Point, hullIndices []int) []ConvexityDefect {
	n := len(contour)
	if n < 4 {
		return nil
	}
	hull := hullIndices
	if hull == nil {
		hull = ConvexHullIndices(contour)
	} else {
		hull = append([]int(nil), hull...)
		sort.Ints(hull)
	}
	if len(hull) < 3 {
		return nil
	}
	var defects []ConvexityDefect
	h := len(hull)
	for k := 0; k < h; k++ {
		startIdx := hull[k]
		endIdx := hull[(k+1)%h]
		a := contour[startIdx]
		b := contour[endIdx]
		// Walk contour points strictly between startIdx and endIdx.
		farthest := -1
		maxDepth := 0.0
		i := (startIdx + 1) % n
		for i != endIdx {
			d := contours2distToLine(contour[i], a, b)
			if d > maxDepth {
				maxDepth = d
				farthest = i
			}
			i = (i + 1) % n
		}
		if farthest >= 0 && maxDepth > 0 {
			defects = append(defects, ConvexityDefect{
				StartIndex:         startIdx,
				EndIndex:           endIdx,
				FarthestPointIndex: farthest,
				Depth:              maxDepth,
			})
		}
	}
	return defects
}
