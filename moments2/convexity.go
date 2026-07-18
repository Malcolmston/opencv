package moments2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ConvexityDefect describes a concavity of a contour relative to its convex
// hull: the segment of the hull it lies under (from Start to End) and the
// contour point Far that departs furthest from that segment, at distance Depth.
type ConvexityDefect struct {
	// StartIndex is the index into the contour of the defect's start hull point.
	StartIndex int
	// EndIndex is the index into the contour of the defect's end hull point.
	EndIndex int
	// FarIndex is the index into the contour of the deepest interior point.
	FarIndex int
	// Start is the hull point where the defect begins.
	Start cv.Point
	// End is the hull point where the defect ends.
	End cv.Point
	// Far is the contour point of maximum depth inside the defect.
	Far cv.Point
	// Depth is the perpendicular distance from Far to the segment Start-End.
	Depth float64
}

// ConvexHullIndices returns the indices into contour of its convex hull vertices
// in counter-clockwise order (in image coordinates, where y grows downward),
// computed with Andrew's monotone chain algorithm. Duplicate and collinear
// points are dropped. It returns nil for an empty contour.
func ConvexHullIndices(contour []cv.Point) []int {
	n := len(contour)
	if n == 0 {
		return nil
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	// Sort indices by (X, Y).
	sortIndicesByPoint(contour, idx)
	// Deduplicate identical coordinates.
	uniq := idx[:1]
	for _, id := range idx[1:] {
		last := contour[uniq[len(uniq)-1]]
		if contour[id] != last {
			uniq = append(uniq, id)
		}
	}
	if len(uniq) < 3 {
		out := make([]int, len(uniq))
		copy(out, uniq)
		return out
	}
	cross := func(o, a, b int) int {
		ox, oy := contour[o].X, contour[o].Y
		return (contour[a].X-ox)*(contour[b].Y-oy) - (contour[a].Y-oy)*(contour[b].X-ox)
	}
	m := len(uniq)
	hull := make([]int, 0, 2*m)
	// Lower hull.
	for _, id := range uniq {
		for len(hull) >= 2 && cross(hull[len(hull)-2], hull[len(hull)-1], id) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, id)
	}
	// Upper hull.
	lower := len(hull) + 1
	for i := m - 2; i >= 0; i-- {
		id := uniq[i]
		for len(hull) >= lower && cross(hull[len(hull)-2], hull[len(hull)-1], id) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, id)
	}
	return hull[:len(hull)-1]
}

// sortIndicesByPoint sorts idx so that the referenced points are in ascending
// (X, then Y) order, using a simple insertion sort to avoid a sort import.
func sortIndicesByPoint(pts []cv.Point, idx []int) {
	less := func(a, b int) bool {
		pa, pb := pts[a], pts[b]
		if pa.X != pb.X {
			return pa.X < pb.X
		}
		return pa.Y < pb.Y
	}
	for i := 1; i < len(idx); i++ {
		key := idx[i]
		j := i - 1
		for j >= 0 && less(key, idx[j]) {
			idx[j+1] = idx[j]
			j--
		}
		idx[j+1] = key
	}
}

// moments2pointDistToSegment returns the perpendicular distance from point p to
// the infinite line through a and b (or the distance to a if a==b).
func moments2pointDistToSegment(p, a, b cv.Point) float64 {
	ax, ay := float64(a.X), float64(a.Y)
	bx, by := float64(b.X), float64(b.Y)
	px, py := float64(p.X), float64(p.Y)
	dx, dy := bx-ax, by-ay
	den := math.Hypot(dx, dy)
	if den == 0 {
		return math.Hypot(px-ax, py-ay)
	}
	return math.Abs(dx*(ay-py)-dy*(ax-px)) / den
}

// ConvexityDefects finds the concavities of a contour with respect to its convex
// hull. The hull is given as indices into contour in traversal order, as
// returned by [ConvexHullIndices]. For each pair of consecutive hull vertices it
// scans the contour points between them and, if any lies farther than minDepth
// from the connecting chord, records the deepest one as a defect. Defects are
// returned in hull order. It returns nil when the contour or hull is too small.
func ConvexityDefects(contour []cv.Point, hull []int, minDepth float64) []ConvexityDefect {
	n := len(contour)
	h := len(hull)
	if n < 3 || h < 3 {
		return nil
	}
	// Reorder the hull vertices into contour-traversal order (ascending index)
	// so that consecutive hull vertices bracket a contiguous arc of the
	// contour. This assumes contour points are ordered around the boundary, as
	// produced by contour tracing.
	ordered := make([]int, h)
	copy(ordered, hull)
	for i := 1; i < h; i++ {
		key := ordered[i]
		j := i - 1
		for j >= 0 && ordered[j] > key {
			ordered[j+1] = ordered[j]
			j--
		}
		ordered[j+1] = key
	}
	var defects []ConvexityDefect
	for k := 0; k < h; k++ {
		startIdx := ordered[k]
		endIdx := ordered[(k+1)%h]
		start := contour[startIdx]
		end := contour[endIdx]
		// Walk the contour from startIdx towards endIdx in increasing index
		// (mod n), measuring depth of each intermediate point.
		bestDepth := 0.0
		bestIdx := -1
		i := (startIdx + 1) % n
		for i != endIdx {
			d := moments2pointDistToSegment(contour[i], start, end)
			if d > bestDepth {
				bestDepth = d
				bestIdx = i
			}
			i = (i + 1) % n
		}
		if bestIdx >= 0 && bestDepth > minDepth {
			defects = append(defects, ConvexityDefect{
				StartIndex: startIdx,
				EndIndex:   endIdx,
				FarIndex:   bestIdx,
				Start:      start,
				End:        end,
				Far:        contour[bestIdx],
				Depth:      bestDepth,
			})
		}
	}
	return defects
}

// IsContourConvex reports whether a polygon is convex, that is whether every
// turn along its boundary has the same sign. Collinear vertices are permitted.
// It returns true for fewer than four vertices.
func IsContourConvex(contour []cv.Point) bool {
	n := len(contour)
	if n < 4 {
		return true
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

// ConvexityRatio returns a scalar convexity measure equal to one divided by one
// plus the total defect depth of a contour, normalized so that a convex shape
// scores 1 and deeper concavities score lower. It computes the hull internally.
func ConvexityRatio(contour []cv.Point) float64 {
	hull := ConvexHullIndices(contour)
	defects := ConvexityDefects(contour, hull, 0)
	var total float64
	for _, d := range defects {
		total += d.Depth
	}
	return 1 / (1 + total)
}
