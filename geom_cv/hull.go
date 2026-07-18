package geom_cv

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// ConvexHull computes the convex hull of a set of points using Andrew's
// monotone-chain algorithm and returns the hull vertices in counter-clockwise
// order (positive [SignedArea]) with no repeated closing vertex. Duplicate and
// strictly interior collinear points are dropped. Zero, one or two distinct
// input points are returned deduplicated and unchanged. The input slice is not
// modified.
func ConvexHull(pts []cv.Point2f) []cv.Point2f {
	uniq := geom_cvSortedUnique(pts)
	n := len(uniq)
	if n < 3 {
		out := make([]cv.Point2f, n)
		copy(out, uniq)
		return out
	}
	hull := make([]cv.Point2f, 0, 2*n)
	// Lower hull.
	for _, p := range uniq {
		for len(hull) >= 2 && Cross(Sub(hull[len(hull)-1], hull[len(hull)-2]), Sub(p, hull[len(hull)-2])) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, p)
	}
	// Upper hull.
	lower := len(hull) + 1
	for i := n - 2; i >= 0; i-- {
		p := uniq[i]
		for len(hull) >= lower && Cross(Sub(hull[len(hull)-1], hull[len(hull)-2]), Sub(p, hull[len(hull)-2])) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, p)
	}
	return hull[:len(hull)-1]
}

// geom_cvSortedUnique returns the input points sorted by X then Y with exact
// duplicates removed. The input is not modified.
func geom_cvSortedUnique(pts []cv.Point2f) []cv.Point2f {
	s := make([]cv.Point2f, len(pts))
	copy(s, pts)
	sort.Slice(s, func(i, j int) bool {
		if s[i].X != s[j].X {
			return s[i].X < s[j].X
		}
		return s[i].Y < s[j].Y
	})
	out := s[:0]
	for i, p := range s {
		if i == 0 || p.X != s[i-1].X || p.Y != s[i-1].Y {
			out = append(out, p)
		}
	}
	return out
}

// ConvexHullArea returns the area enclosed by the convex hull of the given
// points. It is a convenience wrapper over [ConvexHull] and [PolygonArea].
func ConvexHullArea(pts []cv.Point2f) float64 {
	return PolygonArea(ConvexHull(pts))
}

// AntipodalPairs returns the indices of all antipodal vertex pairs of a convex
// polygon given in counter-clockwise order, computed with the rotating-calipers
// method in linear time. Every diameter (farthest pair) of the polygon is one
// of the returned pairs. The polygon must be convex with no three collinear
// vertices; fewer than two vertices yield nil.
func AntipodalPairs(hull []cv.Point2f) [][2]int {
	n := len(hull)
	if n < 2 {
		return nil
	}
	if n == 2 {
		return [][2]int{{0, 1}}
	}
	area := func(i, j, k int) float64 {
		return math.Abs(Cross(Sub(hull[j], hull[i]), Sub(hull[k], hull[i])))
	}
	var pairs [][2]int
	p := n - 1
	q := 0
	// Advance q until it is the farthest vertex from edge (p, p+1).
	for area(p, (p+1)%n, (q+1)%n) > area(p, (p+1)%n, q) {
		q = (q + 1) % n
	}
	q0 := q
	for i := 0; i <= q0; i++ {
		pairs = append(pairs, [2]int{i, q})
		for area(i, (i+1)%n, (q+1)%n) > area(i, (i+1)%n, q) {
			q = (q + 1) % n
			if !(i == q0 && q == 0) {
				pairs = append(pairs, [2]int{i, q})
			}
		}
		if area(i, (i+1)%n, (q+1)%n) == area(i, (i+1)%n, q) {
			if !(i == q0 && q == n-1) {
				pairs = append(pairs, [2]int{i, (q + 1) % n})
			}
		}
	}
	return pairs
}

// ConvexDiameter returns the diameter of a point set: the largest distance
// between any two of its points. It builds the convex hull and applies rotating
// calipers, so it runs in O(n log n). Fewer than two distinct points yield 0.
func ConvexDiameter(pts []cv.Point2f) float64 {
	hull := ConvexHull(pts)
	n := len(hull)
	if n < 2 {
		return 0
	}
	if n == 2 {
		return Distance(hull[0], hull[1])
	}
	best := 0.0
	for _, pr := range AntipodalPairs(hull) {
		if d := Distance(hull[pr[0]], hull[pr[1]]); d > best {
			best = d
		}
	}
	return best
}

// ConvexWidth returns the minimum width of a point set: the smallest distance
// between two parallel supporting lines. The optimal orientation is always flush
// with a hull edge, so the width is found by measuring, for each hull edge, the
// greatest perpendicular distance to the remaining vertices and taking the
// minimum over all edges. Fewer than three distinct points yield 0.
func ConvexWidth(pts []cv.Point2f) float64 {
	hull := ConvexHull(pts)
	n := len(hull)
	if n < 3 {
		return 0
	}
	best := math.Inf(1)
	for i := 0; i < n; i++ {
		a := hull[i]
		b := hull[(i+1)%n]
		maxDist := 0.0
		for _, p := range hull {
			if d := PointToLineDistance(a, b, p); d > maxDist {
				maxDist = d
			}
		}
		if maxDist < best {
			best = maxDist
		}
	}
	return best
}

// MinAreaRect returns the minimum-area rotated rectangle enclosing the points as
// the parent library's [github.com/malcolmston/opencv.RotatedRect]. It applies
// rotating calipers over the convex hull, measuring the axis-aligned bounding
// box in the frame of each hull edge and keeping the smallest. It panics on an
// empty point set.
func MinAreaRect(pts []cv.Point2f) cv.RotatedRect {
	if len(pts) == 0 {
		panic("geom_cv: MinAreaRect on empty point set")
	}
	hull := ConvexHull(pts)
	if len(hull) == 1 {
		return cv.RotatedRect{CenterX: hull[0].X, CenterY: hull[0].Y}
	}
	if len(hull) == 2 {
		d := Sub(hull[1], hull[0])
		c := Midpoint(hull[0], hull[1])
		return cv.RotatedRect{
			CenterX: c.X,
			CenterY: c.Y,
			Width:   Norm(d),
			Height:  0,
			Angle:   math.Atan2(d.Y, d.X) * 180 / math.Pi,
		}
	}
	n := len(hull)
	bestArea := math.Inf(1)
	var best cv.RotatedRect
	for i := 0; i < n; i++ {
		a := hull[i]
		b := hull[(i+1)%n]
		u := Normalize(Sub(b, a))
		if u == (cv.Point2f{}) {
			continue
		}
		v := Perpendicular(u)
		var minU, maxU, minV, maxV = math.Inf(1), math.Inf(-1), math.Inf(1), math.Inf(-1)
		for _, p := range hull {
			rel := Sub(p, a)
			pu := Dot(rel, u)
			pv := Dot(rel, v)
			minU, maxU = math.Min(minU, pu), math.Max(maxU, pu)
			minV, maxV = math.Min(minV, pv), math.Max(maxV, pv)
		}
		w := maxU - minU
		h := maxV - minV
		if area := w * h; area < bestArea {
			bestArea = area
			cu := (minU + maxU) / 2
			cvv := (minV + maxV) / 2
			center := Add(a, Add(Scale(u, cu), Scale(v, cvv)))
			best = cv.RotatedRect{
				CenterX: center.X,
				CenterY: center.Y,
				Width:   w,
				Height:  h,
				Angle:   math.Atan2(u.Y, u.X) * 180 / math.Pi,
			}
		}
	}
	return best
}
