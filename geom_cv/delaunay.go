package geom_cv

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// Circumcenter returns the center of the circle passing through the three points
// a, b and c, and true on success. It returns the zero point and false when the
// points are collinear (or coincident), which has no finite circumcenter.
func Circumcenter(a, b, c cv.Point2f) (cv.Point2f, bool) {
	circ, ok := geom_cvCircumcircle(a, b, c)
	if !ok {
		return cv.Point2f{}, false
	}
	return circ.Center, true
}

// Circumcircle returns the unique circle passing through the three points a, b
// and c, and true on success. It returns the zero circle and false when the
// points are collinear.
func Circumcircle(a, b, c cv.Point2f) (Circle, bool) {
	return geom_cvCircumcircle(a, b, c)
}

// geom_cvCircumcircle is the shared implementation behind [Circumcircle] and
// [Circumcenter].
func geom_cvCircumcircle(a, b, c cv.Point2f) (Circle, bool) {
	d := 2 * (a.X*(b.Y-c.Y) + b.X*(c.Y-a.Y) + c.X*(a.Y-b.Y))
	if math.Abs(d) < geom_cvEps {
		return Circle{}, false
	}
	a2 := a.X*a.X + a.Y*a.Y
	b2 := b.X*b.X + b.Y*b.Y
	c2 := c.X*c.X + c.Y*c.Y
	ux := (a2*(b.Y-c.Y) + b2*(c.Y-a.Y) + c2*(a.Y-b.Y)) / d
	uy := (a2*(c.X-b.X) + b2*(a.X-c.X) + c2*(b.X-a.X)) / d
	center := cv.Point2f{X: ux, Y: uy}
	return Circle{Center: center, Radius: Distance(center, a)}, true
}

// InCircumcircle reports whether the point p lies strictly inside the circle
// circumscribing triangle abc. It is the incircle predicate that drives Delaunay
// triangulation. Collinear abc yield false.
func InCircumcircle(a, b, c, p cv.Point2f) bool {
	circ, ok := geom_cvCircumcircle(a, b, c)
	if !ok {
		return false
	}
	return Distance(circ.Center, p) < circ.Radius-geom_cvEps
}

// geom_cvTri holds a triangle as three indices into a working point slice.
type geom_cvTri struct {
	a, b, c int
}

// geom_cvEdge is an unordered pair of point indices used to detect shared edges.
type geom_cvEdge struct {
	u, v int
}

// geom_cvMakeEdge returns an edge with its endpoints in canonical (sorted)
// order so that opposite orientations compare equal.
func geom_cvMakeEdge(u, v int) geom_cvEdge {
	if u > v {
		u, v = v, u
	}
	return geom_cvEdge{u, v}
}

// DelaunayTriangulation computes a Delaunay triangulation of the input points
// using the Bowyer–Watson incremental algorithm and returns the resulting
// triangles. Exact duplicate points are removed and the survivors are sorted, so
// the output is deterministic. Fewer than three distinct points, or fully
// collinear input, yield an empty slice. The input slice is not modified.
func DelaunayTriangulation(points []cv.Point2f) []Triangle {
	pts := geom_cvSortedUnique(points)
	n := len(pts)
	if n < 3 {
		return nil
	}

	// Build a super-triangle that comfortably contains every point.
	minX, minY := pts[0].X, pts[0].Y
	maxX, maxY := pts[0].X, pts[0].Y
	for _, p := range pts {
		minX, maxX = math.Min(minX, p.X), math.Max(maxX, p.X)
		minY, maxY = math.Min(minY, p.Y), math.Max(maxY, p.Y)
	}
	dx := maxX - minX
	dy := maxY - minY
	dmax := math.Max(dx, dy)
	if dmax < geom_cvEps {
		return nil
	}
	midX := (minX + maxX) / 2
	midY := (minY + maxY) / 2
	work := make([]cv.Point2f, n, n+3)
	copy(work, pts)
	s := 20 * dmax
	work = append(work,
		cv.Point2f{X: midX - s, Y: midY - dmax},
		cv.Point2f{X: midX, Y: midY + s},
		cv.Point2f{X: midX + s, Y: midY - dmax},
	)
	super0, super1, super2 := n, n+1, n+2

	tris := []geom_cvTri{{super0, super1, super2}}

	for i := 0; i < n; i++ {
		p := work[i]
		// Find triangles whose circumcircle contains p ("bad" triangles).
		var bad []int
		for ti, t := range tris {
			if geom_cvInCirc(work[t.a], work[t.b], work[t.c], p) {
				bad = append(bad, ti)
			}
		}
		// Collect boundary edges of the polygonal hole (edges not shared by two
		// bad triangles).
		edgeCount := map[geom_cvEdge]int{}
		for _, ti := range bad {
			t := tris[ti]
			edgeCount[geom_cvMakeEdge(t.a, t.b)]++
			edgeCount[geom_cvMakeEdge(t.b, t.c)]++
			edgeCount[geom_cvMakeEdge(t.c, t.a)]++
		}
		// Remove bad triangles (descending index to keep slice valid).
		badSet := make(map[int]bool, len(bad))
		for _, ti := range bad {
			badSet[ti] = true
		}
		kept := tris[:0]
		for ti, t := range tris {
			if !badSet[ti] {
				kept = append(kept, t)
			}
		}
		tris = kept
		// Re-triangulate the hole by connecting p to each boundary edge.
		for e, cnt := range edgeCount {
			if cnt == 1 {
				tris = append(tris, geom_cvTri{e.u, e.v, i})
			}
		}
	}

	// Drop triangles touching the super-triangle vertices and emit the rest.
	var out []Triangle
	for _, t := range tris {
		if t.a >= super0 || t.b >= super0 || t.c >= super0 {
			continue
		}
		out = append(out, Triangle{A: work[t.a], B: work[t.b], C: work[t.c]})
	}
	// Sort for deterministic ordering.
	sort.Slice(out, func(i, j int) bool {
		return geom_cvTriLess(out[i], out[j])
	})
	return out
}

// geom_cvInCirc reports whether p lies inside the circumcircle of abc, treating
// a degenerate (collinear) triangle as containing nothing.
func geom_cvInCirc(a, b, c, p cv.Point2f) bool {
	circ, ok := geom_cvCircumcircle(a, b, c)
	if !ok {
		return false
	}
	return Distance(circ.Center, p) < circ.Radius-geom_cvEps
}

// geom_cvTriLess gives a total order on triangles by their sorted corner
// coordinates, used to make triangulation output deterministic.
func geom_cvTriLess(a, b Triangle) bool {
	ka := geom_cvTriKey(a)
	kb := geom_cvTriKey(b)
	for i := range ka {
		if ka[i] != kb[i] {
			return ka[i] < kb[i]
		}
	}
	return false
}

// geom_cvTriKey returns the six coordinates of a triangle's corners sorted so
// that congruent orderings map to the same key.
func geom_cvTriKey(t Triangle) [6]float64 {
	c := []cv.Point2f{t.A, t.B, t.C}
	sort.Slice(c, func(i, j int) bool {
		if c[i].X != c[j].X {
			return c[i].X < c[j].X
		}
		return c[i].Y < c[j].Y
	})
	return [6]float64{c[0].X, c[0].Y, c[1].X, c[1].Y, c[2].X, c[2].Y}
}
