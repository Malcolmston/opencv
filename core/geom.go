package core

import (
	"math"
	"sort"
)

// Dist2i returns the Euclidean distance between two integer points.
func Dist2i(a, b Point2i) float64 { return math.Hypot(float64(a.X-b.X), float64(a.Y-b.Y)) }

// Dist2f returns the Euclidean distance between two float32 points.
func Dist2f(a, b Point2f) float64 {
	return math.Hypot(float64(a.X-b.X), float64(a.Y-b.Y))
}

// Dist2d returns the Euclidean distance between two float64 points.
func Dist2d(a, b Point2d) float64 { return math.Hypot(a.X-b.X, a.Y-b.Y) }

// Dist3d returns the Euclidean distance between two 3D float64 points.
func Dist3d(a, b Point3d) float64 {
	dx, dy, dz := a.X-b.X, a.Y-b.Y, a.Z-b.Z
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

// ManhattanDist2i returns the L1 (city-block) distance between two integer
// points.
func ManhattanDist2i(a, b Point2i) int {
	return absInt(a.X-b.X) + absInt(a.Y-b.Y)
}

// MidPoint2f returns the midpoint of segment a-b.
func MidPoint2f(a, b Point2f) Point2f { return Point2f{(a.X + b.X) / 2, (a.Y + b.Y) / 2} }

// Lerp2f linearly interpolates from a to b by parameter t (0 gives a, 1 gives b).
func Lerp2f(a, b Point2f, t float64) Point2f {
	return Point2f{
		float32(float64(a.X) + (float64(b.X)-float64(a.X))*t),
		float32(float64(a.Y) + (float64(b.Y)-float64(a.Y))*t),
	}
}

// Lerp2d linearly interpolates from a to b by parameter t.
func Lerp2d(a, b Point2d, t float64) Point2d {
	return Point2d{a.X + (b.X-a.X)*t, a.Y + (b.Y-a.Y)*t}
}

// BoundingRect2i returns the smallest upright rectangle containing every point.
// Width and Height are inclusive of the extreme pixels, so a single point gives
// a 1x1 rectangle. It panics on an empty slice.
func BoundingRect2i(pts []Point2i) Rect2i {
	if len(pts) == 0 {
		panic("core: BoundingRect2i on empty point set")
	}
	minX, minY := pts[0].X, pts[0].Y
	maxX, maxY := pts[0].X, pts[0].Y
	for _, p := range pts[1:] {
		minX, maxX = minInt(minX, p.X), maxInt(maxX, p.X)
		minY, maxY = minInt(minY, p.Y), maxInt(maxY, p.Y)
	}
	return Rect2i{minX, minY, maxX - minX + 1, maxY - minY + 1}
}

// BoundingRect2f returns the smallest upright rectangle containing every point.
// It panics on an empty slice.
func BoundingRect2f(pts []Point2f) Rect2f {
	if len(pts) == 0 {
		panic("core: BoundingRect2f on empty point set")
	}
	minX, minY := pts[0].X, pts[0].Y
	maxX, maxY := pts[0].X, pts[0].Y
	for _, p := range pts[1:] {
		minX, maxX = minF32(minX, p.X), maxF32(maxX, p.X)
		minY, maxY = minF32(minY, p.Y), maxF32(maxY, p.Y)
	}
	return Rect2f{minX, minY, maxX - minX, maxY - minY}
}

// BoundingRect2d returns the smallest upright rectangle containing every point.
// It panics on an empty slice.
func BoundingRect2d(pts []Point2d) Rect2d {
	if len(pts) == 0 {
		panic("core: BoundingRect2d on empty point set")
	}
	minX, minY := pts[0].X, pts[0].Y
	maxX, maxY := pts[0].X, pts[0].Y
	for _, p := range pts[1:] {
		minX, maxX = minF64(minX, p.X), maxF64(maxX, p.X)
		minY, maxY = minF64(minY, p.Y), maxF64(maxY, p.Y)
	}
	return Rect2d{minX, minY, maxX - minX, maxY - minY}
}

// Centroid2f returns the arithmetic mean of the points. It panics on an empty
// slice.
func Centroid2f(pts []Point2f) Point2f {
	if len(pts) == 0 {
		panic("core: Centroid2f on empty point set")
	}
	var sx, sy float64
	for _, p := range pts {
		sx += float64(p.X)
		sy += float64(p.Y)
	}
	n := float64(len(pts))
	return Point2f{float32(sx / n), float32(sy / n)}
}

// Centroid2d returns the arithmetic mean of the points. It panics on an empty
// slice.
func Centroid2d(pts []Point2d) Point2d {
	if len(pts) == 0 {
		panic("core: Centroid2d on empty point set")
	}
	var sx, sy float64
	for _, p := range pts {
		sx += p.X
		sy += p.Y
	}
	n := float64(len(pts))
	return Point2d{sx / n, sy / n}
}

// SignedPolygonArea2f returns the signed area of the polygon by the shoelace
// formula; the sign is positive when the vertices wind counter-clockwise in a
// y-up frame (negative in image coordinates where y grows downward).
func SignedPolygonArea2f(pts []Point2f) float64 {
	n := len(pts)
	if n < 3 {
		return 0
	}
	var s float64
	for i := 0; i < n; i++ {
		a := pts[i]
		b := pts[(i+1)%n]
		s += float64(a.X)*float64(b.Y) - float64(b.X)*float64(a.Y)
	}
	return s / 2
}

// PolygonArea2f returns the absolute area of the polygon.
func PolygonArea2f(pts []Point2f) float64 { return math.Abs(SignedPolygonArea2f(pts)) }

// Perimeter2f returns the total edge length of a point sequence. When closed is
// true the segment from the last point back to the first is included.
func Perimeter2f(pts []Point2f, closed bool) float64 {
	n := len(pts)
	if n < 2 {
		return 0
	}
	var total float64
	for i := 0; i+1 < n; i++ {
		total += Dist2f(pts[i], pts[i+1])
	}
	if closed {
		total += Dist2f(pts[n-1], pts[0])
	}
	return total
}

// orient2f returns twice the signed area of triangle o,a,b (>0 means a left
// turn in a y-up frame).
func orient2f(o, a, b Point2f) float64 {
	return float64(a.X-o.X)*float64(b.Y-o.Y) - float64(a.Y-o.Y)*float64(b.X-o.X)
}

// ConvexHull2f computes the convex hull of the points with Andrew's monotone
// chain algorithm and returns the hull vertices in counter-clockwise order (in
// a y-up frame). Fewer than three input points are returned deduplicated.
func ConvexHull2f(pts []Point2f) []Point2f {
	n := len(pts)
	if n < 3 {
		out := make([]Point2f, n)
		copy(out, pts)
		return out
	}
	p := make([]Point2f, n)
	copy(p, pts)
	sort.Slice(p, func(i, j int) bool {
		if p[i].X != p[j].X {
			return p[i].X < p[j].X
		}
		return p[i].Y < p[j].Y
	})
	hull := make([]Point2f, 0, 2*n)
	for _, q := range p { // lower
		for len(hull) >= 2 && orient2f(hull[len(hull)-2], hull[len(hull)-1], q) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, q)
	}
	lower := len(hull) + 1
	for i := n - 2; i >= 0; i-- { // upper
		q := p[i]
		for len(hull) >= lower && orient2f(hull[len(hull)-2], hull[len(hull)-1], q) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, q)
	}
	return hull[:len(hull)-1]
}

// IsConvex2f reports whether the closed polygon is convex, i.e. all its turns
// have the same orientation. Polygons with fewer than three vertices are
// treated as convex.
func IsConvex2f(pts []Point2f) bool {
	n := len(pts)
	if n < 3 {
		return true
	}
	var sign int
	for i := 0; i < n; i++ {
		o := pts[i]
		a := pts[(i+1)%n]
		b := pts[(i+2)%n]
		cr := orient2f(o, a, b)
		if cr != 0 {
			s := 1
			if cr < 0 {
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

// RotatePoint2f rotates p about center by angleDeg degrees clockwise in image
// coordinates (y down) and returns the result.
func RotatePoint2f(p, center Point2f, angleDeg float64) Point2f {
	rad := angleDeg * math.Pi / 180
	c, s := math.Cos(rad), math.Sin(rad)
	dx := float64(p.X - center.X)
	dy := float64(p.Y - center.Y)
	return Point2f{
		center.X + float32(dx*c-dy*s),
		center.Y + float32(dx*s+dy*c),
	}
}

// RotatePoint2d rotates p about center by angleDeg degrees clockwise in image
// coordinates (y down) and returns the result.
func RotatePoint2d(p, center Point2d, angleDeg float64) Point2d {
	rad := angleDeg * math.Pi / 180
	c, s := math.Cos(rad), math.Sin(rad)
	dx := p.X - center.X
	dy := p.Y - center.Y
	return Point2d{center.X + dx*c - dy*s, center.Y + dx*s + dy*c}
}

// Angle2f returns the direction of the vector from a to b in radians, measured
// with math.Atan2.
func Angle2f(a, b Point2f) float64 {
	return math.Atan2(float64(b.Y-a.Y), float64(b.X-a.X))
}

// PointToSegmentDist2f returns the shortest distance from point p to the
// segment a-b.
func PointToSegmentDist2f(p, a, b Point2f) float64 {
	ax, ay := float64(a.X), float64(a.Y)
	bx, by := float64(b.X), float64(b.Y)
	px, py := float64(p.X), float64(p.Y)
	dx, dy := bx-ax, by-ay
	if dx == 0 && dy == 0 {
		return math.Hypot(px-ax, py-ay)
	}
	t := ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)
	if t < 0 {
		t = 0
	} else if t > 1 {
		t = 1
	}
	return math.Hypot(px-(ax+t*dx), py-(ay+t*dy))
}

// SegmentsIntersect2f reports whether segments p1-p2 and p3-p4 cross and, when
// they do, returns their intersection point. Collinear overlap returns false.
func SegmentsIntersect2f(p1, p2, p3, p4 Point2f) (Point2f, bool) {
	x1, y1 := float64(p1.X), float64(p1.Y)
	x2, y2 := float64(p2.X), float64(p2.Y)
	x3, y3 := float64(p3.X), float64(p3.Y)
	x4, y4 := float64(p4.X), float64(p4.Y)
	den := (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
	if den == 0 {
		return Point2f{}, false
	}
	t := ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den
	u := ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / den
	if t < 0 || t > 1 || u < 0 || u > 1 {
		return Point2f{}, false
	}
	return Point2f{float32(x1 + t*(x2-x1)), float32(y1 + t*(y2-y1))}, true
}

// PointInTriangle2f reports whether point p lies inside (or on the boundary of)
// triangle a,b,c.
func PointInTriangle2f(p, a, b, c Point2f) bool {
	d1 := orient2f(p, a, b)
	d2 := orient2f(p, b, c)
	d3 := orient2f(p, c, a)
	hasNeg := d1 < 0 || d2 < 0 || d3 < 0
	hasPos := d1 > 0 || d2 > 0 || d3 > 0
	return !(hasNeg && hasPos)
}

// Clamp2i clamps point p to lie within rectangle r (the returned point is on
// the closed range [X, X+Width-1] x [Y, Y+Height-1]).
func Clamp2i(p Point2i, r Rect2i) Point2i {
	x := p.X
	if x < r.X {
		x = r.X
	} else if x > r.X+r.Width-1 {
		x = r.X + r.Width - 1
	}
	y := p.Y
	if y < r.Y {
		y = r.Y
	} else if y > r.Y+r.Height-1 {
		y = r.Y + r.Height - 1
	}
	return Point2i{x, y}
}

// absInt returns the absolute value of x.
func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
