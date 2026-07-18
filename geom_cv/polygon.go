package geom_cv

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SignedArea returns the signed area of the simple polygon given by its vertices
// in order (the polygon is implicitly closed). The result is positive when the
// vertices wind counter-clockwise in a right-handed frame and negative when they
// wind clockwise. Fewer than three vertices yield 0.
func SignedArea(poly []cv.Point2f) float64 {
	n := len(poly)
	if n < 3 {
		return 0
	}
	var sum float64
	for i := 0; i < n; i++ {
		a := poly[i]
		b := poly[(i+1)%n]
		sum += a.X*b.Y - b.X*a.Y
	}
	return sum / 2
}

// PolygonArea returns the (non-negative) area enclosed by the simple polygon,
// independent of winding direction. Fewer than three vertices yield 0.
func PolygonArea(poly []cv.Point2f) float64 {
	return math.Abs(SignedArea(poly))
}

// PolygonPerimeter returns the total length of the polygon boundary, treating
// the vertex list as a closed loop. Fewer than two vertices yield 0.
func PolygonPerimeter(poly []cv.Point2f) float64 {
	n := len(poly)
	if n < 2 {
		return 0
	}
	var total float64
	for i := 0; i < n; i++ {
		total += Distance(poly[i], poly[(i+1)%n])
	}
	return total
}

// PolygonCentroid returns the area centroid (center of mass of the filled
// region) of the simple polygon. For a degenerate polygon with (near) zero area
// it falls back to the arithmetic mean of the vertices. It panics on an empty
// vertex list.
func PolygonCentroid(poly []cv.Point2f) cv.Point2f {
	n := len(poly)
	if n == 0 {
		panic("geom_cv: PolygonCentroid on empty polygon")
	}
	if n < 3 {
		return geom_cvMean(poly)
	}
	var cx, cy, a2 float64
	for i := 0; i < n; i++ {
		p := poly[i]
		q := poly[(i+1)%n]
		cross := p.X*q.Y - q.X*p.Y
		a2 += cross
		cx += (p.X + q.X) * cross
		cy += (p.Y + q.Y) * cross
	}
	if math.Abs(a2) < geom_cvEps {
		return geom_cvMean(poly)
	}
	return cv.Point2f{X: cx / (3 * a2), Y: cy / (3 * a2)}
}

// geom_cvMean returns the arithmetic mean of a non-empty point slice.
func geom_cvMean(pts []cv.Point2f) cv.Point2f {
	var sx, sy float64
	for _, p := range pts {
		sx += p.X
		sy += p.Y
	}
	n := float64(len(pts))
	return cv.Point2f{X: sx / n, Y: sy / n}
}

// PolygonBoundingBox returns the smallest axis-aligned [BoundingBox] that
// contains every vertex. It panics on an empty vertex list.
func PolygonBoundingBox(poly []cv.Point2f) BoundingBox {
	if len(poly) == 0 {
		panic("geom_cv: PolygonBoundingBox on empty polygon")
	}
	minP := poly[0]
	maxP := poly[0]
	for _, p := range poly[1:] {
		minP.X = math.Min(minP.X, p.X)
		minP.Y = math.Min(minP.Y, p.Y)
		maxP.X = math.Max(maxP.X, p.X)
		maxP.Y = math.Max(maxP.Y, p.Y)
	}
	return BoundingBox{Min: minP, Max: maxP}
}

// PointInPolygon reports whether the point p lies strictly inside the simple
// polygon, using the even-odd ray-casting rule. Points exactly on an edge give
// an implementation-defined but deterministic result; use
// [PointOnPolygonBoundary] first if boundary membership matters.
func PointInPolygon(poly []cv.Point2f, p cv.Point2f) bool {
	n := len(poly)
	if n < 3 {
		return false
	}
	inside := false
	j := n - 1
	for i := 0; i < n; i++ {
		pi := poly[i]
		pj := poly[j]
		if (pi.Y > p.Y) != (pj.Y > p.Y) {
			xint := (pj.X-pi.X)*(p.Y-pi.Y)/(pj.Y-pi.Y) + pi.X
			if p.X < xint {
				inside = !inside
			}
		}
		j = i
	}
	return inside
}

// PointOnPolygonBoundary reports whether the point p lies on any edge of the
// polygon, within the package tolerance.
func PointOnPolygonBoundary(poly []cv.Point2f, p cv.Point2f) bool {
	n := len(poly)
	if n < 2 {
		return false
	}
	for i := 0; i < n; i++ {
		if PointOnSegment(poly[i], poly[(i+1)%n], p) {
			return true
		}
	}
	return false
}

// WindingNumber returns the winding number of the closed polygon around the
// point p: the signed number of times the polygon boundary wraps around p
// counter-clockwise. It is nonzero exactly when p is enclosed, and unlike the
// even-odd rule it distinguishes self-overlapping regions.
func WindingNumber(poly []cv.Point2f, p cv.Point2f) int {
	n := len(poly)
	if n < 3 {
		return 0
	}
	wn := 0
	for i := 0; i < n; i++ {
		a := poly[i]
		b := poly[(i+1)%n]
		if a.Y <= p.Y {
			if b.Y > p.Y && Cross(Sub(b, a), Sub(p, a)) > 0 {
				wn++
			}
		} else {
			if b.Y <= p.Y && Cross(Sub(b, a), Sub(p, a)) < 0 {
				wn--
			}
		}
	}
	return wn
}

// IsConvex reports whether the simple polygon is convex, i.e. every interior
// turn has the same orientation. Collinear vertices are tolerated. Fewer than
// three vertices are not convex.
func IsConvex(poly []cv.Point2f) bool {
	n := len(poly)
	if n < 3 {
		return false
	}
	sign := 0
	for i := 0; i < n; i++ {
		o := Orientation(poly[i], poly[(i+1)%n], poly[(i+2)%n])
		if o == 0 {
			continue
		}
		if sign == 0 {
			sign = o
		} else if o != sign {
			return false
		}
	}
	return true
}

// IsClockwise reports whether the polygon vertices wind clockwise in a
// right-handed frame (equivalently, whether [SignedArea] is negative). Because
// the image Y axis points down, this winding appears counter-clockwise on
// screen.
func IsClockwise(poly []cv.Point2f) bool {
	return SignedArea(poly) < 0
}

// ReversePolygon returns a new slice with the vertices in reverse order, which
// flips the polygon winding. The input is not modified.
func ReversePolygon(poly []cv.Point2f) []cv.Point2f {
	n := len(poly)
	out := make([]cv.Point2f, n)
	for i := 0; i < n; i++ {
		out[i] = poly[n-1-i]
	}
	return out
}

// EnsureCounterClockwise returns the polygon with a guaranteed counter-clockwise
// winding (positive [SignedArea]): the input is returned as a copy if it already
// winds counter-clockwise and reversed otherwise. The input is not modified.
func EnsureCounterClockwise(poly []cv.Point2f) []cv.Point2f {
	if SignedArea(poly) < 0 {
		return ReversePolygon(poly)
	}
	out := make([]cv.Point2f, len(poly))
	copy(out, poly)
	return out
}
