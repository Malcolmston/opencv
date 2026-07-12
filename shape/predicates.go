package shape

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// IsContourConvex reports whether the closed polygon described by contour is
// convex. A polygon is convex when, walking its boundary, every turn has the
// same orientation (all left turns or all right turns); collinear vertices are
// permitted. This mirrors OpenCV's cv::isContourConvex.
//
// The contour is treated as closed (the last vertex joins the first). Contours
// with fewer than three vertices are trivially convex and return true. The test
// also rejects self-intersecting polygons whose total turning exceeds one full
// revolution.
func IsContourConvex(contour []cv.Point) bool {
	n := len(contour)
	if n < 3 {
		return true
	}
	var sign int // 0 = undecided, +1 / -1 once an orientation is seen
	var turnSum float64
	prevAngle := 0.0
	haveAngle := false
	for i := 0; i < n; i++ {
		a := contour[i]
		b := contour[(i+1)%n]
		c := contour[(i+2)%n]
		dx1 := float64(b.X - a.X)
		dy1 := float64(b.Y - a.Y)
		dx2 := float64(c.X - b.X)
		dy2 := float64(c.Y - b.Y)
		cross := dx1*dy2 - dy1*dx2
		if cross > 0 {
			if sign < 0 {
				return false
			}
			sign = 1
		} else if cross < 0 {
			if sign > 0 {
				return false
			}
			sign = -1
		}
		// Accumulate signed exterior angles to detect polygons that wind more
		// than once (which are non-convex even if every turn shares a sign).
		if dx2 != 0 || dy2 != 0 {
			ang := math.Atan2(dy2, dx2)
			if haveAngle {
				d := ang - prevAngle
				for d > math.Pi {
					d -= 2 * math.Pi
				}
				for d < -math.Pi {
					d += 2 * math.Pi
				}
				turnSum += d
			}
			prevAngle = ang
			haveAngle = true
		}
	}
	return math.Abs(turnSum) <= 2*math.Pi+1e-6
}

// PointPolygonTest determines the relationship between the point pt and the
// closed polygon contour, mirroring OpenCV's cv::pointPolygonTest.
//
// When measureDist is false the result is a pure sign: +1 if pt is strictly
// inside, -1 if strictly outside and 0 if it lies on an edge or vertex. When
// measureDist is true the result is the signed Euclidean distance from pt to the
// nearest edge — positive inside, negative outside and 0 exactly on the
// boundary. Insideness is decided by the even–odd ray-crossing rule.
func PointPolygonTest(contour []cv.Point, pt Point2D, measureDist bool) float64 {
	n := len(contour)
	if n < 2 {
		if measureDist {
			if n == 1 {
				d := dist2D(pt, Point2D{X: float64(contour[0].X), Y: float64(contour[0].Y)})
				return -d
			}
			return -math.MaxFloat64
		}
		return -1
	}

	inside := false
	minDist := math.Inf(1)
	onEdge := false
	for i := 0; i < n; i++ {
		a := Point2D{X: float64(contour[i].X), Y: float64(contour[i].Y)}
		b := Point2D{X: float64(contour[(i+1)%n].X), Y: float64(contour[(i+1)%n].Y)}
		// Ray-crossing test (horizontal ray to +X).
		if (a.Y > pt.Y) != (b.Y > pt.Y) {
			xCross := a.X + (pt.Y-a.Y)/(b.Y-a.Y)*(b.X-a.X)
			if pt.X < xCross {
				inside = !inside
			}
		}
		d := segmentDistance(pt, a, b)
		if d < minDist {
			minDist = d
		}
		if d < 1e-9 {
			onEdge = true
		}
	}

	if !measureDist {
		switch {
		case onEdge:
			return 0
		case inside:
			return 1
		default:
			return -1
		}
	}
	if onEdge {
		return 0
	}
	if inside {
		return minDist
	}
	return -minDist
}

// segmentDistance returns the distance from p to the segment a–b.
func segmentDistance(p, a, b Point2D) float64 {
	dx := b.X - a.X
	dy := b.Y - a.Y
	len2 := dx*dx + dy*dy
	if len2 < 1e-18 {
		return dist2D(p, a)
	}
	t := ((p.X-a.X)*dx + (p.Y-a.Y)*dy) / len2
	t = clamp(t, 0, 1)
	proj := Point2D{X: a.X + t*dx, Y: a.Y + t*dy}
	return dist2D(p, proj)
}

// IntersectionKind classifies the outcome of [RotatedRectangleIntersection],
// mirroring OpenCV's RectanglesIntersectTypes.
type IntersectionKind int

const (
	// IntersectNone means the rectangles do not overlap.
	IntersectNone IntersectionKind = 0
	// IntersectPartial means the rectangles overlap in a region that is neither
	// empty nor a whole rectangle.
	IntersectPartial IntersectionKind = 1
	// IntersectFull means one rectangle is fully contained in the other, so the
	// intersection equals the smaller rectangle.
	IntersectFull IntersectionKind = 2
)

// RotatedRectangleIntersection computes the overlap of two rotated rectangles,
// mirroring OpenCV's cv::rotatedRectangleIntersection. It returns the kind of
// intersection and, when there is one, the vertices of the (convex) intersection
// polygon in order. The polygon is empty when the kind is [IntersectNone].
//
// The intersection is found by clipping the first rectangle's quadrilateral
// against the second with the Sutherland–Hodgman algorithm, then deduplicating
// near-coincident vertices. Because both operands are convex the result is a
// convex polygon of at most eight vertices.
func RotatedRectangleIntersection(r1, r2 cv.RotatedRect) (IntersectionKind, []Point2D) {
	poly1 := rectCorners(r1)
	poly2 := rectCorners(r2)
	clipped := clipPolygon(poly1, poly2)
	clipped = dedupePolygon(clipped)
	if len(clipped) < 3 {
		// Fall back: a rectangle fully inside the other yields its own corners.
		if allInside(poly1, poly2) {
			return IntersectFull, orderConvex(poly1)
		}
		if allInside(poly2, poly1) {
			return IntersectFull, orderConvex(poly2)
		}
		return IntersectNone, nil
	}
	kind := IntersectPartial
	if allInside(poly1, poly2) || allInside(poly2, poly1) {
		kind = IntersectFull
	}
	return kind, orderConvex(clipped)
}

// rectCorners returns the four corners of a rotated rectangle as float points,
// in order around the rectangle.
func rectCorners(r cv.RotatedRect) []Point2D {
	rad := r.Angle * math.Pi / 180
	c, s := math.Cos(rad), math.Sin(rad)
	hw, hh := r.Width/2, r.Height/2
	local := [4][2]float64{{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}}
	out := make([]Point2D, 4)
	for i, p := range local {
		out[i] = Point2D{
			X: r.CenterX + p[0]*c - p[1]*s,
			Y: r.CenterY + p[0]*s + p[1]*c,
		}
	}
	return out
}

// clipPolygon clips the convex subject polygon against the convex clip polygon
// using the Sutherland–Hodgman algorithm and returns the intersection.
func clipPolygon(subject, clip []Point2D) []Point2D {
	// Ensure the clip polygon is counter-clockwise so "inside" is a left turn.
	clip = ensureCCW(clip)
	output := make([]Point2D, len(subject))
	copy(output, subject)
	nc := len(clip)
	for i := 0; i < nc; i++ {
		a := clip[i]
		b := clip[(i+1)%nc]
		input := output
		output = output[:0:0]
		m := len(input)
		if m == 0 {
			break
		}
		for j := 0; j < m; j++ {
			cur := input[j]
			nxt := input[(j+1)%m]
			curIn := edgeSide(a, b, cur) >= -1e-9
			nxtIn := edgeSide(a, b, nxt) >= -1e-9
			if curIn {
				output = append(output, cur)
				if !nxtIn {
					if p, ok := lineIntersect(a, b, cur, nxt); ok {
						output = append(output, p)
					}
				}
			} else if nxtIn {
				if p, ok := lineIntersect(a, b, cur, nxt); ok {
					output = append(output, p)
				}
			}
		}
	}
	return output
}

// edgeSide returns a positive value when p is to the left of the directed edge
// a→b, negative to the right and zero on the line.
func edgeSide(a, b, p Point2D) float64 {
	return (b.X-a.X)*(p.Y-a.Y) - (b.Y-a.Y)*(p.X-a.X)
}

// lineIntersect intersects the infinite line through a,b with the segment c–d.
func lineIntersect(a, b, c, d Point2D) (Point2D, bool) {
	r1x, r1y := b.X-a.X, b.Y-a.Y
	r2x, r2y := d.X-c.X, d.Y-c.Y
	den := r1x*r2y - r1y*r2x
	if math.Abs(den) < 1e-12 {
		return Point2D{}, false
	}
	t := ((c.X-a.X)*r2y - (c.Y-a.Y)*r2x) / den
	return Point2D{X: a.X + t*r1x, Y: a.Y + t*r1y}, true
}

// ensureCCW returns poly reordered (reversed if necessary) to counter-clockwise
// winding in image coordinates (where y grows downward, that is clockwise in
// screen terms — we use the signed-area sign consistently).
func ensureCCW(poly []Point2D) []Point2D {
	if polygonSignedArea(poly) < 0 {
		out := make([]Point2D, len(poly))
		for i, p := range poly {
			out[len(poly)-1-i] = p
		}
		return out
	}
	return poly
}

// polygonSignedArea returns twice the signed area of poly.
func polygonSignedArea(poly []Point2D) float64 {
	var a float64
	n := len(poly)
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		a += poly[i].X*poly[j].Y - poly[j].X*poly[i].Y
	}
	return a
}

// dedupePolygon removes consecutive near-coincident vertices.
func dedupePolygon(poly []Point2D) []Point2D {
	if len(poly) == 0 {
		return poly
	}
	out := poly[:1:1]
	out[0] = poly[0]
	for _, p := range poly[1:] {
		if dist2D(p, out[len(out)-1]) > 1e-6 {
			out = append(out, p)
		}
	}
	if len(out) > 1 && dist2D(out[0], out[len(out)-1]) < 1e-6 {
		out = out[:len(out)-1]
	}
	return out
}

// orderConvex sorts the vertices of a convex polygon counter-clockwise about
// their centroid so the returned outline is well-formed.
func orderConvex(poly []Point2D) []Point2D {
	n := len(poly)
	if n < 3 {
		return poly
	}
	var cx, cy float64
	for _, p := range poly {
		cx += p.X
		cy += p.Y
	}
	cx /= float64(n)
	cy /= float64(n)
	out := make([]Point2D, n)
	copy(out, poly)
	// Insertion sort by polar angle about the centroid (stable, deterministic).
	angle := func(p Point2D) float64 { return math.Atan2(p.Y-cy, p.X-cx) }
	for i := 1; i < n; i++ {
		for j := i; j > 0 && angle(out[j-1]) > angle(out[j]); j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}

// allInside reports whether every vertex of inner lies inside (or on) outer.
func allInside(inner, outer []Point2D) bool {
	outer = ensureCCW(outer)
	no := len(outer)
	for _, p := range inner {
		for i := 0; i < no; i++ {
			a := outer[i]
			b := outer[(i+1)%no]
			if edgeSide(a, b, p) < -1e-6 {
				return false
			}
		}
	}
	return true
}
