package geom_cv

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// PointOnSegment reports whether the point p lies on the segment ab, within the
// package tolerance. It requires both collinearity and containment within the
// segment's bounding box.
func PointOnSegment(a, b, p cv.Point2f) bool {
	if math.Abs(Cross(Sub(b, a), Sub(p, a))) > geom_cvEps*(1+Norm(Sub(b, a))) {
		return false
	}
	minX, maxX := math.Min(a.X, b.X), math.Max(a.X, b.X)
	minY, maxY := math.Min(a.Y, b.Y), math.Max(a.Y, b.Y)
	return p.X >= minX-geom_cvEps && p.X <= maxX+geom_cvEps &&
		p.Y >= minY-geom_cvEps && p.Y <= maxY+geom_cvEps
}

// ClosestPointOnSegment returns the point on segment ab that is nearest to p,
// which is one of the endpoints when the perpendicular foot falls outside the
// segment.
func ClosestPointOnSegment(a, b, p cv.Point2f) cv.Point2f {
	ab := Sub(b, a)
	l2 := Dot(ab, ab)
	if l2 < geom_cvEps {
		return a
	}
	t := Dot(Sub(p, a), ab) / l2
	if t < 0 {
		t = 0
	} else if t > 1 {
		t = 1
	}
	return Add(a, Scale(ab, t))
}

// PointToSegmentDistance returns the shortest distance from the point p to the
// segment ab.
func PointToSegmentDistance(a, b, p cv.Point2f) float64 {
	return Distance(p, ClosestPointOnSegment(a, b, p))
}

// PointToLineDistance returns the perpendicular distance from the point p to the
// infinite line through a and b. If a and b coincide it returns the distance to
// that point.
func PointToLineDistance(a, b, p cv.Point2f) float64 {
	ab := Sub(b, a)
	n := Norm(ab)
	if n < geom_cvEps {
		return Distance(a, p)
	}
	return math.Abs(Cross(ab, Sub(p, a))) / n
}

// SegmentsIntersect reports whether the two closed segments p1p2 and p3p4 share
// at least one point, including collinear overlap and touching at an endpoint.
func SegmentsIntersect(p1, p2, p3, p4 cv.Point2f) bool {
	d1 := Orientation(p3, p4, p1)
	d2 := Orientation(p3, p4, p2)
	d3 := Orientation(p1, p2, p3)
	d4 := Orientation(p1, p2, p4)
	if d1 != d2 && d3 != d4 {
		return true
	}
	if d1 == 0 && PointOnSegment(p3, p4, p1) {
		return true
	}
	if d2 == 0 && PointOnSegment(p3, p4, p2) {
		return true
	}
	if d3 == 0 && PointOnSegment(p1, p2, p3) {
		return true
	}
	if d4 == 0 && PointOnSegment(p1, p2, p4) {
		return true
	}
	return false
}

// SegmentIntersection computes the intersection point of the two segments p1p2
// and p3p4. It returns the point and true when the segments cross at a single
// point. When the segments are parallel, collinear, or only meet outside their
// extents it returns the zero point and false; collinear overlap is reported as
// no single intersection.
func SegmentIntersection(p1, p2, p3, p4 cv.Point2f) (cv.Point2f, bool) {
	r := Sub(p2, p1)
	s := Sub(p4, p3)
	denom := Cross(r, s)
	if math.Abs(denom) < geom_cvEps {
		return cv.Point2f{}, false
	}
	qp := Sub(p3, p1)
	t := Cross(qp, s) / denom
	u := Cross(qp, r) / denom
	if t < -geom_cvEps || t > 1+geom_cvEps || u < -geom_cvEps || u > 1+geom_cvEps {
		return cv.Point2f{}, false
	}
	return Add(p1, Scale(r, t)), true
}

// LineIntersection computes the intersection point of the two infinite lines
// through (a1,a2) and (b1,b2). It returns the point and true, or the zero point
// and false when the lines are parallel (or degenerate).
func LineIntersection(a1, a2, b1, b2 cv.Point2f) (cv.Point2f, bool) {
	r := Sub(a2, a1)
	s := Sub(b2, b1)
	denom := Cross(r, s)
	if math.Abs(denom) < geom_cvEps {
		return cv.Point2f{}, false
	}
	t := Cross(Sub(b1, a1), s) / denom
	return Add(a1, Scale(r, t)), true
}

// SegmentDistance returns the shortest distance between the two closed segments
// p1p2 and p3p4. It is 0 when the segments intersect.
func SegmentDistance(p1, p2, p3, p4 cv.Point2f) float64 {
	if SegmentsIntersect(p1, p2, p3, p4) {
		return 0
	}
	d := PointToSegmentDistance(p1, p2, p3)
	d = math.Min(d, PointToSegmentDistance(p1, p2, p4))
	d = math.Min(d, PointToSegmentDistance(p3, p4, p1))
	d = math.Min(d, PointToSegmentDistance(p3, p4, p2))
	return d
}
