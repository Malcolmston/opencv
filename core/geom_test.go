package core

import (
	"math"
	"testing"
)

func TestConvexHull2f(t *testing.T) {
	pts := []Point2f{
		{0, 0}, {1, 0}, {2, 0}, {2, 2}, {0, 2}, {1, 1},
	}
	hull := ConvexHull2f(pts)
	if len(hull) != 4 {
		t.Fatalf("hull size = %d, want 4 (%v)", len(hull), hull)
	}
	// Area of the 2x2 square hull.
	if math.Abs(PolygonArea2f(hull)-4) > 1e-6 {
		t.Errorf("hull area = %v, want 4", PolygonArea2f(hull))
	}
}

func TestPolygonAreaAndPerimeter(t *testing.T) {
	sq := []Point2f{{0, 0}, {3, 0}, {3, 3}, {0, 3}}
	if math.Abs(PolygonArea2f(sq)-9) > 1e-6 {
		t.Errorf("area = %v", PolygonArea2f(sq))
	}
	if math.Abs(Perimeter2f(sq, true)-12) > 1e-6 {
		t.Errorf("perimeter = %v", Perimeter2f(sq, true))
	}
}

func TestBoundingRectAndCentroid(t *testing.T) {
	pts := []Point2f{{1, 2}, {5, 2}, {3, 8}}
	r := BoundingRect2f(pts)
	if r.X != 1 || r.Y != 2 || r.Width != 4 || r.Height != 6 {
		t.Errorf("bounding = %v", r)
	}
	c := Centroid2f(pts)
	if math.Abs(float64(c.X)-3) > 1e-6 || math.Abs(float64(c.Y)-4) > 1e-6 {
		t.Errorf("centroid = %v", c)
	}
}

func TestSegmentsIntersect(t *testing.T) {
	p, ok := SegmentsIntersect2f(Pt2f(0, 0), Pt2f(2, 2), Pt2f(0, 2), Pt2f(2, 0))
	if !ok {
		t.Fatal("segments should intersect")
	}
	if math.Abs(float64(p.X)-1) > 1e-6 || math.Abs(float64(p.Y)-1) > 1e-6 {
		t.Errorf("intersection = %v", p)
	}
	if _, ok := SegmentsIntersect2f(Pt2f(0, 0), Pt2f(1, 1), Pt2f(2, 2), Pt2f(3, 3)); ok {
		t.Error("parallel/collinear should not report crossing")
	}
}

func TestPointInTriangleAndDist(t *testing.T) {
	a, b, c := Pt2f(0, 0), Pt2f(4, 0), Pt2f(0, 4)
	if !PointInTriangle2f(Pt2f(1, 1), a, b, c) {
		t.Error("point should be inside triangle")
	}
	if PointInTriangle2f(Pt2f(3, 3), a, b, c) {
		t.Error("point should be outside triangle")
	}
	if d := PointToSegmentDist2f(Pt2f(0, 2), Pt2f(0, 0), Pt2f(4, 0)); math.Abs(d-2) > 1e-6 {
		t.Errorf("dist = %v, want 2", d)
	}
}

func TestIsConvex(t *testing.T) {
	sq := []Point2f{{0, 0}, {2, 0}, {2, 2}, {0, 2}}
	if !IsConvex2f(sq) {
		t.Error("square should be convex")
	}
	arrow := []Point2f{{0, 0}, {2, 0}, {1, 1}, {2, 2}, {0, 2}}
	if IsConvex2f(arrow) {
		t.Error("arrow should be non-convex")
	}
}
