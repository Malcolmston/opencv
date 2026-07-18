package geom_cv_test

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
	geom "github.com/malcolmston/opencv/geom_cv"
)

func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func p(x, y float64) cv.Point2f { return cv.Point2f{X: x, Y: y} }

// square is the axis-aligned 4x4 square used by several tests.
func square() []cv.Point2f {
	return []cv.Point2f{p(0, 0), p(4, 0), p(4, 4), p(0, 4)}
}

func TestPrimitives(t *testing.T) {
	if got := geom.Dot(p(1, 2), p(3, 4)); got != 11 {
		t.Errorf("Dot = %v, want 11", got)
	}
	if got := geom.Cross(p(1, 0), p(0, 1)); got != 1 {
		t.Errorf("Cross = %v, want 1", got)
	}
	if got := geom.Distance(p(0, 0), p(3, 4)); got != 5 {
		t.Errorf("Distance = %v, want 5", got)
	}
	if got := geom.DistanceSquared(p(0, 0), p(3, 4)); got != 25 {
		t.Errorf("DistanceSquared = %v, want 25", got)
	}
	if got := geom.Orientation(p(0, 0), p(1, 0), p(1, 1)); got != 1 {
		t.Errorf("Orientation CCW = %v, want 1", got)
	}
	if got := geom.Orientation(p(0, 0), p(1, 0), p(1, -1)); got != -1 {
		t.Errorf("Orientation CW = %v, want -1", got)
	}
	if got := geom.Orientation(p(0, 0), p(1, 1), p(2, 2)); got != 0 {
		t.Errorf("Orientation collinear = %v, want 0", got)
	}
	if !geom.Collinear(p(0, 0), p(2, 2), p(5, 5)) {
		t.Error("Collinear should be true")
	}
	if got := geom.Midpoint(p(0, 0), p(4, 2)); got != p(2, 1) {
		t.Errorf("Midpoint = %v, want (2,1)", got)
	}
	if got := geom.Lerp(p(0, 0), p(10, 0), 0.3); !approx(got.X, 3, 1e-12) {
		t.Errorf("Lerp = %v, want x=3", got)
	}
	if got := geom.Norm(p(3, 4)); got != 5 {
		t.Errorf("Norm = %v, want 5", got)
	}
	nrm := geom.Normalize(p(0, 8))
	if !approx(nrm.Y, 1, 1e-12) || !approx(nrm.X, 0, 1e-12) {
		t.Errorf("Normalize = %v, want (0,1)", nrm)
	}
	if got := geom.AngleBetween(p(1, 0), p(0, 1)); !approx(got, math.Pi/2, 1e-9) {
		t.Errorf("AngleBetween = %v, want pi/2", got)
	}
	if got := geom.Perpendicular(p(1, 0)); got != p(0, 1) {
		t.Errorf("Perpendicular = %v, want (0,1)", got)
	}
}

func TestPolygonAreaCentroid(t *testing.T) {
	sq := square()
	if got := geom.PolygonArea(sq); !approx(got, 16, 1e-9) {
		t.Errorf("PolygonArea = %v, want 16", got)
	}
	if got := geom.SignedArea(sq); !approx(got, 16, 1e-9) {
		t.Errorf("SignedArea = %v, want 16 (CCW)", got)
	}
	if got := geom.PolygonPerimeter(sq); !approx(got, 16, 1e-9) {
		t.Errorf("PolygonPerimeter = %v, want 16", got)
	}
	c := geom.PolygonCentroid(sq)
	if !approx(c.X, 2, 1e-9) || !approx(c.Y, 2, 1e-9) {
		t.Errorf("PolygonCentroid = %v, want (2,2)", c)
	}
	// A right triangle's centroid is the mean of its corners.
	tri := []cv.Point2f{p(0, 0), p(6, 0), p(0, 3)}
	ct := geom.PolygonCentroid(tri)
	if !approx(ct.X, 2, 1e-9) || !approx(ct.Y, 1, 1e-9) {
		t.Errorf("triangle centroid = %v, want (2,1)", ct)
	}
	if got := geom.PolygonArea(tri); !approx(got, 9, 1e-9) {
		t.Errorf("triangle area = %v, want 9", got)
	}
}

func TestPointInPolygon(t *testing.T) {
	sq := square()
	if !geom.PointInPolygon(sq, p(2, 2)) {
		t.Error("(2,2) should be inside")
	}
	if geom.PointInPolygon(sq, p(5, 5)) {
		t.Error("(5,5) should be outside")
	}
	if geom.PointInPolygon(sq, p(-1, 2)) {
		t.Error("(-1,2) should be outside")
	}
	if geom.WindingNumber(sq, p(2, 2)) == 0 {
		t.Error("winding number should be nonzero inside")
	}
	if geom.WindingNumber(sq, p(9, 9)) != 0 {
		t.Error("winding number should be zero outside")
	}
	if !geom.PointOnPolygonBoundary(sq, p(4, 2)) {
		t.Error("(4,2) should be on boundary")
	}
	if geom.PointOnPolygonBoundary(sq, p(2, 2)) {
		t.Error("(2,2) should not be on boundary")
	}
}

func TestConvexPredicates(t *testing.T) {
	sq := square()
	if !geom.IsConvex(sq) {
		t.Error("square should be convex")
	}
	if geom.IsClockwise(sq) {
		t.Error("square is CCW (positive signed area)")
	}
	// Non-convex arrowhead.
	arrow := []cv.Point2f{p(0, 0), p(4, 2), p(0, 4), p(2, 2)}
	if geom.IsConvex(arrow) {
		t.Error("arrowhead should not be convex")
	}
	rev := geom.ReversePolygon(sq)
	if !geom.IsClockwise(rev) {
		t.Error("reversed square should be clockwise")
	}
	ccw := geom.EnsureCounterClockwise(rev)
	if geom.IsClockwise(ccw) {
		t.Error("EnsureCounterClockwise failed")
	}
}

func TestConvexHull(t *testing.T) {
	pts := []cv.Point2f{p(0, 0), p(4, 0), p(4, 4), p(0, 4), p(2, 2), p(1, 1), p(3, 1)}
	hull := geom.ConvexHull(pts)
	if len(hull) != 4 {
		t.Fatalf("hull vertices = %d, want 4", len(hull))
	}
	if !approx(geom.PolygonArea(hull), 16, 1e-9) {
		t.Errorf("hull area = %v, want 16", geom.PolygonArea(hull))
	}
	if !approx(geom.ConvexHullArea(pts), 16, 1e-9) {
		t.Errorf("ConvexHullArea = %v, want 16", geom.ConvexHullArea(pts))
	}
	if geom.IsClockwise(hull) {
		t.Error("ConvexHull must return CCW winding")
	}
}

func TestRotatingCalipers(t *testing.T) {
	sq := square()
	if got := geom.ConvexDiameter(sq); !approx(got, math.Sqrt(32), 1e-9) {
		t.Errorf("ConvexDiameter = %v, want %v", got, math.Sqrt(32))
	}
	if got := geom.ConvexWidth(sq); !approx(got, 4, 1e-9) {
		t.Errorf("ConvexWidth = %v, want 4", got)
	}
	// Cross-check diameter against brute force on an irregular set.
	pts := []cv.Point2f{p(0, 0), p(3, 0), p(5, 2), p(4, 5), p(1, 4), p(2, 2)}
	var brute float64
	for i := range pts {
		for j := i + 1; j < len(pts); j++ {
			if d := geom.Distance(pts[i], pts[j]); d > brute {
				brute = d
			}
		}
	}
	if got := geom.ConvexDiameter(pts); !approx(got, brute, 1e-9) {
		t.Errorf("ConvexDiameter = %v, want brute %v", got, brute)
	}
	if pairs := geom.AntipodalPairs(geom.ConvexHull(sq)); len(pairs) == 0 {
		t.Error("AntipodalPairs returned none for a square")
	}
}

func TestMinAreaRect(t *testing.T) {
	r := geom.MinAreaRect(square())
	area := r.Width * r.Height
	if !approx(area, 16, 1e-6) {
		t.Errorf("MinAreaRect area = %v, want 16", area)
	}
	if !approx(r.CenterX, 2, 1e-6) || !approx(r.CenterY, 2, 1e-6) {
		t.Errorf("MinAreaRect center = (%v,%v), want (2,2)", r.CenterX, r.CenterY)
	}
	// A diamond (rotated square) side sqrt(2)*... should still enclose tightly.
	diamond := []cv.Point2f{p(2, 0), p(4, 2), p(2, 4), p(0, 2)}
	rd := geom.MinAreaRect(diamond)
	if !approx(rd.Width*rd.Height, 8, 1e-6) {
		t.Errorf("diamond MinAreaRect area = %v, want 8", rd.Width*rd.Height)
	}
}

func TestMinEnclosing(t *testing.T) {
	c := geom.MinEnclosingCircle(square())
	if !approx(c.Center.X, 2, 1e-6) || !approx(c.Center.Y, 2, 1e-6) {
		t.Errorf("MinEnclosingCircle center = %v, want (2,2)", c.Center)
	}
	if !approx(c.Radius, math.Sqrt(8), 1e-6) {
		t.Errorf("MinEnclosingCircle radius = %v, want %v", c.Radius, math.Sqrt(8))
	}
	// Every input point must be enclosed.
	for _, q := range square() {
		if !c.Contains(q) {
			t.Errorf("circle does not contain %v", q)
		}
	}
	box := geom.MinEnclosingBox([]cv.Point2f{p(1, 2), p(5, -1), p(3, 4)})
	if box.Min != p(1, -1) || box.Max != p(5, 4) {
		t.Errorf("MinEnclosingBox = %v..%v, want (1,-1)..(5,4)", box.Min, box.Max)
	}
	if !approx(box.Width(), 4, 1e-12) || !approx(box.Height(), 5, 1e-12) {
		t.Errorf("box size = %vx%v, want 4x5", box.Width(), box.Height())
	}
}

func TestIntersection(t *testing.T) {
	pt, ok := geom.SegmentIntersection(p(0, 0), p(4, 4), p(0, 4), p(4, 0))
	if !ok || !approx(pt.X, 2, 1e-9) || !approx(pt.Y, 2, 1e-9) {
		t.Errorf("SegmentIntersection = %v ok=%v, want (2,2)", pt, ok)
	}
	if !geom.SegmentsIntersect(p(0, 0), p(4, 4), p(0, 4), p(4, 0)) {
		t.Error("segments should intersect")
	}
	if geom.SegmentsIntersect(p(0, 0), p(1, 1), p(2, 2), p(3, 3)) {
		// collinear but disjoint
		t.Error("disjoint collinear segments should not intersect")
	}
	_, ok = geom.SegmentIntersection(p(0, 0), p(1, 0), p(0, 1), p(1, 1))
	if ok {
		t.Error("parallel segments should not intersect")
	}
	lp, ok := geom.LineIntersection(p(0, 0), p(1, 1), p(0, 2), p(2, 0))
	if !ok || !approx(lp.X, 1, 1e-9) || !approx(lp.Y, 1, 1e-9) {
		t.Errorf("LineIntersection = %v ok=%v, want (1,1)", lp, ok)
	}
	if d := geom.PointToSegmentDistance(p(0, 0), p(4, 0), p(2, 3)); !approx(d, 3, 1e-9) {
		t.Errorf("PointToSegmentDistance = %v, want 3", d)
	}
	if d := geom.PointToSegmentDistance(p(0, 0), p(4, 0), p(-2, 0)); !approx(d, 2, 1e-9) {
		t.Errorf("PointToSegmentDistance clamped = %v, want 2", d)
	}
	if d := geom.PointToLineDistance(p(0, 0), p(4, 0), p(2, 3)); !approx(d, 3, 1e-9) {
		t.Errorf("PointToLineDistance = %v, want 3", d)
	}
	cp := geom.ClosestPointOnSegment(p(0, 0), p(4, 0), p(2, 5))
	if !approx(cp.X, 2, 1e-9) || !approx(cp.Y, 0, 1e-9) {
		t.Errorf("ClosestPointOnSegment = %v, want (2,0)", cp)
	}
	if d := geom.SegmentDistance(p(0, 0), p(1, 0), p(0, 3), p(1, 3)); !approx(d, 3, 1e-9) {
		t.Errorf("SegmentDistance = %v, want 3", d)
	}
}

func TestTypes(t *testing.T) {
	s := geom.Segment{A: p(0, 0), B: p(3, 4)}
	if !approx(s.Length(), 5, 1e-12) {
		t.Errorf("Segment.Length = %v, want 5", s.Length())
	}
	if m := s.Midpoint(); !approx(m.X, 1.5, 1e-12) {
		t.Errorf("Segment.Midpoint = %v", m)
	}
	circ := geom.Circle{Center: p(0, 0), Radius: 5}
	if !circ.Contains(p(3, 4)) {
		t.Error("circle should contain boundary point")
	}
	if !approx(circ.Area(), math.Pi*25, 1e-9) {
		t.Errorf("Circle.Area = %v", circ.Area())
	}
	tri := geom.Triangle{A: p(0, 0), B: p(4, 0), C: p(0, 4)}
	if !approx(tri.Area(), 8, 1e-9) {
		t.Errorf("Triangle.Area = %v, want 8", tri.Area())
	}
	if !tri.Contains(p(1, 1)) {
		t.Error("triangle should contain (1,1)")
	}
	if tri.Contains(p(3, 3)) {
		t.Error("triangle should not contain (3,3)")
	}
	cen := tri.Centroid()
	if !approx(cen.X, 4.0/3, 1e-9) || !approx(cen.Y, 4.0/3, 1e-9) {
		t.Errorf("Triangle.Centroid = %v", cen)
	}
	l := geom.LineThrough(p(0, 0), p(2, 2))
	if q := l.PointAt(0.5); !approx(q.X, 1, 1e-12) || !approx(q.Y, 1, 1e-12) {
		t.Errorf("Line.PointAt = %v, want (1,1)", q)
	}
	bb := geom.BoundingBox{Min: p(0, 0), Max: p(4, 2)}
	if !approx(bb.Area(), 8, 1e-12) {
		t.Errorf("BoundingBox.Area = %v", bb.Area())
	}
	if !bb.Contains(p(1, 1)) || bb.Contains(p(5, 1)) {
		t.Error("BoundingBox.Contains wrong")
	}
	if len(bb.Corners()) != 4 {
		t.Error("BoundingBox.Corners should have 4 points")
	}
	r := bb.Rect()
	if r.Width != 4 || r.Height != 2 {
		t.Errorf("BoundingBox.Rect = %+v", r)
	}
}

func TestCircumcircle(t *testing.T) {
	// Right triangle: circumcenter is the hypotenuse midpoint.
	circ, ok := geom.Circumcircle(p(0, 0), p(4, 0), p(4, 4))
	if !ok || !approx(circ.Center.X, 2, 1e-9) || !approx(circ.Center.Y, 2, 1e-9) {
		t.Errorf("Circumcircle center = %v ok=%v, want (2,2)", circ.Center, ok)
	}
	if !approx(circ.Radius, math.Sqrt(8), 1e-9) {
		t.Errorf("Circumcircle radius = %v, want %v", circ.Radius, math.Sqrt(8))
	}
	c, ok := geom.Circumcenter(p(0, 0), p(2, 0), p(0, 2))
	if !ok || !approx(c.X, 1, 1e-9) || !approx(c.Y, 1, 1e-9) {
		t.Errorf("Circumcenter = %v, want (1,1)", c)
	}
	if _, ok := geom.Circumcenter(p(0, 0), p(1, 1), p(2, 2)); ok {
		t.Error("collinear points must have no circumcenter")
	}
	if !geom.InCircumcircle(p(0, 0), p(4, 0), p(4, 4), p(2, 2)) {
		t.Error("(2,2) should be inside circumcircle")
	}
	if geom.InCircumcircle(p(0, 0), p(4, 0), p(4, 4), p(20, 20)) {
		t.Error("(20,20) should be outside circumcircle")
	}
}

func TestDelaunay(t *testing.T) {
	tris := geom.DelaunayTriangulation(square())
	if len(tris) != 2 {
		t.Fatalf("Delaunay of square = %d triangles, want 2", len(tris))
	}
	var total float64
	for _, tr := range tris {
		total += tr.Area()
	}
	if !approx(total, 16, 1e-9) {
		t.Errorf("total triangle area = %v, want 16", total)
	}
	// A denser deterministic grid should triangulate the whole bounding area.
	grid := []cv.Point2f{p(0, 0), p(2, 0), p(4, 0), p(0, 2), p(2, 2), p(4, 2), p(0, 4), p(2, 4), p(4, 4)}
	gt := geom.DelaunayTriangulation(grid)
	var gArea float64
	for _, tr := range gt {
		gArea += tr.Area()
	}
	if !approx(gArea, 16, 1e-6) {
		t.Errorf("grid triangulation area = %v, want 16", gArea)
	}
	if len(geom.DelaunayTriangulation([]cv.Point2f{p(0, 0), p(1, 1)})) != 0 {
		t.Error("fewer than 3 points must yield no triangles")
	}
	if len(geom.DelaunayTriangulation([]cv.Point2f{p(0, 0), p(1, 1), p(2, 2)})) != 0 {
		t.Error("collinear points must yield no triangles")
	}
}

func TestVoronoi(t *testing.T) {
	// Two triangles sharing edge (0,0)-(4,0): exactly one bounded Voronoi edge.
	sites := []cv.Point2f{p(0, 0), p(4, 0), p(2, 3), p(2, -3)}
	edges := geom.VoronoiEdges(sites)
	if len(edges) != 1 {
		t.Fatalf("VoronoiEdges = %d, want 1", len(edges))
	}
	e := edges[0]
	if !approx(e.A.X, 2, 1e-9) || !approx(e.B.X, 2, 1e-9) {
		t.Errorf("Voronoi edge x should be 2: %v", e)
	}
	if !approx(math.Abs(e.A.Y), 5.0/6, 1e-9) || !approx(math.Abs(e.B.Y), 5.0/6, 1e-9) {
		t.Errorf("Voronoi edge y should be +-5/6: %v", e)
	}

	// Two symmetric sites split a box into two equal-area cells.
	box := geom.BoundingBox{Min: p(0, 0), Max: p(4, 4)}
	cells := geom.VoronoiCells([]cv.Point2f{p(1, 2), p(3, 2)}, box)
	if len(cells) != 2 {
		t.Fatalf("VoronoiCells = %d, want 2", len(cells))
	}
	for i, c := range cells {
		if a := geom.PolygonArea(c.Vertices); !approx(a, 8, 1e-6) {
			t.Errorf("cell %d area = %v, want 8", i, a)
		}
	}

	sitesN := []cv.Point2f{p(0, 0), p(10, 10), p(0, 10)}
	if geom.NearestSite(sitesN, p(1, 1)) != 0 {
		t.Error("nearest to (1,1) should be site 0")
	}
	if geom.NearestSite(sitesN, p(9, 9)) != 1 {
		t.Error("nearest to (9,9) should be site 1")
	}
	if geom.NearestSite(nil, p(0, 0)) != -1 {
		t.Error("nearest of empty should be -1")
	}
}

func TestClipping(t *testing.T) {
	subject := square()
	clip := []cv.Point2f{p(2, 2), p(6, 2), p(6, 6), p(2, 6)}
	out := geom.ClipPolygon(subject, clip)
	if a := geom.PolygonArea(out); !approx(a, 4, 1e-9) {
		t.Errorf("clip area = %v, want 4", a)
	}
	if a := geom.PolygonIntersectionArea(subject, clip); !approx(a, 4, 1e-9) {
		t.Errorf("PolygonIntersectionArea = %v, want 4", a)
	}
	// Disjoint polygons clip to nothing.
	far := []cv.Point2f{p(20, 20), p(24, 20), p(24, 24), p(20, 24)}
	if a := geom.PolygonIntersectionArea(subject, far); !approx(a, 0, 1e-9) {
		t.Errorf("disjoint clip area = %v, want 0", a)
	}

	box := geom.BoundingBox{Min: p(0, 0), Max: p(4, 4)}
	seg, ok := geom.ClipSegmentToBox(p(-1, 2), p(5, 2), box)
	if !ok || !approx(seg.A.X, 0, 1e-9) || !approx(seg.B.X, 4, 1e-9) {
		t.Errorf("ClipSegmentToBox = %v ok=%v, want (0,2)-(4,2)", seg, ok)
	}
	if _, ok := geom.ClipSegmentToBox(p(-5, -5), p(-1, -1), box); ok {
		t.Error("segment outside box should be rejected")
	}
}

func TestAlphaShape(t *testing.T) {
	sq := square()
	// Large alpha keeps all Delaunay triangles -> boundary is the 4 square sides.
	edges := geom.AlphaShapeEdges(sq, 10)
	if len(edges) != 4 {
		t.Fatalf("AlphaShapeEdges (large alpha) = %d, want 4", len(edges))
	}
	var perim float64
	for _, e := range edges {
		perim += e.Length()
	}
	if !approx(perim, 16, 1e-9) {
		t.Errorf("alpha shape perimeter = %v, want 16", perim)
	}
	// Alpha too small to admit any triangle (circumradius ~2.83).
	if len(geom.AlphaShapeEdges(sq, 1)) != 0 {
		t.Error("tiny alpha should yield no edges")
	}
	if len(geom.AlphaComplexTriangles(sq, 10)) != 2 {
		t.Error("large-alpha complex should equal full Delaunay (2 triangles)")
	}
	if len(geom.AlphaComplexTriangles(sq, 0)) != 0 {
		t.Error("non-positive alpha should yield no triangles")
	}
}

func TestRaster(t *testing.T) {
	// Rectangle covering pixel centers with x in {1,2,3,4}, y in {1,2,3} => 12.
	poly := []cv.Point2f{p(1, 1), p(5, 1), p(5, 4), p(1, 4)}
	mask := geom.PolygonMask(6, 6, poly)
	if mask.Rows != 6 || mask.Cols != 6 || mask.Channels != 1 {
		t.Fatalf("mask dims = %dx%dx%d", mask.Rows, mask.Cols, mask.Channels)
	}
	count := 0
	for y := 0; y < mask.Rows; y++ {
		for x := 0; x < mask.Cols; x++ {
			if mask.At(y, x, 0) == 255 {
				count++
			}
		}
	}
	if count != 12 {
		t.Errorf("filled pixels = %d, want 12", count)
	}
	// Interior sample is filled, exterior is not.
	if mask.At(2, 2, 0) != 255 {
		t.Error("interior pixel (2,2) should be filled")
	}
	if mask.At(0, 0, 0) != 0 {
		t.Error("exterior pixel (0,0) should be empty")
	}

	img := cv.NewMat(6, 6, 1)
	geom.FillPolygon(img, poly, 200)
	if img.At(2, 2, 0) != 200 {
		t.Error("FillPolygon should write value 200 inside")
	}

	out := cv.NewMat(8, 8, 1)
	geom.DrawPolygonOutline(out, []cv.Point2f{p(1, 1), p(6, 1), p(6, 6), p(1, 6)}, 255)
	if out.At(1, 1, 0) != 255 || out.At(1, 3, 0) != 255 {
		t.Error("DrawPolygonOutline should draw top edge")
	}
	if out.At(3, 3, 0) != 0 {
		t.Error("DrawPolygonOutline interior should be untouched")
	}
}

func TestConversions(t *testing.T) {
	if got := geom.ToPoint(p(2.6, 3.4)); got.X != 3 || got.Y != 3 {
		t.Errorf("ToPoint = %+v, want (3,3)", got)
	}
	if got := geom.ToPoint2f(cv.Point{X: 2, Y: 5}); got.X != 2 || got.Y != 5 {
		t.Errorf("ToPoint2f = %v", got)
	}
	if got := geom.VectorAngle(p(0, 1)); !approx(got, math.Pi/2, 1e-9) {
		t.Errorf("VectorAngle = %v, want pi/2", got)
	}
}

// geom_cvBenchPoints builds a deterministic pseudo-random point cloud for the
// benchmark.
func geom_cvBenchPoints(n int) []cv.Point2f {
	pts := make([]cv.Point2f, n)
	seed := uint64(0x9e3779b97f4a7c15)
	for i := range pts {
		seed ^= seed << 13
		seed ^= seed >> 7
		seed ^= seed << 17
		x := float64(seed%10000) / 10.0
		seed ^= seed << 13
		seed ^= seed >> 7
		seed ^= seed << 17
		y := float64(seed%10000) / 10.0
		pts[i] = cv.Point2f{X: x, Y: y}
	}
	return pts
}

// BenchmarkDelaunayTriangulation exercises the heaviest routine in the package.
func BenchmarkDelaunayTriangulation(b *testing.B) {
	pts := geom_cvBenchPoints(400)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = geom.DelaunayTriangulation(pts)
	}
}
