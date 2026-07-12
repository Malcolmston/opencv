package cv

import (
	"math"
	"testing"
)

func TestFindContoursSquareCountAndArea(t *testing.T) {
	m := synthSquare(40, 10, 10, 20)
	contours, hierarchy := FindContours(m, RetrExternal, ChainApproxNone)
	if len(contours) != 1 {
		t.Fatalf("expected 1 external contour, got %d", len(contours))
	}
	if len(hierarchy) != 1 {
		t.Fatalf("hierarchy length = %d, want 1", len(hierarchy))
	}
	// A 20x20 filled block traces a border polygon of area (20-1)^2 = 361.
	area := ContourArea(contours[0])
	if math.Abs(area-361) > 0.5 {
		t.Errorf("ContourArea = %v, want 361", area)
	}
	r := BoundingRect(contours[0])
	if r.X != 10 || r.Y != 10 || r.Width != 20 || r.Height != 20 {
		t.Errorf("BoundingRect = %+v, want {10 10 20 20}", r)
	}
}

func TestFindContoursTwoBlobs(t *testing.T) {
	m := NewMat(40, 60, 1)
	// Two separated filled squares.
	for y := 5; y < 15; y++ {
		for x := 5; x < 15; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	for y := 20; y < 30; y++ {
		for x := 40; x < 50; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	contours, _ := FindContours(m, RetrExternal, ChainApproxSimple)
	if len(contours) != 2 {
		t.Fatalf("expected 2 blobs, got %d", len(contours))
	}
	for _, c := range contours {
		a := ContourArea(c)
		if math.Abs(a-81) > 0.5 { // (10-1)^2
			t.Errorf("blob area = %v, want 81", a)
		}
	}
}

func TestFindContoursSimpleApproxRectCorners(t *testing.T) {
	m := synthSquare(40, 10, 10, 20)
	contours, _ := FindContours(m, RetrExternal, ChainApproxSimple)
	if len(contours) != 1 {
		t.Fatalf("got %d contours", len(contours))
	}
	// A rectangle collapses to its four corners under CHAIN_APPROX_SIMPLE.
	if len(contours[0]) != 4 {
		t.Errorf("simple approx of rectangle kept %d points, want 4", len(contours[0]))
	}
}

func TestFindContoursTreeHoleHierarchy(t *testing.T) {
	// Filled square with a square hole -> outer border + hole border.
	m := NewMat(40, 40, 1)
	for y := 5; y < 35; y++ {
		for x := 5; x < 35; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	for y := 15; y < 25; y++ {
		for x := 15; x < 25; x++ {
			m.Set(y, x, 0, 0)
		}
	}
	contours, hierarchy := FindContours(m, RetrTree, ChainApproxSimple)
	if len(contours) != 2 {
		t.Fatalf("expected outer + hole = 2 contours, got %d", len(contours))
	}
	// Exactly one contour is a child (the hole), one is a root.
	roots, children := 0, 0
	for _, h := range hierarchy {
		if h.Parent == -1 {
			roots++
		} else {
			children++
		}
	}
	if roots != 1 || children != 1 {
		t.Errorf("hierarchy roots=%d children=%d, want 1/1", roots, children)
	}
	// External retrieval drops the hole.
	ext, _ := FindContours(m, RetrExternal, ChainApproxSimple)
	if len(ext) != 1 {
		t.Errorf("external of holed square = %d, want 1", len(ext))
	}
}

func TestArcLength(t *testing.T) {
	sq := []Point{{0, 0}, {4, 0}, {4, 3}, {0, 3}}
	// Perimeter closed = 4 + 3 + 4 + 3 = 14.
	if l := ArcLength(sq, true); math.Abs(l-14) > 1e-9 {
		t.Errorf("closed ArcLength = %v, want 14", l)
	}
	// Open = 4 + 3 + 4 = 11.
	if l := ArcLength(sq, false); math.Abs(l-11) > 1e-9 {
		t.Errorf("open ArcLength = %v, want 11", l)
	}
}

func TestConvexHullSquareWithInterior(t *testing.T) {
	pts := []Point{{0, 0}, {4, 0}, {4, 4}, {0, 4}, {2, 2}, {1, 1}, {3, 1}}
	hull := ConvexHull(pts)
	if len(hull) != 4 {
		t.Fatalf("hull has %d points, want 4", len(hull))
	}
	corners := map[Point]bool{{0, 0}: false, {4, 0}: false, {4, 4}: false, {0, 4}: false}
	for _, p := range hull {
		if _, ok := corners[p]; ok {
			corners[p] = true
		} else {
			t.Errorf("hull contains non-corner %v", p)
		}
	}
	for c, seen := range corners {
		if !seen {
			t.Errorf("hull missing corner %v", c)
		}
	}
}

func TestMinAreaRectSquare(t *testing.T) {
	pts := []Point{{0, 0}, {10, 0}, {10, 10}, {0, 10}}
	r := MinAreaRect(pts)
	if math.Abs(r.Width-10) > 1e-6 || math.Abs(r.Height-10) > 1e-6 {
		t.Errorf("MinAreaRect size = %vx%v, want 10x10", r.Width, r.Height)
	}
	if math.Abs(r.CenterX-5) > 1e-6 || math.Abs(r.CenterY-5) > 1e-6 {
		t.Errorf("MinAreaRect center = (%v,%v), want (5,5)", r.CenterX, r.CenterY)
	}
}

func TestApproxPolyDPOpenLine(t *testing.T) {
	line := []Point{{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}}
	out := ApproxPolyDP(line, 0.1, false)
	if len(out) != 2 || out[0] != (Point{0, 0}) || out[1] != (Point{4, 0}) {
		t.Errorf("ApproxPolyDP collinear = %v, want endpoints only", out)
	}
}

func TestApproxPolyDPTriangleBump(t *testing.T) {
	// A path with a clear apex should keep the apex.
	pts := []Point{{0, 0}, {2, 0}, {4, 0}, {4, 4}, {2, 8}, {0, 4}}
	out := ApproxPolyDP(pts, 1.0, false)
	if len(out) < 3 {
		t.Errorf("ApproxPolyDP dropped the shape entirely: %v", out)
	}
}

func TestImageMomentsCentroidCenteredSquare(t *testing.T) {
	m := NewMat(11, 11, 1)
	for y := 3; y <= 7; y++ {
		for x := 3; x <= 7; x++ {
			m.Set(y, x, 0, 255)
		}
	}
	mo := ImageMoments(m)
	cx, cy := mo.Centroid()
	if math.Abs(cx-5) > 1e-9 || math.Abs(cy-5) > 1e-9 {
		t.Errorf("centroid = (%v,%v), want (5,5)", cx, cy)
	}
	// M00 = sum of weights = 25 pixels * 255.
	if mo.M00 != 25*255 {
		t.Errorf("M00 = %v, want %v", mo.M00, 25*255)
	}
}

func TestDrawContoursRoundTrip(t *testing.T) {
	m := synthSquare(40, 10, 10, 20)
	contours, _ := FindContours(m, RetrExternal, ChainApproxNone)
	canvas := NewMat(40, 40, 1)
	DrawContours(canvas, contours, -1, NewScalar(255), 1)
	// The drawn outline should cover the square's top-left corner pixel.
	if canvas.At(10, 10, 0) != 255 {
		t.Error("DrawContours did not draw the contour outline")
	}
}
