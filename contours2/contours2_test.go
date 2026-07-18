package contours2_test

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/contours2"
)

// buildMat builds a single-channel Mat from a rectangular pattern of 0/1 rows.
// A '1' becomes 255 (foreground), a ' ' or '0' becomes 0 (background).
func buildMat(rows []string) *cv.Mat {
	h := len(rows)
	w := len(rows[0])
	m := cv.NewMat(h, w, 1)
	for y, row := range rows {
		for x := 0; x < w; x++ {
			if x < len(row) && (row[x] == '1' || row[x] == '#') {
				m.Data[y*w+x] = 255
			}
		}
	}
	return m
}

func approxEq(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// --- Border following ------------------------------------------------------

func TestFindContoursSquareSimple(t *testing.T) {
	// 5x5 solid square inside a 9x9 image, occupying x,y in [2,6].
	m := buildMat([]string{
		"000000000",
		"000000000",
		"001111100",
		"001111100",
		"001111100",
		"001111100",
		"001111100",
		"000000000",
		"000000000",
	})
	cs, hier := contours2.FindContours(m, contours2.RetrExternal, contours2.ChainApproxSimple)
	if len(cs) != 1 {
		t.Fatalf("expected 1 contour, got %d", len(cs))
	}
	if len(hier) != 1 || hier[0].Parent != -1 {
		t.Fatalf("unexpected hierarchy %+v", hier)
	}
	if len(cs[0]) != 4 {
		t.Fatalf("ChainApproxSimple should give 4 corners, got %d: %v", len(cs[0]), cs[0])
	}
	// contourArea of a 5x5 pixel square boundary (4x4 through centres) is 16.
	if a := contours2.ContourArea(cs[0]); !approxEq(a, 16, 1e-9) {
		t.Fatalf("expected area 16, got %v", a)
	}
}

func TestFindContoursNoneCountsPerimeter(t *testing.T) {
	m := buildMat([]string{
		"00000",
		"01110",
		"01110",
		"01110",
		"00000",
	})
	cs := contours2.FindExternalContours(m, contours2.ChainApproxNone)
	if len(cs) != 1 {
		t.Fatalf("expected 1 contour, got %d", len(cs))
	}
	// A 3x3 solid square traced fully has 8 boundary pixels.
	if len(cs[0]) != 8 {
		t.Fatalf("expected 8 boundary points, got %d: %v", len(cs[0]), cs[0])
	}
}

func TestFindContoursHierarchyTree(t *testing.T) {
	// Outer 7x7 filled square with a hole in the middle.
	m := buildMat([]string{
		"000000000",
		"011111110",
		"011111110",
		"011000110",
		"011000110",
		"011000110",
		"011111110",
		"011111110",
		"000000000",
	})
	cs, hier := contours2.FindContours(m, contours2.RetrTree, contours2.ChainApproxSimple)
	if len(cs) != 2 {
		t.Fatalf("expected 2 contours (outer+hole), got %d", len(cs))
	}
	// Exactly one contour is a root and one is its child.
	roots, children := 0, 0
	for i, h := range hier {
		if h.Parent == -1 {
			roots++
			if h.FirstChild < 0 {
				t.Fatalf("root contour %d has no child", i)
			}
		} else {
			children++
		}
	}
	if roots != 1 || children != 1 {
		t.Fatalf("expected 1 root + 1 child, got roots=%d children=%d hier=%+v", roots, children, hier)
	}
}

func TestFindContoursTwoObjects(t *testing.T) {
	m := buildMat([]string{
		"0000000000",
		"0110000110",
		"0110000110",
		"0000000000",
	})
	cs := contours2.FindExternalContours(m, contours2.ChainApproxSimple)
	if len(cs) != 2 {
		t.Fatalf("expected 2 objects, got %d", len(cs))
	}
}

// --- Area / perimeter / moments -------------------------------------------

func squarePoly() []cv.Point {
	return []cv.Point{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10}}
}

func TestContourAreaAndArcLength(t *testing.T) {
	sq := squarePoly()
	if a := contours2.ContourArea(sq); !approxEq(a, 100, 1e-9) {
		t.Fatalf("expected area 100, got %v", a)
	}
	if p := contours2.ArcLength(sq, true); !approxEq(p, 40, 1e-9) {
		t.Fatalf("expected perimeter 40, got %v", p)
	}
	if p := contours2.ArcLength(sq, false); !approxEq(p, 30, 1e-9) {
		t.Fatalf("expected open length 30, got %v", p)
	}
	// Signed area sign follows winding.
	if s := contours2.ContourAreaSigned(sq); s <= 0 {
		t.Fatalf("expected positive signed area for clockwise-in-image square, got %v", s)
	}
}

func TestContourMoments(t *testing.T) {
	m := contours2.ContourMoments(squarePoly())
	if !approxEq(m.M00, 100, 1e-9) {
		t.Fatalf("expected M00=100, got %v", m.M00)
	}
	c := m.Centroid()
	if !approxEq(c.X, 5, 1e-9) || !approxEq(c.Y, 5, 1e-9) {
		t.Fatalf("expected centroid (5,5), got (%v,%v)", c.X, c.Y)
	}
	// For a square of side 10, mu20 = mu02 = a^4/12 = 10000/12.
	if !approxEq(m.Mu20, 10000.0/12, 1e-6) {
		t.Fatalf("expected Mu20=%v, got %v", 10000.0/12, m.Mu20)
	}
	if !approxEq(m.Mu11, 0, 1e-6) {
		t.Fatalf("expected Mu11=0, got %v", m.Mu11)
	}
}

func TestMatchShapesInvariance(t *testing.T) {
	small := []cv.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 4}, {X: 0, Y: 4}}
	// Same square scaled x3 and translated: Hu moments should match closely.
	big := []cv.Point{{X: 10, Y: 10}, {X: 22, Y: 10}, {X: 22, Y: 22}, {X: 10, Y: 22}}
	ma := contours2.ContourMoments(small)
	mb := contours2.ContourMoments(big)
	d := contours2.MatchShapes(ma, mb, contours2.ContoursMatchI1)
	if d > 1e-6 {
		t.Fatalf("expected near-zero shape distance for similar squares, got %v", d)
	}
	// A very different shape (thin rectangle) should score higher.
	thin := []cv.Point{{X: 0, Y: 0}, {X: 20, Y: 0}, {X: 20, Y: 2}, {X: 0, Y: 2}}
	mc := contours2.ContourMoments(thin)
	if d2 := contours2.MatchShapes(ma, mc, contours2.ContoursMatchI1); d2 <= d {
		t.Fatalf("expected thin rectangle to differ more, got %v vs %v", d2, d)
	}
}

func TestHuMomentsMethod(t *testing.T) {
	m := contours2.ContourMoments(squarePoly())
	if m.HuMoments() != contours2.HuMoments(m) {
		t.Fatalf("method and function forms disagree")
	}
}

// --- Convexity -------------------------------------------------------------

func TestConvexHull(t *testing.T) {
	pts := []cv.Point{
		{X: 0, Y: 0}, {X: 5, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10},
		{X: 5, Y: 5}, // interior point, must be dropped
		{X: 0, Y: 10},
	}
	hull := contours2.ConvexHull(pts, false)
	if len(hull) != 4 {
		t.Fatalf("expected 4 hull vertices, got %d: %v", len(hull), hull)
	}
	for _, h := range hull {
		if h.X == 5 && h.Y == 5 {
			t.Fatalf("interior point should not be on hull")
		}
	}
	if a := contours2.ContourArea(hull); !approxEq(a, 100, 1e-9) {
		t.Fatalf("expected hull area 100, got %v", a)
	}
}

func TestIsContourConvex(t *testing.T) {
	if !contours2.IsContourConvex(squarePoly()) {
		t.Fatalf("square should be convex")
	}
	notch := []cv.Point{
		{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10},
		{X: 5, Y: 5}, // concave notch
		{X: 0, Y: 10},
	}
	if contours2.IsContourConvex(notch) {
		t.Fatalf("notched polygon should not be convex")
	}
}

func TestConvexityDefects(t *testing.T) {
	// Arrowhead-like contour with a single concavity at the bottom middle.
	contour := []cv.Point{
		{X: 0, Y: 0},
		{X: 10, Y: 0},
		{X: 10, Y: 10},
		{X: 5, Y: 5}, // dent inward
		{X: 0, Y: 10},
	}
	defects := contours2.ConvexityDefects(contour, nil)
	if len(defects) != 1 {
		t.Fatalf("expected 1 defect, got %d: %+v", len(defects), defects)
	}
	if contour[defects[0].FarthestPointIndex] != (cv.Point{X: 5, Y: 5}) {
		t.Fatalf("expected farthest point (5,5), got %v", contour[defects[0].FarthestPointIndex])
	}
	if defects[0].Depth <= 0 {
		t.Fatalf("expected positive depth, got %v", defects[0].Depth)
	}
}

// --- Enclosing shapes ------------------------------------------------------

func TestMinAreaRect(t *testing.T) {
	pts := []cv.Point{{X: 0, Y: 0}, {X: 20, Y: 0}, {X: 20, Y: 10}, {X: 0, Y: 10}}
	r := contours2.MinAreaRect(pts)
	if !approxEq(r.CenterX, 10, 1e-6) || !approxEq(r.CenterY, 5, 1e-6) {
		t.Fatalf("expected centre (10,5), got (%v,%v)", r.CenterX, r.CenterY)
	}
	long := math.Max(r.Width, r.Height)
	short := math.Min(r.Width, r.Height)
	if !approxEq(long, 20, 1e-6) || !approxEq(short, 10, 1e-6) {
		t.Fatalf("expected sides 20x10, got %vx%v", long, short)
	}
	if !approxEq(r.Area(), 200, 1e-6) {
		t.Fatalf("expected area 200, got %v", r.Area())
	}
}

func TestMinEnclosingCircle(t *testing.T) {
	pts := []cv.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 4}, {X: 0, Y: 4}}
	c := contours2.MinEnclosingCircle(pts)
	if !approxEq(c.Center.X, 2, 1e-6) || !approxEq(c.Center.Y, 2, 1e-6) {
		t.Fatalf("expected centre (2,2), got %v", c.Center)
	}
	if !approxEq(c.Radius, math.Sqrt(8), 1e-6) {
		t.Fatalf("expected radius %v, got %v", math.Sqrt(8), c.Radius)
	}
	for _, p := range pts {
		if !c.Contains(cv.Point2f{X: float64(p.X), Y: float64(p.Y)}) {
			t.Fatalf("point %v not enclosed", p)
		}
	}
}

func TestBoundingRect(t *testing.T) {
	pts := []cv.Point{{X: 3, Y: 4}, {X: 10, Y: 4}, {X: 10, Y: 12}, {X: 3, Y: 12}}
	r := contours2.BoundingRect(pts)
	want := contours2.Rect{X: 3, Y: 4, Width: 8, Height: 9}
	if r != want {
		t.Fatalf("expected %+v, got %+v", want, r)
	}
	if !r.Contains(cv.Point{X: 5, Y: 6}) {
		t.Fatalf("expected point inside rect")
	}
	if r.Contains(cv.Point{X: 11, Y: 13}) {
		t.Fatalf("expected point outside rect")
	}
}

func TestFitLine(t *testing.T) {
	// Points on the line y = 2x.
	pts := []cv.Point{{X: 0, Y: 0}, {X: 1, Y: 2}, {X: 2, Y: 4}, {X: 3, Y: 6}, {X: 4, Y: 8}}
	vx, vy, x0, y0 := contours2.FitLine(pts)
	// Direction should be parallel to (1,2): slope vy/vx == 2.
	if math.Abs(vx) < 1e-9 {
		t.Fatalf("degenerate direction")
	}
	if !approxEq(vy/vx, 2, 1e-6) {
		t.Fatalf("expected slope 2, got %v", vy/vx)
	}
	// The centroid (2,4) lies on the line.
	if !approxEq(x0, 2, 1e-9) || !approxEq(y0, 4, 1e-9) {
		t.Fatalf("expected point (2,4), got (%v,%v)", x0, y0)
	}
}

func TestFitEllipseCircle(t *testing.T) {
	// Sample 40 points on a circle of radius 10 centred at (15,20).
	var pts []cv.Point
	for k := 0; k < 40; k++ {
		a := 2 * math.Pi * float64(k) / 40
		pts = append(pts, cv.Point{
			X: int(math.Round(15 + 10*math.Cos(a))),
			Y: int(math.Round(20 + 10*math.Sin(a))),
		})
	}
	e := contours2.FitEllipse(pts)
	if !approxEq(e.Center.X, 15, 0.6) || !approxEq(e.Center.Y, 20, 0.6) {
		t.Fatalf("expected centre near (15,20), got %v", e.Center)
	}
	// A circle should have near-equal axes.
	ratio := e.Width / e.Height
	if ratio < 0.9 || ratio > 1.1 {
		t.Fatalf("expected near-circular axes, got ratio %v (%v x %v)", ratio, e.Width, e.Height)
	}
}

// --- Point / polygon and descriptors --------------------------------------

func TestPointPolygonTest(t *testing.T) {
	sq := squarePoly()
	if contours2.PointPolygonTest(sq, cv.Point{X: 5, Y: 5}, false) != 1 {
		t.Fatalf("expected (5,5) inside")
	}
	if contours2.PointPolygonTest(sq, cv.Point{X: 20, Y: 5}, false) != -1 {
		t.Fatalf("expected (20,5) outside")
	}
	if contours2.PointPolygonTest(sq, cv.Point{X: 0, Y: 5}, false) != 0 {
		t.Fatalf("expected (0,5) on edge")
	}
	if d := contours2.PointPolygonTest(sq, cv.Point{X: 3, Y: 5}, true); !approxEq(d, 3, 1e-9) {
		t.Fatalf("expected inside distance 3, got %v", d)
	}
	if d := contours2.PointPolygonTest(sq, cv.Point{X: -2, Y: 5}, true); !approxEq(d, -2, 1e-9) {
		t.Fatalf("expected outside distance -2, got %v", d)
	}
}

func TestApproxPolyDP(t *testing.T) {
	curve := []cv.Point{
		{X: 0, Y: 0}, {X: 5, Y: 0}, {X: 10, Y: 0},
		{X: 10, Y: 5}, {X: 10, Y: 10},
		{X: 5, Y: 10}, {X: 0, Y: 10}, {X: 0, Y: 5},
	}
	out := contours2.ApproxPolyDP(curve, 1.0, true)
	if len(out) != 4 {
		t.Fatalf("expected 4 corners, got %d: %v", len(out), out)
	}
}

func TestApproxPolyDPOpen(t *testing.T) {
	// A straight-ish polyline with one bump collapses to endpoints when the
	// bump is within epsilon, else keeps the bump.
	curve := []cv.Point{{X: 0, Y: 0}, {X: 5, Y: 1}, {X: 10, Y: 0}}
	if out := contours2.ApproxPolyDP(curve, 2.0, false); len(out) != 2 {
		t.Fatalf("expected 2 points with large epsilon, got %d", len(out))
	}
	if out := contours2.ApproxPolyDP(curve, 0.5, false); len(out) != 3 {
		t.Fatalf("expected 3 points with small epsilon, got %d", len(out))
	}
}

func TestDescriptors(t *testing.T) {
	sq := squarePoly()
	if ar := contours2.AspectRatio(sq); !approxEq(ar, 1, 1e-9) {
		t.Fatalf("expected aspect ratio 1, got %v", ar)
	}
	// Extent: contour area 100 vs bounding box 11x11=121.
	if ex := contours2.Extent(sq); !approxEq(ex, 100.0/121, 1e-9) {
		t.Fatalf("expected extent %v, got %v", 100.0/121, ex)
	}
	if s := contours2.Solidity(sq); !approxEq(s, 1, 1e-9) {
		t.Fatalf("expected solidity 1 for convex square, got %v", s)
	}
	if d := contours2.EquivalentDiameter(sq); !approxEq(d, math.Sqrt(400/math.Pi), 1e-9) {
		t.Fatalf("expected equivalent diameter %v, got %v", math.Sqrt(400/math.Pi), d)
	}
	// A wide rectangle is oriented horizontally (~0 degrees) and eccentric.
	wide := []cv.Point{{X: 0, Y: 0}, {X: 40, Y: 0}, {X: 40, Y: 4}, {X: 0, Y: 4}}
	if o := contours2.Orientation(wide); math.Abs(o) > 5 {
		t.Fatalf("expected near-horizontal orientation, got %v", o)
	}
	if ec := contours2.Eccentricity(wide); ec < 0.9 {
		t.Fatalf("expected high eccentricity for thin rectangle, got %v", ec)
	}
	if ec := contours2.Eccentricity(sq); ec > 0.1 {
		t.Fatalf("expected low eccentricity for square, got %v", ec)
	}
}

func TestExtremePoints(t *testing.T) {
	pts := []cv.Point{{X: 5, Y: 0}, {X: 10, Y: 5}, {X: 5, Y: 10}, {X: 0, Y: 5}}
	l, r, top, bottom := contours2.ExtremePoints(pts)
	if l != (cv.Point{X: 0, Y: 5}) || r != (cv.Point{X: 10, Y: 5}) ||
		top != (cv.Point{X: 5, Y: 0}) || bottom != (cv.Point{X: 5, Y: 10}) {
		t.Fatalf("unexpected extremes l=%v r=%v t=%v b=%v", l, r, top, bottom)
	}
}

func TestRotatedRectPoints(t *testing.T) {
	r := contours2.RotatedRect{CenterX: 0, CenterY: 0, Width: 4, Height: 2, Angle: 0}
	pts := r.Points()
	// Corners at (+-2, +-1).
	seen := map[cv.Point]bool{}
	for _, p := range pts {
		seen[p] = true
	}
	for _, want := range []cv.Point{{X: -2, Y: -1}, {X: 2, Y: -1}, {X: 2, Y: 1}, {X: -2, Y: 1}} {
		if !seen[want] {
			t.Fatalf("missing corner %v in %v", want, pts)
		}
	}
	bb := r.BoundingRect()
	if bb.Width != 4 || bb.Height != 2 {
		t.Fatalf("expected bbox 4x2, got %+v", bb)
	}
}

func TestImageMoments(t *testing.T) {
	// 3x3 solid square of value 255; centroid at (2,2) if placed at [1..3].
	m := buildMat([]string{
		"00000",
		"01110",
		"01110",
		"01110",
		"00000",
	})
	mo := contours2.ImageMoments(m)
	c := mo.Centroid()
	if !approxEq(c.X, 2, 1e-9) || !approxEq(c.Y, 2, 1e-9) {
		t.Fatalf("expected centroid (2,2), got %v", c)
	}
	if !approxEq(mo.M00, 9*255, 1e-6) {
		t.Fatalf("expected mass %v, got %v", 9*255.0, mo.M00)
	}
}

// --- Benchmark (heaviest routine) -----------------------------------------

func BenchmarkFindContours(b *testing.B) {
	// A 200x200 image tiled with small squares to exercise many borders.
	const n = 200
	m := cv.NewMat(n, n, 1)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			if (x/6)%2 == 0 && (y/6)%2 == 0 {
				m.Data[y*n+x] = 255
			}
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cs, _ := contours2.FindContours(m, contours2.RetrTree, contours2.ChainApproxSimple)
		if len(cs) == 0 {
			b.Fatal("no contours")
		}
	}
}
