package shapefit_test

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
	sf "github.com/malcolmston/opencv/shapefit"
)

func pt(x, y float64) cv.Point2f { return cv.Point2f{X: x, Y: y} }

func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// ---- least-squares line ----

func TestFitLine(t *testing.T) {
	// Points on y = 2x + 1.
	var pts []cv.Point2f
	for x := 0.0; x <= 10; x++ {
		pts = append(pts, pt(x, 2*x+1))
	}
	l, err := sf.FitLine(pts)
	if err != nil {
		t.Fatalf("FitLine: %v", err)
	}
	if rmse := sf.LineRMSE(l, pts); rmse > 1e-9 {
		t.Errorf("RMSE = %g, want ~0", rmse)
	}
	// Perpendicular distance of (0,2) to 2x - y + 1 = 0 is 1/sqrt(5).
	if d := l.Distance(pt(0, 2)); !approx(d, 1/math.Sqrt(5), 1e-9) {
		t.Errorf("Distance = %g, want %g", d, 1/math.Sqrt(5))
	}
	// Direction angle should be atan(2).
	if !approx(l.Angle(), math.Atan(2), 1e-9) {
		t.Errorf("Angle = %g, want %g", l.Angle(), math.Atan(2))
	}
}

func TestLineThroughPoints(t *testing.T) {
	l := sf.LineThroughPoints(pt(0, 0), pt(0, 5)) // vertical line x = 0
	if d := l.Distance(pt(3, 2)); !approx(d, 3, 1e-9) {
		t.Errorf("Distance = %g, want 3", d)
	}
	cp := l.ClosestPoint(pt(3, 2))
	if !approx(cp.X, 0, 1e-9) || !approx(cp.Y, 2, 1e-9) {
		t.Errorf("ClosestPoint = %+v, want (0,2)", cp)
	}
}

// ---- least-squares circle ----

func circlePoints(cx, cy, r float64, n int) []cv.Point2f {
	pts := make([]cv.Point2f, n)
	for i := 0; i < n; i++ {
		a := 2 * math.Pi * float64(i) / float64(n)
		pts[i] = pt(cx+r*math.Cos(a), cy+r*math.Sin(a))
	}
	return pts
}

func TestFitCircleKasa(t *testing.T) {
	pts := circlePoints(5, 7, 3, 24)
	c, err := sf.FitCircle(pts)
	if err != nil {
		t.Fatalf("FitCircle: %v", err)
	}
	if !approx(c.Center.X, 5, 1e-6) || !approx(c.Center.Y, 7, 1e-6) || !approx(c.Radius, 3, 1e-6) {
		t.Errorf("circle = %+v, want center (5,7) r 3", c)
	}
}

func TestFitCircleTaubin(t *testing.T) {
	pts := circlePoints(12, 9, 5, 30)
	c, err := sf.FitCircleTaubin(pts)
	if err != nil {
		t.Fatalf("FitCircleTaubin: %v", err)
	}
	if !approx(c.Center.X, 12, 1e-5) || !approx(c.Center.Y, 9, 1e-5) || !approx(c.Radius, 5, 1e-5) {
		t.Errorf("circle = %+v, want center (12,9) r 5", c)
	}
}

func TestCircleThrough(t *testing.T) {
	// Circle center (0,0) r 1 through three known points.
	c, ok := sf.CircleThrough(pt(1, 0), pt(0, 1), pt(-1, 0))
	if !ok {
		t.Fatal("CircleThrough: collinear reported")
	}
	if !approx(c.Center.X, 0, 1e-9) || !approx(c.Center.Y, 0, 1e-9) || !approx(c.Radius, 1, 1e-9) {
		t.Errorf("circle = %+v, want center (0,0) r 1", c)
	}
	if _, ok := sf.CircleThrough(pt(0, 0), pt(1, 1), pt(2, 2)); ok {
		t.Error("CircleThrough: collinear points should fail")
	}
}

// ---- least-squares ellipse ----

func TestFitEllipseAxisAligned(t *testing.T) {
	want := sf.Ellipse{Center: pt(10, 8), SemiMajor: 6, SemiMinor: 3, Angle: 0}
	var pts []cv.Point2f
	for i := 0; i < 40; i++ {
		pts = append(pts, want.PointAt(2*math.Pi*float64(i)/40))
	}
	got, err := sf.FitEllipse(pts)
	if err != nil {
		t.Fatalf("FitEllipse: %v", err)
	}
	if !approx(got.Center.X, 10, 1e-3) || !approx(got.Center.Y, 8, 1e-3) {
		t.Errorf("center = %+v, want (10,8)", got.Center)
	}
	if !approx(got.SemiMajor, 6, 1e-3) || !approx(got.SemiMinor, 3, 1e-3) {
		t.Errorf("axes = (%g,%g), want (6,3)", got.SemiMajor, got.SemiMinor)
	}
	if !approx(got.Angle, 0, 1e-3) {
		t.Errorf("angle = %g, want 0", got.Angle)
	}
}

func TestFitEllipseRotated(t *testing.T) {
	want := sf.Ellipse{Center: pt(20, 15), SemiMajor: 8, SemiMinor: 4, Angle: math.Pi / 6}
	var pts []cv.Point2f
	for i := 0; i < 60; i++ {
		pts = append(pts, want.PointAt(2*math.Pi*float64(i)/60))
	}
	got, err := sf.FitEllipse(pts)
	if err != nil {
		t.Fatalf("FitEllipse: %v", err)
	}
	if !approx(got.Center.X, 20, 1e-2) || !approx(got.Center.Y, 15, 1e-2) {
		t.Errorf("center = %+v, want (20,15)", got.Center)
	}
	if !approx(got.SemiMajor, 8, 1e-2) || !approx(got.SemiMinor, 4, 1e-2) {
		t.Errorf("axes = (%g,%g), want (8,4)", got.SemiMajor, got.SemiMinor)
	}
	if !approx(got.Angle, math.Pi/6, 1e-2) {
		t.Errorf("angle = %g, want %g", got.Angle, math.Pi/6)
	}
}

func TestEllipseConicRoundTrip(t *testing.T) {
	e := sf.Ellipse{Center: pt(3, -2), SemiMajor: 5, SemiMinor: 2, Angle: math.Pi / 5}
	A, B, C, D, E, F := e.Conic()
	back, ok := sf.EllipseFromConic(A, B, C, D, E, F)
	if !ok {
		t.Fatal("EllipseFromConic: not an ellipse")
	}
	if !approx(back.Center.X, 3, 1e-6) || !approx(back.Center.Y, -2, 1e-6) {
		t.Errorf("center = %+v, want (3,-2)", back.Center)
	}
	if !approx(back.SemiMajor, 5, 1e-6) || !approx(back.SemiMinor, 2, 1e-6) {
		t.Errorf("axes = (%g,%g), want (5,2)", back.SemiMajor, back.SemiMinor)
	}
	if !approx(back.Angle, math.Pi/5, 1e-6) {
		t.Errorf("angle = %g, want %g", back.Angle, math.Pi/5)
	}
}

func TestEllipseRotatedRectRoundTrip(t *testing.T) {
	e := sf.Ellipse{Center: pt(1, 1), SemiMajor: 4, SemiMinor: 2, Angle: math.Pi / 4}
	rr := e.RotatedRect()
	if !approx(rr.Width, 8, 1e-9) || !approx(rr.Height, 4, 1e-9) {
		t.Errorf("rr = %+v, want W8 H4", rr)
	}
	back := sf.EllipseFromRotatedRect(rr)
	if !approx(back.SemiMajor, 4, 1e-9) || !approx(back.SemiMinor, 2, 1e-9) {
		t.Errorf("axes = (%g,%g), want (4,2)", back.SemiMajor, back.SemiMinor)
	}
}

// ---- RANSAC ----

func TestRANSACLineOutliers(t *testing.T) {
	var pts []cv.Point2f
	for x := 0.0; x <= 20; x++ {
		pts = append(pts, pt(x, 3)) // horizontal line y = 3
	}
	// Outliers far from the line.
	pts = append(pts, pt(5, 40), pt(10, -25), pt(15, 60))
	p := sf.DefaultRANSACParams()
	p.Threshold = 1
	l, inliers, err := sf.RANSACLine(pts, p)
	if err != nil {
		t.Fatalf("RANSACLine: %v", err)
	}
	if len(inliers) != 21 {
		t.Errorf("inliers = %d, want 21", len(inliers))
	}
	if d := l.Distance(pt(8, 3)); d > 1e-6 {
		t.Errorf("fitted line off: distance %g", d)
	}
}

func TestRANSACCircleOutliers(t *testing.T) {
	pts := circlePoints(15, 15, 6, 40)
	pts = append(pts, pt(1, 1), pt(29, 3), pt(2, 28))
	p := sf.DefaultRANSACParams()
	p.Threshold = 0.5
	c, inliers, err := sf.RANSACCircle(pts, p)
	if err != nil {
		t.Fatalf("RANSACCircle: %v", err)
	}
	if len(inliers) != 40 {
		t.Errorf("inliers = %d, want 40", len(inliers))
	}
	if !approx(c.Center.X, 15, 1e-3) || !approx(c.Center.Y, 15, 1e-3) || !approx(c.Radius, 6, 1e-3) {
		t.Errorf("circle = %+v, want center (15,15) r 6", c)
	}
}

func TestRANSACDeterministic(t *testing.T) {
	pts := circlePoints(10, 10, 4, 30)
	p := sf.DefaultRANSACParams()
	c1, _, _ := sf.RANSACCircle(pts, p)
	c2, _, _ := sf.RANSACCircle(pts, p)
	if c1 != c2 {
		t.Errorf("nondeterministic: %+v vs %+v", c1, c2)
	}
}

func TestDetectLines(t *testing.T) {
	var pts []cv.Point2f
	for x := 0.0; x <= 20; x++ {
		pts = append(pts, pt(x, 0)) // line 1: y = 0
	}
	for y := 0.0; y <= 20; y++ {
		pts = append(pts, pt(0, y)) // line 2: x = 0
	}
	p := sf.DefaultRANSACParams()
	p.Threshold = 0.5
	lines := sf.DetectLines(pts, p, 2)
	if len(lines) != 2 {
		t.Fatalf("lines = %d, want 2", len(lines))
	}
}

// ---- Hough ----

func TestHoughLinesVertical(t *testing.T) {
	m := cv.NewMat(30, 30, 1)
	for y := 0; y < 30; y++ {
		m.Set(y, 12, 0, 255) // vertical line x = 12
	}
	lines := sf.HoughLines(m, 1, math.Pi/180, 15)
	if len(lines) == 0 {
		t.Fatal("no lines found")
	}
	best := lines[0]
	if !approx(best.Theta, 0, 1e-6) || !approx(best.Rho, 12, 1.0) {
		t.Errorf("line theta=%g rho=%g, want theta 0 rho 12", best.Theta, best.Rho)
	}
}

func TestHoughLinesHorizontal(t *testing.T) {
	m := cv.NewMat(30, 30, 1)
	for x := 0; x < 30; x++ {
		m.Set(9, x, 0, 255) // horizontal line y = 9
	}
	lines := sf.HoughLines(m, 1, math.Pi/180, 15)
	if len(lines) == 0 {
		t.Fatal("no lines found")
	}
	best := lines[0]
	if !approx(best.Theta, math.Pi/2, 1e-2) || !approx(best.Rho, 9, 1.0) {
		t.Errorf("line theta=%g rho=%g, want theta pi/2 rho 9", best.Theta, best.Rho)
	}
}

func TestHoughLinesP(t *testing.T) {
	m := cv.NewMat(30, 40, 1)
	for x := 5; x <= 25; x++ {
		m.Set(10, x, 0, 255) // segment y=10, x in [5,25]
	}
	segs := sf.HoughLinesP(m, 1, math.Pi/180, 15, 10, 2)
	if len(segs) == 0 {
		t.Fatal("no segments found")
	}
	if l := segs[0].Length(); !approx(l, 20, 2) {
		t.Errorf("segment length = %g, want ~20", l)
	}
}

func TestHoughCircles(t *testing.T) {
	m := cv.NewMat(40, 40, 1)
	cx, cy, r := 20, 20, 8
	for s := 0; s < 120; s++ {
		a := 2 * math.Pi * float64(s) / 120
		x := int(math.Round(float64(cx) + float64(r)*math.Cos(a)))
		y := int(math.Round(float64(cy) + float64(r)*math.Sin(a)))
		if x >= 0 && x < 40 && y >= 0 && y < 40 {
			m.Set(y, x, 0, 255)
		}
	}
	circles := sf.HoughCircles(m, 6, 10, 20, 5)
	if len(circles) == 0 {
		t.Fatal("no circles found")
	}
	best := circles[0]
	if !approx(best.Center.X, 20, 2) || !approx(best.Center.Y, 20, 2) || !approx(best.Radius, 8, 1) {
		t.Errorf("circle = %+v, want center (20,20) r 8", best)
	}
}

// ---- generalized Hough ----

func squareOutline(x0, y0, side float64) []cv.Point2f {
	var pts []cv.Point2f
	for i := 0.0; i <= side; i++ {
		pts = append(pts, pt(x0+i, y0))
		pts = append(pts, pt(x0+i, y0+side))
		pts = append(pts, pt(x0, y0+i))
		pts = append(pts, pt(x0+side, y0+i))
	}
	return pts
}

func TestGeneralizedHough(t *testing.T) {
	tmpl := squareOutline(0, 0, 10)
	ref := pt(5, 5)
	g := sf.NewGeneralizedHough(tmpl, ref, 36, 3)
	if g == nil {
		t.Fatal("NewGeneralizedHough returned nil")
	}
	// Translate the template by (25, 20) into a 70x70 field.
	var query []cv.Point2f
	for _, p := range tmpl {
		query = append(query, pt(p.X+25, p.Y+20))
	}
	matches := g.Detect(query, 70, 70, len(tmpl)/2)
	if len(matches) == 0 {
		t.Fatal("no matches")
	}
	loc := matches[0].Location
	if !approx(loc.X, 30, 2) || !approx(loc.Y, 25, 2) {
		t.Errorf("match at %+v, want ~(30,25)", loc)
	}
}

// ---- ellipse detection ----

func TestDetectEllipses(t *testing.T) {
	e := sf.Ellipse{Center: pt(25, 20), SemiMajor: 12, SemiMinor: 7, Angle: 0}
	var pts []cv.Point2f
	for i := 0; i < 48; i++ {
		p := e.PointAt(2 * math.Pi * float64(i) / 48)
		pts = append(pts, pt(math.Round(p.X), math.Round(p.Y)))
	}
	dets := sf.DetectEllipses(pts, 8, 5, 20, 5)
	if len(dets) == 0 {
		t.Fatal("no ellipses detected")
	}
	d := dets[0]
	if !approx(d.Ellipse.Center.X, 25, 2) || !approx(d.Ellipse.Center.Y, 20, 2) {
		t.Errorf("center = %+v, want ~(25,20)", d.Ellipse.Center)
	}
	if !approx(d.Ellipse.SemiMajor, 12, 2) {
		t.Errorf("semi-major = %g, want ~12", d.Ellipse.SemiMajor)
	}
}

// ---- rectangle detection ----

func TestDetectRectangles(t *testing.T) {
	corners := []cv.Point2f{pt(0, 0), pt(10, 0), pt(10, 6), pt(0, 6)}
	rects := sf.DetectRectangles(corners, 1)
	if len(rects) != 1 {
		t.Fatalf("rects = %d, want 1", len(rects))
	}
	r := rects[0]
	if !approx(r.Center.X, 5, 1e-6) || !approx(r.Center.Y, 3, 1e-6) {
		t.Errorf("center = %+v, want (5,3)", r.Center)
	}
	rr := r.RotatedRect()
	if !approx(rr.Width, 10, 1e-6) || !approx(rr.Height, 6, 1e-6) {
		t.Errorf("rotatedRect = %+v, want W10 H6", rr)
	}
}

func TestDetectRectanglesNoFalsePositive(t *testing.T) {
	// A triangle plus a stray point: no rectangle.
	corners := []cv.Point2f{pt(0, 0), pt(10, 0), pt(5, 8), pt(3, 3)}
	rects := sf.DetectRectangles(corners, 1)
	if len(rects) != 0 {
		t.Errorf("rects = %d, want 0", len(rects))
	}
}

// ---- symmetry ----

func TestDetectReflectionSymmetry(t *testing.T) {
	// Symmetric only about the vertical axis x = 5.
	pts := []cv.Point2f{pt(2, 0), pt(8, 0), pt(3, 5), pt(7, 5), pt(5, 9)}
	axis, ok := sf.DetectReflectionSymmetry(pts, 180, 0.1)
	if !ok {
		t.Fatal("DetectReflectionSymmetry failed")
	}
	if !approx(axis.Score, 1, 1e-9) {
		t.Errorf("score = %g, want 1", axis.Score)
	}
	if !approx(math.Abs(axis.Angle), math.Pi/2, 1e-6) {
		t.Errorf("angle = %g, want +/- pi/2", axis.Angle)
	}
}

func TestDetectRotationalSymmetry(t *testing.T) {
	pts := []cv.Point2f{pt(0, 0), pt(4, 0), pt(4, 4), pt(0, 4)}
	fold, score := sf.DetectRotationalSymmetry(pts, 6, 0.3)
	if fold != 4 {
		t.Errorf("fold = %d, want 4", fold)
	}
	if !approx(score, 1, 1e-9) {
		t.Errorf("score = %g, want 1", score)
	}
}

// ---- primitive selection ----

func TestFitBestPrimitiveLine(t *testing.T) {
	var pts []cv.Point2f
	for x := 0.0; x <= 20; x++ {
		pts = append(pts, pt(x, 2))
	}
	p := sf.DefaultRANSACParams()
	p.Threshold = 0.5
	prim := sf.FitBestPrimitive(pts, p)
	if prim.Kind != sf.PrimitiveLine {
		t.Errorf("kind = %v, want line", prim.Kind)
	}
}

func TestFitBestPrimitiveCircle(t *testing.T) {
	pts := circlePoints(10, 10, 5, 40)
	p := sf.DefaultRANSACParams()
	p.Threshold = 0.5
	prim := sf.FitBestPrimitive(pts, p)
	if prim.Kind != sf.PrimitiveCircle {
		t.Errorf("kind = %v (%d inliers), want circle", prim.Kind, prim.Inliers)
	}
}

func TestPrimitiveKindString(t *testing.T) {
	if sf.PrimitiveEllipse.String() != "ellipse" || sf.PrimitiveNone.String() != "none" {
		t.Error("PrimitiveKind.String mismatch")
	}
}

// ---- point helpers ----

func TestPointsFromMatRoundTrip(t *testing.T) {
	m := cv.NewMat(10, 10, 1)
	m.Set(2, 3, 0, 255)
	m.Set(7, 8, 0, 200)
	pts := sf.PointsFromMat(m, 0)
	if len(pts) != 2 {
		t.Fatalf("points = %d, want 2", len(pts))
	}
	// Scan order: (3,2) comes before (8,7).
	if pts[0] != (cv.Point2f{X: 3, Y: 2}) {
		t.Errorf("pts[0] = %+v, want (3,2)", pts[0])
	}
	m2 := sf.PointsToMat(pts, 10, 10)
	if m2.At(2, 3, 0) != 255 || m2.At(7, 8, 0) != 255 {
		t.Error("PointsToMat did not set expected pixels")
	}
}

func TestCentroidAndBox(t *testing.T) {
	pts := []cv.Point2f{pt(0, 0), pt(4, 0), pt(4, 2), pt(0, 2)}
	c := sf.Centroid(pts)
	if !approx(c.X, 2, 1e-9) || !approx(c.Y, 1, 1e-9) {
		t.Errorf("centroid = %+v, want (2,1)", c)
	}
	mn, mx := sf.BoundingBox(pts)
	if mn != (cv.Point2f{X: 0, Y: 0}) || mx != (cv.Point2f{X: 4, Y: 2}) {
		t.Errorf("box = %+v..%+v, want (0,0)..(4,2)", mn, mx)
	}
}

// ---- benchmark: heaviest routine ----

func BenchmarkDetectEllipses(b *testing.B) {
	e := sf.Ellipse{Center: pt(30, 25), SemiMajor: 15, SemiMinor: 9, Angle: math.Pi / 8}
	var pts []cv.Point2f
	for i := 0; i < 60; i++ {
		p := e.PointAt(2 * math.Pi * float64(i) / 60)
		pts = append(pts, pt(math.Round(p.X), math.Round(p.Y)))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sf.DetectEllipses(pts, 10, 5, 25, 5)
	}
}
