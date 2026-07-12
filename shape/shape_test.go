package shape_test

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/shape"
)

func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// --- MinEnclosingCircle ---------------------------------------------------

func TestMinEnclosingCircleSquare(t *testing.T) {
	// Four corners of a 2×2 square centred at (1,1). The minimal circle passes
	// through all four corners: centre (1,1), radius = half the diagonal = √2.
	pts := []cv.Point{{X: 0, Y: 0}, {X: 2, Y: 0}, {X: 2, Y: 2}, {X: 0, Y: 2}}
	cx, cy, r := shape.MinEnclosingCircle(pts)
	if !approx(cx, 1, 1e-6) || !approx(cy, 1, 1e-6) {
		t.Fatalf("centre = (%.6f, %.6f), want (1,1)", cx, cy)
	}
	if !approx(r, math.Sqrt2, 1e-6) {
		t.Fatalf("radius = %.6f, want %.6f", r, math.Sqrt2)
	}
}

func TestMinEnclosingCircleEnclosesAll(t *testing.T) {
	pts := []cv.Point{
		{X: 3, Y: 1}, {X: 9, Y: 2}, {X: 5, Y: 7}, {X: 1, Y: 4},
		{X: 6, Y: 6}, {X: 8, Y: 5}, {X: 2, Y: 2}, {X: 4, Y: 3},
	}
	cx, cy, r := shape.MinEnclosingCircle(pts)
	for _, p := range pts {
		d := math.Hypot(float64(p.X)-cx, float64(p.Y)-cy)
		if d > r+1e-6 {
			t.Fatalf("point %v outside circle: dist %.6f > r %.6f", p, d, r)
		}
	}
	// Determinism: same input, same output.
	cx2, cy2, r2 := shape.MinEnclosingCircle(pts)
	if cx != cx2 || cy != cy2 || r != r2 {
		t.Fatalf("non-deterministic result")
	}
}

func TestMinEnclosingCircleSinglePoint(t *testing.T) {
	cx, cy, r := shape.MinEnclosingCircle([]cv.Point{{X: 5, Y: 7}})
	if cx != 5 || cy != 7 || r != 0 {
		t.Fatalf("got (%.1f,%.1f,%.1f), want (5,7,0)", cx, cy, r)
	}
}

// --- MinEnclosingTriangle -------------------------------------------------

func TestMinEnclosingTriangleExact(t *testing.T) {
	// Input hull is already a triangle; the enclosing triangle is itself.
	pts := []cv.Point{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 0, Y: 8}, {X: 2, Y: 2}}
	area, _ := shape.MinEnclosingTriangle(pts)
	if !approx(area, 40, 1e-6) {
		t.Fatalf("area = %.6f, want 40", area)
	}
}

func TestMinEnclosingTriangleSquare(t *testing.T) {
	// The minimal triangle enclosing a square has twice the square's area.
	pts := []cv.Point{{X: 0, Y: 0}, {X: 6, Y: 0}, {X: 6, Y: 6}, {X: 0, Y: 6}}
	area, tri := shape.MinEnclosingTriangle(pts)
	if !approx(area, 72, 1.0) {
		t.Fatalf("area = %.4f, want ~72", area)
	}
	// The triangle must enclose every input point.
	for _, p := range pts {
		if !pointInTriangle(float64(p.X), float64(p.Y), tri) {
			t.Fatalf("point %v not enclosed by triangle %v", p, tri)
		}
	}
}

func pointInTriangle(px, py float64, tri [3][2]float64) bool {
	sign := func(ax, ay, bx, by, cx, cy float64) float64 {
		return (ax-cx)*(by-cy) - (bx-cx)*(ay-cy)
	}
	d1 := sign(px, py, tri[0][0], tri[0][1], tri[1][0], tri[1][1])
	d2 := sign(px, py, tri[1][0], tri[1][1], tri[2][0], tri[2][1])
	d3 := sign(px, py, tri[2][0], tri[2][1], tri[0][0], tri[0][1])
	neg := d1 < -1e-6 || d2 < -1e-6 || d3 < -1e-6
	pos := d1 > 1e-6 || d2 > 1e-6 || d3 > 1e-6
	return !(neg && pos)
}

// --- FitLine --------------------------------------------------------------

func TestFitLinePerfect(t *testing.T) {
	pts := []cv.Point{{X: 0, Y: 0}, {X: 1, Y: 1}, {X: 2, Y: 2}, {X: 3, Y: 3}}
	vx, vy, x0, y0 := shape.FitLine(pts)
	inv := 1 / math.Sqrt2
	if !approx(vx, inv, 1e-9) || !approx(vy, inv, 1e-9) {
		t.Fatalf("direction (%.6f,%.6f), want (%.6f,%.6f)", vx, vy, inv, inv)
	}
	if !approx(x0, 1.5, 1e-9) || !approx(y0, 1.5, 1e-9) {
		t.Fatalf("point (%.6f,%.6f), want (1.5,1.5)", x0, y0)
	}
}

func TestFitLineNoisyRecoversDirection(t *testing.T) {
	// Points along y = 2x with a small alternating ±1 perturbation. The total
	// least-squares fit should recover the direction (1,2)/√5.
	var pts []cv.Point
	for tt := -25; tt <= 25; tt++ {
		noise := 1
		if tt%2 == 0 {
			noise = -1
		}
		pts = append(pts, cv.Point{X: tt, Y: 2*tt + noise})
	}
	vx, vy, _, _ := shape.FitLine(pts)
	wx, wy := 1/math.Sqrt(5), 2/math.Sqrt(5)
	if !approx(vx, wx, 0.02) || !approx(vy, wy, 0.02) {
		t.Fatalf("direction (%.4f,%.4f), want (%.4f,%.4f)", vx, vy, wx, wy)
	}
	// Direction is a unit vector.
	if !approx(math.Hypot(vx, vy), 1, 1e-9) {
		t.Fatalf("direction not unit length: %.6f", math.Hypot(vx, vy))
	}
}

// --- FitEllipse -----------------------------------------------------------

func TestFitEllipseKnown(t *testing.T) {
	// Sample points from an ellipse: centre (200,150), semi-axes a=100 (major),
	// b=60 (minor), rotated 30°. FitEllipse should recover them within tolerance
	// (points are rounded to integer pixels).
	const (
		cx, cy = 200.0, 150.0
		a, b   = 100.0, 60.0
		phi    = 30.0 * math.Pi / 180
	)
	ca, sa := math.Cos(phi), math.Sin(phi)
	var pts []cv.Point
	const n = 72
	for i := 0; i < n; i++ {
		th := 2 * math.Pi * float64(i) / n
		ex := a * math.Cos(th)
		ey := b * math.Sin(th)
		x := cx + ex*ca - ey*sa
		y := cy + ex*sa + ey*ca
		pts = append(pts, cv.Point{X: int(math.Round(x)), Y: int(math.Round(y))})
	}
	rr := shape.FitEllipse(pts)
	if !approx(rr.CenterX, cx, 2) || !approx(rr.CenterY, cy, 2) {
		t.Fatalf("centre (%.2f,%.2f), want (%.1f,%.1f)", rr.CenterX, rr.CenterY, cx, cy)
	}
	if !approx(rr.Width, 2*a, 4) {
		t.Fatalf("width %.2f, want %.1f", rr.Width, 2*a)
	}
	if !approx(rr.Height, 2*b, 4) {
		t.Fatalf("height %.2f, want %.1f", rr.Height, 2*b)
	}
	// Angle mod 180 should match 30°.
	ang := math.Mod(rr.Angle+180, 180)
	if !approx(ang, 30, 3) {
		t.Fatalf("angle %.2f (mod180 %.2f), want ~30", rr.Angle, ang)
	}
}

func TestFitEllipseTooFewPoints(t *testing.T) {
	rr := shape.FitEllipse([]cv.Point{{X: 0, Y: 0}, {X: 1, Y: 1}})
	if rr != (cv.RotatedRect{}) {
		t.Fatalf("expected zero RotatedRect for <5 points, got %+v", rr)
	}
}

// --- HuMoments / MatchShapes ----------------------------------------------

// pentagon is an asymmetric polygon used for moment-invariance tests.
var pentagon = []cv.Point{
	{X: 0, Y: 0}, {X: 40, Y: 10}, {X: 50, Y: 40}, {X: 20, Y: 60}, {X: -10, Y: 30},
}

func rotateScaleTranslate(pts []cv.Point, deg, scale, tx, ty float64) []cv.Point {
	rad := deg * math.Pi / 180
	c, s := math.Cos(rad), math.Sin(rad)
	out := make([]cv.Point, len(pts))
	for i, p := range pts {
		x := float64(p.X) * scale
		y := float64(p.Y) * scale
		rx := x*c - y*s + tx
		ry := x*s + y*c + ty
		out[i] = cv.Point{X: int(math.Round(rx)), Y: int(math.Round(ry))}
	}
	return out
}

func TestHuMomentsInvarianceExact(t *testing.T) {
	// A 90° rotation, 2× scale and integer translation map integer vertices to
	// integer vertices exactly, so Hu moments must match to high precision.
	orig := shape.HuMoments(shape.ContourMoments(pentagon))
	copyPts := rotateScaleTranslate(pentagon, 90, 2, 37, -13)
	got := shape.HuMoments(shape.ContourMoments(copyPts))
	for i := 0; i < 7; i++ {
		if math.Abs(orig[i]) < 1e-12 {
			if math.Abs(got[i]) > 1e-9 {
				t.Fatalf("hu[%d]: got %.3e, want ~0", i, got[i])
			}
			continue
		}
		rel := math.Abs(orig[i]-got[i]) / math.Abs(orig[i])
		if rel > 1e-6 {
			t.Fatalf("hu[%d]: orig %.6e got %.6e rel %.2e", i, orig[i], got[i], rel)
		}
	}
}

func TestHuMomentsInvarianceRotated(t *testing.T) {
	// A generic 37° rotation with scale introduces rounding; compare the
	// log-magnitude Hu signature within a looser tolerance.
	big := rotateScaleTranslate(pentagon, 0, 5, 0, 0) // enlarge to cut rounding
	orig := shape.HuMoments(shape.ContourMoments(big))
	copyPts := rotateScaleTranslate(big, 37, 1.3, 120, 200)
	got := shape.HuMoments(shape.ContourMoments(copyPts))
	for i := 0; i < 5; i++ { // first five Hu moments are the robust ones
		lo := math.Log10(math.Abs(orig[i]))
		lg := math.Log10(math.Abs(got[i]))
		if math.Abs(lo-lg) > 0.05 {
			t.Fatalf("hu[%d]: log|orig| %.4f log|got| %.4f", i, lo, lg)
		}
	}
}

func TestMatchShapesIdentical(t *testing.T) {
	for _, m := range []int{shape.ContoursMatchI1, shape.ContoursMatchI2, shape.ContoursMatchI3} {
		if v := shape.MatchShapes(pentagon, pentagon, m); v != 0 {
			t.Fatalf("method %d: identical shapes scored %.6g, want 0", m, v)
		}
	}
}

func TestMatchShapesCongruentSmall(t *testing.T) {
	copyPts := rotateScaleTranslate(pentagon, 90, 3, 50, 80)
	for _, m := range []int{shape.ContoursMatchI1, shape.ContoursMatchI2, shape.ContoursMatchI3} {
		v := shape.MatchShapes(pentagon, copyPts, m)
		if v > 1e-3 {
			t.Fatalf("method %d: congruent shapes scored %.6g, want ~0", m, v)
		}
	}
}

func TestMatchShapesDistinct(t *testing.T) {
	square := []cv.Point{{X: 0, Y: 0}, {X: 50, Y: 0}, {X: 50, Y: 50}, {X: 0, Y: 50}}
	thin := []cv.Point{{X: 0, Y: 0}, {X: 100, Y: 0}, {X: 100, Y: 5}, {X: 0, Y: 5}}
	same := shape.MatchShapes(square, square, shape.ContoursMatchI2)
	diff := shape.MatchShapes(square, thin, shape.ContoursMatchI2)
	if !(diff > same) {
		t.Fatalf("distinct shapes (%.4f) should score above identical (%.4f)", diff, same)
	}
}

func TestMatchShapesUnknownMethodPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on unknown method")
		}
	}()
	shape.MatchShapes(pentagon, pentagon, 99)
}

// --- ConvexityDefects -----------------------------------------------------

func TestConvexityDefectsNotch(t *testing.T) {
	// A rectangle whose top edge has a V-notch dipping to (5,3).
	contour := []cv.Point{
		{X: 0, Y: 0},   // 0 hull
		{X: 4, Y: 0},   // 1
		{X: 5, Y: 3},   // 2 far point of the notch
		{X: 6, Y: 0},   // 3
		{X: 10, Y: 0},  // 4 hull
		{X: 10, Y: 10}, // 5 hull
		{X: 0, Y: 10},  // 6 hull
	}
	hull := shape.ConvexHullIndices(contour)
	defects := shape.ConvexityDefects(contour, hull)
	if len(defects) != 1 {
		t.Fatalf("got %d defects, want 1: %+v", len(defects), defects)
	}
	d := defects[0]
	if d.Far != (cv.Point{X: 5, Y: 3}) {
		t.Fatalf("far point = %v, want (5,3)", d.Far)
	}
	if d.FarIndex != 2 {
		t.Fatalf("far index = %d, want 2", d.FarIndex)
	}
	if !approx(d.Depth, 3, 1e-9) {
		t.Fatalf("depth = %.6f, want 3", d.Depth)
	}
	if d.Start != (cv.Point{X: 0, Y: 0}) || d.End != (cv.Point{X: 10, Y: 0}) {
		t.Fatalf("edge %v-%v, want (0,0)-(10,0)", d.Start, d.End)
	}
}

func TestConvexityDefectsConvexHasNone(t *testing.T) {
	// A convex quadrilateral has no defects.
	contour := []cv.Point{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10}}
	hull := shape.ConvexHullIndices(contour)
	if d := shape.ConvexityDefects(contour, hull); d != nil {
		t.Fatalf("convex contour has defects: %+v", d)
	}
}
