package matching2

import (
	"math"
	"testing"

	"github.com/malcolmston/opencv/core"
)

func TestFitLine(t *testing.T) {
	// Points on y = 2x + 1.
	pts := []core.Point2d{{X: 0, Y: 1}, {X: 1, Y: 3}, {X: 2, Y: 5}, {X: 3, Y: 7}}
	l, ok := FitLine(pts)
	if !ok {
		t.Fatal("FitLine failed")
	}
	for _, p := range pts {
		if d := l.Distance(p); d > 1e-9 {
			t.Fatalf("point %v is %v off the line", p, d)
		}
	}
}

func TestFitLineRANSAC(t *testing.T) {
	// Inliers on y = -x + 10, plus outliers.
	pts := []core.Point2d{
		{X: 0, Y: 10}, {X: 1, Y: 9}, {X: 2, Y: 8}, {X: 3, Y: 7},
		{X: 4, Y: 6}, {X: 5, Y: 5}, {X: 6, Y: 4}, {X: 7, Y: 3},
		{X: 2, Y: 30}, {X: 5, Y: -20}, // outliers
	}
	res := FitLineRANSAC(pts, 0.5, 300, DefaultRANSACSeed)
	if !res.Ok {
		t.Fatal("FitLineRANSAC failed")
	}
	if res.NumInliers != 8 {
		t.Fatalf("inliers = %d, want 8", res.NumInliers)
	}
	for i := 0; i < 8; i++ {
		if !res.Inliers[i] {
			t.Fatalf("point %d should be an inlier", i)
		}
	}
}

func TestLMedS(t *testing.T) {
	// Fit a line via the generic LMedS. Fewer than half are outliers.
	pts := []core.Point2d{
		{X: 0, Y: 0}, {X: 1, Y: 1}, {X: 2, Y: 2}, {X: 3, Y: 3},
		{X: 4, Y: 4}, {X: 5, Y: 5}, {X: 6, Y: 6}, {X: 2, Y: 9},
	}
	fit := func(sample []int) (Line2D, bool) {
		return FitLine([]core.Point2d{pts[sample[0]], pts[sample[1]]})
	}
	residual := func(l Line2D, i int) float64 { return l.Distance(pts[i]) }
	res := LMedS(len(pts), 2, 200, DefaultRANSACSeed, fit, residual)
	if !res.Ok {
		t.Fatal("LMedS failed")
	}
	// The single outlier (index 7) must be rejected.
	if res.Inliers[7] {
		t.Fatal("LMedS accepted the outlier")
	}
	if res.NumInliers != 7 {
		t.Fatalf("LMedS inliers = %d, want 7", res.NumInliers)
	}
}

func TestNormalizePoints2D(t *testing.T) {
	pts := []core.Point2d{{X: 10, Y: 20}, {X: 30, Y: 40}, {X: 50, Y: 10}, {X: 5, Y: 60}}
	norm, T := NormalizePoints2D(pts)
	// Centroid of normalized points is the origin.
	var cx, cy float64
	for _, p := range norm {
		cx += p.X
		cy += p.Y
	}
	cx /= float64(len(norm))
	cy /= float64(len(norm))
	if math.Abs(cx) > 1e-9 || math.Abs(cy) > 1e-9 {
		t.Fatalf("normalized centroid = (%v,%v), want origin", cx, cy)
	}
	// Mean distance from origin is sqrt(2).
	var md float64
	for _, p := range norm {
		md += math.Hypot(p.X, p.Y)
	}
	md /= float64(len(norm))
	if !approx(md, math.Sqrt2, 1e-9) {
		t.Fatalf("mean distance = %v, want sqrt(2)", md)
	}
	// T maps original points to the normalized ones.
	for i, p := range pts {
		v := Mat3VecMul(T, [3]float64{p.X, p.Y, 1})
		if !approx(v[0], norm[i].X, 1e-9) || !approx(v[1], norm[i].Y, 1e-9) {
			t.Fatalf("T does not map point %d correctly", i)
		}
	}
}

func TestRANSACDeterministic(t *testing.T) {
	pts := []core.Point2d{
		{X: 0, Y: 1}, {X: 1, Y: 1}, {X: 2, Y: 1}, {X: 3, Y: 1},
		{X: 1.5, Y: 9},
	}
	a := FitLineRANSAC(pts, 0.1, 100, DefaultRANSACSeed)
	b := FitLineRANSAC(pts, 0.1, 100, DefaultRANSACSeed)
	if a.Model != b.Model || a.NumInliers != b.NumInliers {
		t.Fatal("RANSAC is not deterministic for a fixed seed")
	}
}
