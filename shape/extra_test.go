package shape_test

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/shape"
)

// distinctShape is an asymmetric polygon used for shape-matching tests.
var distinctShape = []cv.Point{
	{X: 0, Y: 0}, {X: 60, Y: 5}, {X: 80, Y: 40}, {X: 55, Y: 70},
	{X: 30, Y: 55}, {X: 20, Y: 90}, {X: -10, Y: 45},
}

// --- SolveAssignment (Hungarian) ------------------------------------------

func TestSolveAssignmentSquare(t *testing.T) {
	cost := [][]float64{{1, 2}, {2, 1}}
	assign, total := shape.SolveAssignment(cost)
	if assign[0] != 0 || assign[1] != 1 {
		t.Fatalf("assignment = %v, want [0 1]", assign)
	}
	if !approx(total, 2, 1e-9) {
		t.Fatalf("total = %.4f, want 2", total)
	}
}

func TestSolveAssignmentKnownOptimum(t *testing.T) {
	// Classic 3x3 example; optimal assignment has total cost 5.
	cost := [][]float64{
		{9, 11, 14},
		{6, 15, 13},
		{12, 13, 6},
	}
	assign, total := shape.SolveAssignment(cost)
	// Verify it is a permutation and the total matches a brute-force optimum.
	best := bruteForceAssignment(cost)
	if !approx(total, best, 1e-9) {
		t.Fatalf("total = %.4f, want optimum %.4f (assign %v)", total, best, assign)
	}
	seen := map[int]bool{}
	for _, j := range assign {
		if j < 0 || seen[j] {
			t.Fatalf("not a permutation: %v", assign)
		}
		seen[j] = true
	}
}

func TestSolveAssignmentRectangular(t *testing.T) {
	cost := [][]float64{{1, 2, 3}, {3, 2, 1}}
	assign, total := shape.SolveAssignment(cost)
	if assign[0] != 0 || assign[1] != 2 {
		t.Fatalf("assignment = %v, want [0 2]", assign)
	}
	if !approx(total, 2, 1e-9) {
		t.Fatalf("total = %.4f, want 2", total)
	}
}

func bruteForceAssignment(cost [][]float64) float64 {
	n := len(cost)
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	best := math.Inf(1)
	var rec func(k int, used []bool, sum float64)
	rec = func(k int, used []bool, sum float64) {
		if k == n {
			if sum < best {
				best = sum
			}
			return
		}
		for j := 0; j < n; j++ {
			if !used[j] {
				used[j] = true
				rec(k+1, used, sum+cost[k][j])
				used[j] = false
			}
		}
	}
	rec(0, make([]bool, n), 0)
	return best
}

// --- IsContourConvex ------------------------------------------------------

func TestIsContourConvex(t *testing.T) {
	square := []cv.Point{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10}}
	if !shape.IsContourConvex(square) {
		t.Fatalf("square reported non-convex")
	}
	notch := []cv.Point{
		{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 5, Y: 3}, {X: 6, Y: 0},
		{X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10},
	}
	if shape.IsContourConvex(notch) {
		t.Fatalf("notched polygon reported convex")
	}
}

// --- PointPolygonTest -----------------------------------------------------

func TestPointPolygonTestSigns(t *testing.T) {
	square := []cv.Point{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10}}
	if v := shape.PointPolygonTest(square, shape.Point2D{X: 5, Y: 5}, false); v != 1 {
		t.Fatalf("inside sign = %v, want 1", v)
	}
	if v := shape.PointPolygonTest(square, shape.Point2D{X: 15, Y: 5}, false); v != -1 {
		t.Fatalf("outside sign = %v, want -1", v)
	}
	if v := shape.PointPolygonTest(square, shape.Point2D{X: 10, Y: 5}, false); v != 0 {
		t.Fatalf("on-edge sign = %v, want 0", v)
	}
}

func TestPointPolygonTestDistance(t *testing.T) {
	square := []cv.Point{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10}}
	// Inside point (5,5) is 5 from the nearest edge.
	if v := shape.PointPolygonTest(square, shape.Point2D{X: 5, Y: 5}, true); !approx(v, 5, 1e-9) {
		t.Fatalf("inside dist = %.4f, want 5", v)
	}
	// Outside point (13,5) is 3 from the right edge.
	if v := shape.PointPolygonTest(square, shape.Point2D{X: 13, Y: 5}, true); !approx(v, -3, 1e-9) {
		t.Fatalf("outside dist = %.4f, want -3", v)
	}
}

// --- RotatedRectangleIntersection -----------------------------------------

func TestRotatedRectangleIntersectionPartial(t *testing.T) {
	r1 := cv.RotatedRect{CenterX: 0, CenterY: 0, Width: 4, Height: 4, Angle: 0}
	r2 := cv.RotatedRect{CenterX: 2, CenterY: 2, Width: 4, Height: 4, Angle: 0}
	kind, poly := shape.RotatedRectangleIntersection(r1, r2)
	if kind != shape.IntersectPartial {
		t.Fatalf("kind = %v, want partial", kind)
	}
	// Overlap is the square [0,2]x[0,2], area 4.
	if a := polyArea(poly); !approx(a, 4, 1e-6) {
		t.Fatalf("overlap area = %.4f, want 4", a)
	}
}

func TestRotatedRectangleIntersectionFull(t *testing.T) {
	outer := cv.RotatedRect{CenterX: 0, CenterY: 0, Width: 10, Height: 10, Angle: 0}
	inner := cv.RotatedRect{CenterX: 0, CenterY: 0, Width: 4, Height: 4, Angle: 30}
	kind, poly := shape.RotatedRectangleIntersection(outer, inner)
	if kind != shape.IntersectFull {
		t.Fatalf("kind = %v, want full", kind)
	}
	if a := polyArea(poly); !approx(a, 16, 1e-4) {
		t.Fatalf("overlap area = %.4f, want 16 (inner square)", a)
	}
}

func TestRotatedRectangleIntersectionNone(t *testing.T) {
	r1 := cv.RotatedRect{CenterX: 0, CenterY: 0, Width: 2, Height: 2, Angle: 0}
	r2 := cv.RotatedRect{CenterX: 10, CenterY: 10, Width: 2, Height: 2, Angle: 0}
	kind, poly := shape.RotatedRectangleIntersection(r1, r2)
	if kind != shape.IntersectNone || poly != nil {
		t.Fatalf("kind = %v poly = %v, want none/nil", kind, poly)
	}
}

func polyArea(poly []shape.Point2D) float64 {
	n := len(poly)
	if n < 3 {
		return 0
	}
	var a float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		a += poly[i].X*poly[j].Y - poly[j].X*poly[i].Y
	}
	return math.Abs(a) / 2
}

// --- EMDL1 ----------------------------------------------------------------

func TestEMDL1(t *testing.T) {
	// Mass at bin 0 vs mass at bin 2: must move one unit across two bins.
	if v := shape.EMDL1([]float64{1, 0, 0}, []float64{0, 0, 1}); !approx(v, 2, 1e-9) {
		t.Fatalf("EMDL1 = %.4f, want 2", v)
	}
	// Identical distributions have zero distance.
	if v := shape.EMDL1([]float64{2, 3, 5}, []float64{2, 3, 5}); !approx(v, 0, 1e-9) {
		t.Fatalf("EMDL1 identical = %.4f, want 0", v)
	}
	// Scale invariance after normalisation.
	if v := shape.EMDL1([]float64{2, 0}, []float64{0, 8}); !approx(v, 1, 1e-9) {
		t.Fatalf("EMDL1 normalised = %.4f, want 1", v)
	}
}

// --- FitLineRobust --------------------------------------------------------

func TestFitLineRobustIgnoresOutliers(t *testing.T) {
	var pts []cv.Point
	for x := -20; x <= 20; x++ {
		pts = append(pts, cv.Point{X: x, Y: 0}) // on the x-axis
	}
	// A few gross high-leverage outliers off to one side.
	pts = append(pts, cv.Point{X: 25, Y: 40}, cv.Point{X: 26, Y: 42}, cv.Point{X: 27, Y: 41})

	// Plain L2 is pulled off the axis.
	_, vyL2, _, _ := shape.FitLine(pts)
	// Robust Welsch stays near the x-axis direction (vy ~ 0).
	vx, vy, _, _ := shape.FitLineRobust(pts, shape.DistWelsch)
	if math.Abs(vy) > 0.05 {
		t.Fatalf("robust vy = %.4f, want ~0 (L2 vy = %.4f)", vy, vyL2)
	}
	if !approx(math.Hypot(vx, vy), 1, 1e-9) {
		t.Fatalf("direction not unit length")
	}
	if math.Abs(vy) >= math.Abs(vyL2) {
		t.Fatalf("robust fit (vy=%.4f) not better than L2 (vy=%.4f)", vy, vyL2)
	}
}

func TestFitLineRobustL2MatchesFitLine(t *testing.T) {
	pts := []cv.Point{{X: 0, Y: 0}, {X: 1, Y: 1}, {X: 2, Y: 2}, {X: 3, Y: 3}}
	a, b, c, d := shape.FitLineRobust(pts, shape.DistL2)
	e, f, g, h := shape.FitLine(pts)
	if a != e || b != f || c != g || d != h {
		t.Fatalf("DistL2 differs from FitLine: (%v,%v,%v,%v) vs (%v,%v,%v,%v)", a, b, c, d, e, f, g, h)
	}
}

// --- Thin-plate spline ----------------------------------------------------

func TestTPSInterpolatesControlPoints(t *testing.T) {
	src := []shape.Point2D{
		{X: 0, Y: 0}, {X: 1, Y: 0}, {X: 0, Y: 1}, {X: 1, Y: 1},
		{X: 0.5, Y: 0.5}, {X: 2, Y: 3}, {X: -1, Y: 2},
	}
	dst := []shape.Point2D{
		{X: 0.1, Y: -0.2}, {X: 1.3, Y: 0.4}, {X: -0.2, Y: 1.1}, {X: 1.4, Y: 1.5},
		{X: 0.6, Y: 0.3}, {X: 2.5, Y: 2.7}, {X: -1.2, Y: 2.4},
	}
	tps := shape.NewThinPlateSplineShapeTransformer(0)
	tps.EstimateTransformation(src, dst)
	got := tps.ApplyTransformation(src)
	for i := range src {
		if !approx(got[i].X, dst[i].X, 1e-6) || !approx(got[i].Y, dst[i].Y, 1e-6) {
			t.Fatalf("control %d mapped to (%.6f,%.6f), want (%.6f,%.6f)",
				i, got[i].X, got[i].Y, dst[i].X, dst[i].Y)
		}
	}
}

func TestTPSAffineHasZeroBending(t *testing.T) {
	src := []shape.Point2D{
		{X: 0, Y: 0}, {X: 1, Y: 0}, {X: 0, Y: 1}, {X: 1, Y: 1}, {X: 2, Y: 3}, {X: -1, Y: 2},
	}
	// A pure affine correspondence (rotation+scale+translation).
	dst := make([]shape.Point2D, len(src))
	c, s := math.Cos(0.5), math.Sin(0.5)
	for i, p := range src {
		dst[i] = shape.Point2D{
			X: 1.5*(p.X*c-p.Y*s) + 3,
			Y: 1.5*(p.X*s+p.Y*c) - 2,
		}
	}
	tps := shape.NewThinPlateSplineShapeTransformer(0)
	tps.EstimateTransformation(src, dst)
	if be := tps.BendingEnergy(); math.Abs(be) > 1e-6 {
		t.Fatalf("bending energy of affine map = %.3e, want ~0", be)
	}
}

// --- AffineTransformer ----------------------------------------------------

func TestAffineTransformerRecoversMatrix(t *testing.T) {
	src := []shape.Point2D{{X: 0, Y: 0}, {X: 1, Y: 0}, {X: 0, Y: 1}, {X: 2, Y: 3}}
	// Known affine: [[2,0.5,1],[-0.3,1.5,-2]].
	m := [2][3]float64{{2, 0.5, 1}, {-0.3, 1.5, -2}}
	dst := make([]shape.Point2D, len(src))
	for i, p := range src {
		dst[i] = shape.Point2D{
			X: m[0][0]*p.X + m[0][1]*p.Y + m[0][2],
			Y: m[1][0]*p.X + m[1][1]*p.Y + m[1][2],
		}
	}
	at := shape.NewAffineTransformer(true)
	at.EstimateTransformation(src, dst)
	got := at.Matrix()
	for r := 0; r < 2; r++ {
		for c := 0; c < 3; c++ {
			if !approx(got[r][c], m[r][c], 1e-6) {
				t.Fatalf("matrix[%d][%d] = %.6f, want %.6f", r, c, got[r][c], m[r][c])
			}
		}
	}
}

func TestAffineTransformerWarpTranslation(t *testing.T) {
	src := cv.NewMat(20, 20, 1)
	src.Set(10, 10, 0, 200) // a bright pixel at (col=10,row=10)

	// A pure translation by (+3, +2): source point p maps to p+(3,2).
	sp := []shape.Point2D{{X: 0, Y: 0}, {X: 1, Y: 0}, {X: 0, Y: 1}}
	dp := []shape.Point2D{{X: 3, Y: 2}, {X: 4, Y: 2}, {X: 3, Y: 3}}
	at := shape.NewAffineTransformer(true)
	at.EstimateTransformation(sp, dp)
	warped := at.WarpImage(src)
	if got := warped.At(12, 13, 0); got < 150 {
		t.Fatalf("translated bright pixel at (13,12) = %d, want bright", got)
	}
}

// --- Shape context distance -----------------------------------------------

func TestShapeContextMatchesRotatedScaledCopy(t *testing.T) {
	rotated := rotateScaleTranslate(distinctShape, 30, 1.4, 120, 60)
	square := []cv.Point{{X: 0, Y: 0}, {X: 60, Y: 0}, {X: 60, Y: 60}, {X: 0, Y: 60}}

	ext := shape.NewShapeContextDistanceExtractor(60, true)
	dSame := ext.ComputeDistance(distinctShape, rotated)
	dDiff := ext.ComputeDistance(distinctShape, square)
	if !(dSame < dDiff) {
		t.Fatalf("rotated/scaled copy (%.4f) not closer than a different shape (%.4f)", dSame, dDiff)
	}
	// Determinism.
	if d2 := ext.ComputeDistance(distinctShape, rotated); d2 != dSame {
		t.Fatalf("non-deterministic: %.6f vs %.6f", d2, dSame)
	}
}

func TestShapeContextDescriptorNormalised(t *testing.T) {
	pts := shape.FloatPoints(distinctShape)
	sc := shape.ShapeContext{}
	hist := sc.Descriptor(pts)
	for i, h := range hist {
		var sum float64
		for _, v := range h {
			sum += v
		}
		if !approx(sum, 1, 1e-9) {
			t.Fatalf("histogram %d sums to %.6f, want 1", i, sum)
		}
	}
}

// --- Hausdorff ------------------------------------------------------------

func TestHausdorffDistance(t *testing.T) {
	a := []cv.Point{{X: 0, Y: 0}, {X: 10, Y: 0}}
	b := []cv.Point{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 5, Y: 3}}
	ext := shape.NewHausdorffDistanceExtractor(shape.HausdorffL2, 1.0)
	// a's points match exactly in b; b's (5,3) is √34 from its nearest corner in a.
	if d := ext.ComputeDistance(a, b); !approx(d, math.Sqrt(34), 1e-9) {
		t.Fatalf("Hausdorff = %.4f, want %.4f", d, math.Sqrt(34))
	}
}

func TestDirectedHausdorffAsymmetric(t *testing.T) {
	a := []shape.Point2D{{X: 0, Y: 0}}
	b := []shape.Point2D{{X: 0, Y: 0}, {X: 4, Y: 0}}
	if d := shape.DirectedHausdorff(a, b, shape.HausdorffL2); !approx(d, 0, 1e-9) {
		t.Fatalf("directed a->b = %.4f, want 0", d)
	}
	if d := shape.DirectedHausdorff(b, a, shape.HausdorffL2); !approx(d, 4, 1e-9) {
		t.Fatalf("directed b->a = %.4f, want 4", d)
	}
}
