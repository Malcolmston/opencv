package moments2

import (
	"math"
	"math/cmplx"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestShapeDescriptorsRectangle(t *testing.T) {
	// A wide filled rectangle: 4 rows x 20 cols.
	m := ImageMoments(filledMat(4, 20, 255))
	if e := Eccentricity(m); e <= 0.5 || e >= 1 {
		t.Errorf("Eccentricity = %g, want in (0.5,1)", e)
	}
	if el := Elongation(m); el <= 1 {
		t.Errorf("Elongation = %g, want > 1", el)
	}
	// Wide rectangle: major axis horizontal -> orientation near 0.
	approx(t, "Orientation wide", Orientation(m), 0, 1e-9)
	// Tall rectangle: orientation near +/- pi/2.
	mt := ImageMoments(filledMat(20, 4, 255))
	if o := math.Abs(Orientation(mt)); math.Abs(o-math.Pi/2) > 1e-9 {
		t.Errorf("Orientation tall = %g, want +/-pi/2", Orientation(mt))
	}
}

func TestShapeDescriptorsCircleEccentricity(t *testing.T) {
	m := MaskMoments(filledDisk(61, 28))
	if e := Eccentricity(m); e > 0.15 {
		t.Errorf("disk eccentricity = %g, want ~0", e)
	}
}

func TestSolidityExtentSquare(t *testing.T) {
	square := []cv.Point{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10}}
	approx(t, "solidity square", Solidity(square), 1, 1e-9)
	approx(t, "extent square", Extent(square), 100.0/121.0, 1e-9)
	if !IsContourConvex(square) {
		t.Error("square should be convex")
	}
	approx(t, "convexity square", Convexity(square), 1, 1e-9)
}

func TestSolidityConcave(t *testing.T) {
	// An L-shaped (concave) polygon has solidity below 1.
	ell := []cv.Point{
		{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 4},
		{X: 4, Y: 4}, {X: 4, Y: 10}, {X: 0, Y: 10},
	}
	if s := Solidity(ell); s >= 1 || s <= 0 {
		t.Errorf("L-shape solidity = %g, want in (0,1)", s)
	}
	if IsContourConvex(ell) {
		t.Error("L-shape should not be convex")
	}
}

func TestPolygonAreaPerimeterCentroid(t *testing.T) {
	sq := []cv.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 4}, {X: 0, Y: 4}}
	approx(t, "PolygonArea", PolygonArea(sq), 16, 1e-9)
	approx(t, "PolygonPerimeter", PolygonPerimeter(sq), 16, 1e-9)
	c := PolygonCentroid(sq)
	approx(t, "centroid x", c.X, 2, 1e-9)
	approx(t, "centroid y", c.Y, 2, 1e-9)
	approx(t, "EquivDiameter", EquivalentDiameter(math.Pi*4), 4, 1e-9)
	approx(t, "Circularity of circle", Circularity(math.Pi*25, 2*math.Pi*5), 1, 1e-9)
	approx(t, "Compactness", Compactness(16, 16), 16, 1e-9)
}

func TestRadialPolynomialKnown(t *testing.T) {
	approx(t, "R(0,0)", RadialPolynomial(0, 0, 0.7), 1, 1e-12)
	approx(t, "R(1,1)", RadialPolynomial(1, 1, 0.7), 0.7, 1e-12)
	approx(t, "R(2,0)", RadialPolynomial(2, 0, 0.7), 2*0.49-1, 1e-12)
	approx(t, "R(2,2)", RadialPolynomial(2, 2, 0.7), 0.49, 1e-12)
	approx(t, "R(3,1)", RadialPolynomial(3, 1, 0.7), 3*math.Pow(0.7, 3)-2*0.7, 1e-12)
	approx(t, "R(4,0)", RadialPolynomial(4, 0, 0.7),
		6*math.Pow(0.7, 4)-6*0.49+1, 1e-12)
	// Undefined parity yields zero.
	approx(t, "R(2,1)", RadialPolynomial(2, 1, 0.7), 0, 0)
	approx(t, "PseudoR(0,0)", PseudoZernikeRadial(0, 0, 0.5), 1, 1e-12)
	approx(t, "PseudoR(1,1)", PseudoZernikeRadial(1, 1, 0.5), 0.5, 1e-12)
}

func TestZernikeDisk(t *testing.T) {
	disk := filledDisk(61, 28)
	a00 := ZernikeMoment(disk, 0, 0)
	// A_00 of an inscribed disk approaches value*(r/radius)^2. The disk is
	// filled with 255, so A00 real ~ 255*(28/30.5)^2 ~ 214.
	if real(a00) < 190 || real(a00) > 230 {
		t.Errorf("Zernike A00 real = %g, want ~214", real(a00))
	}
	approx(t, "A00 imag", imag(a00), 0, 1e-6)
	// A rotationally symmetric disk has near-zero A_11 relative to A_00.
	if ratio := ZernikeMagnitude(disk, 1, 1) / cmplx.Abs(a00); ratio > 0.02 {
		t.Errorf("Zernike |A11|/|A00| = %g, want ~0 for symmetric disk", ratio)
	}
	// Invalid order returns exactly zero.
	if ZernikeMoment(disk, 2, 1) != 0 {
		t.Error("Zernike A(2,1) should be 0 (parity)")
	}
	coeffs := ZernikeMoments(disk, 2)
	if len(coeffs) != 4 { // (0,0),(1,1),(2,0),(2,2)
		t.Errorf("ZernikeMoments count = %d, want 4", len(coeffs))
	}
}

func TestLegendrePolynomialKnown(t *testing.T) {
	for n := 0; n <= 5; n++ {
		approx(t, "Pn(1)", LegendrePolynomial(n, 1), 1, 1e-12)
	}
	approx(t, "P2(0)", LegendrePolynomial(2, 0), -0.5, 1e-12)
	approx(t, "P3(0.5)", LegendrePolynomial(3, 0.5), -0.4375, 1e-12)
	approx(t, "P2(0.3)", LegendrePolynomial(2, 0.3), (3*0.09-1)/2, 1e-12)
}

func TestLegendreMomentConstant(t *testing.T) {
	src := filledMat(16, 16, 50)
	// L00 of a constant image equals the constant value.
	approx(t, "L00", LegendreMoment(src, 0, 0), 50, 1e-9)
	// Odd moments of a symmetric constant image vanish.
	approx(t, "L10", LegendreMoment(src, 1, 0), 0, 1e-9)
	approx(t, "L01", LegendreMoment(src, 0, 1), 0, 1e-9)
	moms := LegendreMoments(src, 2)
	approx(t, "L00 matrix", moms[0][0], 50, 1e-9)
}

func TestFlusserRectangle(t *testing.T) {
	W, H := 5, 3
	m := ImageMoments(filledMat(H, W, 1))
	want := float64((W*W-1)*(H*H-1)) / (144 * float64(W*W*H*H))
	approx(t, "FlusserI1", FlusserI1(m), want, 1e-12)
	// Affine invariants are translation invariant: shift the shape.
	big := cv.NewMat(9, 11, 1)
	for y := 3; y < 3+H; y++ {
		for x := 4; x < 4+W; x++ {
			big.Data[y*big.Cols+x] = 1
		}
	}
	mb := ImageMoments(big)
	inv := FlusserInvariants(m)
	invb := FlusserInvariants(mb)
	for i := range inv {
		if math.Abs(inv[i]-invb[i]) > 1e-12 {
			t.Errorf("FlusserI%d not translation invariant: %g vs %g", i+1, inv[i], invb[i])
		}
	}
}

func TestFourierCircle(t *testing.T) {
	const N = 64
	R := 20.0
	pts := make([]cv.Point2f, N)
	for i := 0; i < N; i++ {
		ang := 2 * math.Pi * float64(i) / float64(N)
		pts[i] = cv.Point2f{X: 50 + R*math.Cos(ang), Y: 50 + R*math.Sin(ang)}
	}
	fd := FourierDescriptors(pts)
	// For a uniformly sampled circle, only FD[1] is significant.
	approx(t, "|FD[1]|", cmplx.Abs(fd[1]), N*R, 1e-6)
	approx(t, "|FD[2]|", cmplx.Abs(fd[2]), 0, 1e-6)
	norm := NormalizedFourierDescriptors(fd)
	approx(t, "normalized[0]", norm[0], 1, 1e-9)
	if norm[2] > 1e-6 {
		t.Errorf("normalized[2] = %g, want ~0", norm[2])
	}
	// Distance to itself is zero.
	approx(t, "self distance", FourierDescriptorDistance(norm, norm), 0, 0)
	// Reconstruction with the fundamental returns a circle of radius R.
	rec := ReconstructContour(fd, 32, 2)
	for _, p := range rec {
		r := math.Hypot(p.X-50, p.Y-50)
		approx(t, "reconstructed radius", r, R, 1e-6)
	}
}

func TestResampleContour(t *testing.T) {
	sq := []cv.Point{{X: 0, Y: 0}, {X: 8, Y: 0}, {X: 8, Y: 8}, {X: 0, Y: 8}}
	rs := ResampleContour(sq, 40)
	if len(rs) != 40 {
		t.Fatalf("ResampleContour len = %d, want 40", len(rs))
	}
	// Every resampled point lies on the square boundary (within tolerance).
	for _, p := range rs {
		onEdge := p.X <= 1e-9 || math.Abs(p.X-8) <= 1e-9 ||
			p.Y <= 1e-9 || math.Abs(p.Y-8) <= 1e-9
		if !onEdge && !(p.X >= -1e-9 && p.X <= 8+1e-9 && p.Y >= -1e-9 && p.Y <= 8+1e-9) {
			t.Errorf("resampled point %v off boundary", p)
		}
	}
}

func TestShapeContext(t *testing.T) {
	// Points on a small grid.
	pts := []cv.Point2f{
		{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10}, {X: 5, Y: 5},
	}
	sc := ComputeShapeContext(pts, 5, 12, 0.125, 2.0)
	if len(sc) != len(pts) {
		t.Fatalf("shape context count = %d, want %d", len(sc), len(pts))
	}
	if sc[0].RadialBins != 5 || sc[0].AngularBins != 12 {
		t.Errorf("histogram dims = %dx%d", sc[0].RadialBins, sc[0].AngularBins)
	}
	// A histogram compared with itself has zero cost.
	approx(t, "self cost", ShapeContextCost(sc[0], sc[0]), 0, 1e-12)
	// A descriptor matched to itself has zero global cost.
	approx(t, "self match", ShapeContextMatchCost(sc, sc), 0, 1e-12)
}

func TestConvexityDefects(t *testing.T) {
	// Square with an inward notch on the top edge (index 2 juts to y=3).
	contour := []cv.Point{
		{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 5, Y: 3}, {X: 6, Y: 0},
		{X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10},
	}
	hull := ConvexHullIndices(contour)
	defects := ConvexityDefects(contour, hull, 0.5)
	if len(defects) != 1 {
		t.Fatalf("defect count = %d, want 1", len(defects))
	}
	d := defects[0]
	if d.Far != (cv.Point{X: 5, Y: 3}) {
		t.Errorf("defect far point = %v, want (5,3)", d.Far)
	}
	approx(t, "defect depth", d.Depth, 3, 1e-9)
	if IsContourConvex(contour) {
		t.Error("notched contour should not be convex")
	}
	// Convexity ratio below 1 for a concave shape.
	if r := ConvexityRatio(contour); r >= 1 {
		t.Errorf("ConvexityRatio = %g, want < 1", r)
	}
}

func TestConvexHullIndicesSquare(t *testing.T) {
	// Interior point must be excluded from the hull.
	contour := []cv.Point{
		{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10}, {X: 5, Y: 5},
	}
	hull := ConvexHullIndices(contour)
	if len(hull) != 4 {
		t.Fatalf("hull size = %d, want 4", len(hull))
	}
	for _, idx := range hull {
		if idx == 4 {
			t.Error("interior point (index 4) should not be on the hull")
		}
	}
}

func BenchmarkZernikeMoments(b *testing.B) {
	disk := filledDisk(64, 30)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ZernikeMoments(disk, 8)
	}
}
