package matching2

import (
	"math"
	"testing"

	"github.com/malcolmston/opencv/core"
)

func TestFindFundamentalMatEpipolarConstraint(t *testing.T) {
	_, _, _, img1, img2 := scene(t)
	F, ok := FindFundamentalMat(img1, img2)
	if !ok {
		t.Fatal("FindFundamentalMat failed")
	}
	for i := range img1 {
		if e := math.Abs(EpipolarConstraint(F, img1[i], img2[i])); e > 1e-6 {
			t.Fatalf("epipolar constraint = %v at %d, want ~0", e, i)
		}
		if s := SampsonDistance(F, img1[i], img2[i]); s > 1e-3 {
			t.Fatalf("Sampson distance = %v at %d, want ~0", s, i)
		}
	}
}

func TestFundamentalMatchesGroundTruth(t *testing.T) {
	_, R2, t2, img1, img2 := scene(t)
	F, ok := FindFundamentalMat(img1, img2)
	if !ok {
		t.Fatal("FindFundamentalMat failed")
	}
	// Ground-truth F = K^-T [t]_x R K^-1.
	E := Mat3Mul(skew(t2), R2)
	Fgt, ok := FundamentalFromEssential(E, testK)
	if !ok {
		t.Fatal("FundamentalFromEssential failed")
	}
	if d := frob(F, Fgt); d > 1e-4 {
		t.Fatalf("recovered F differs from ground truth, frob=%v", d)
	}
}

func TestEpipolarLineContainsMatch(t *testing.T) {
	_, _, _, img1, img2 := scene(t)
	F, _ := FindFundamentalMat(img1, img2)
	for i := range img1 {
		l := EpipolarLine(F, img1[i], 1) // line in image 2
		if d := PointLineDistance(l, img2[i]); d > 1e-3 {
			t.Fatalf("point %d is %v px off its epipolar line", i, d)
		}
	}
}

func TestEpipoles(t *testing.T) {
	_, _, _, img1, img2 := scene(t)
	F, _ := FindFundamentalMat(img1, img2)
	e1, e2, ok := Epipoles(F)
	if !ok {
		t.Fatal("Epipoles failed")
	}
	// F must annihilate the epipoles: F e1 = 0 and F^T e2 = 0.
	fe1 := Mat3VecMul(F, [3]float64{e1.X, e1.Y, 1})
	if math.Hypot(fe1[0], math.Hypot(fe1[1], fe1[2])) > 1e-6 {
		t.Fatalf("F e1 = %v, want ~0", fe1)
	}
	fte2 := Mat3VecMul(Mat3Transpose(F), [3]float64{e2.X, e2.Y, 1})
	if math.Hypot(fte2[0], math.Hypot(fte2[1], fte2[2])) > 1e-6 {
		t.Fatalf("F^T e2 = %v, want ~0", fte2)
	}
}

func TestFindEssentialAndRecoverPose(t *testing.T) {
	_, R2, t2, img1, img2 := scene(t)
	E, ok := FindEssentialMat(img1, img2, testK)
	if !ok {
		t.Fatal("FindEssentialMat failed")
	}
	// Ground-truth essential matrix.
	Egt := Mat3Mul(skew(t2), R2)
	if d := frob(E, Egt); d > 1e-3 {
		t.Fatalf("recovered E differs from ground truth, frob=%v", d)
	}

	R, tvec, good := RecoverPose(E, img1, img2, testK)
	if good < len(img1)-1 {
		t.Fatalf("cheirality count = %d, want >= %d", good, len(img1)-1)
	}
	// Rotation matches.
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if !approx(R[i][j], R2[i][j], 1e-3) {
				t.Fatalf("recovered R differs at %d,%d: %v vs %v", i, j, R[i][j], R2[i][j])
			}
		}
	}
	// Translation direction matches (recovered only up to positive scale).
	tn := t2
	nn := matching2norm3(tn)
	tn = [3]float64{tn[0] / nn, tn[1] / nn, tn[2] / nn}
	for k := 0; k < 3; k++ {
		if !approx(tvec[k], tn[k], 1e-3) {
			t.Fatalf("recovered t dir differs at %d: %v vs %v", k, tvec[k], tn[k])
		}
	}
}

func TestDecomposeEssentialMat(t *testing.T) {
	_, R2, t2, _, _ := scene(t)
	E := Mat3Mul(skew(t2), R2)
	R1, Rb, tdir := DecomposeEssentialMat(E)
	if !approx(Mat3Det(R1), 1, 1e-6) || !approx(Mat3Det(Rb), 1, 1e-6) {
		t.Fatalf("decomposed rotations must have det +1: %v %v", Mat3Det(R1), Mat3Det(Rb))
	}
	// One of the two rotations equals the ground truth.
	if frob3full(R1, R2) > 1e-3 && frob3full(Rb, R2) > 1e-3 {
		t.Fatalf("neither decomposed rotation matches ground truth")
	}
	// Translation direction (up to sign) matches t2.
	tn := [3]float64{t2[0], t2[1], t2[2]}
	nn := matching2norm3(tn)
	dot := (tdir[0]*tn[0] + tdir[1]*tn[1] + tdir[2]*tn[2]) / nn
	if math.Abs(math.Abs(dot)-1) > 1e-3 {
		t.Fatalf("translation direction off, |dot|=%v", math.Abs(dot))
	}
}

func TestTriangulatePoint(t *testing.T) {
	world, R2, t2, img1, img2 := scene(t)
	P1 := ProjectionMatrix(testK, Mat3Identity(), [3]float64{0, 0, 0})
	P2 := ProjectionMatrix(testK, R2, t2)
	got := TriangulatePoints(P1, P2, img1, img2)
	for i := range world {
		if !approx(got[i].X, world[i].X, 1e-6) ||
			!approx(got[i].Y, world[i].Y, 1e-6) ||
			!approx(got[i].Z, world[i].Z, 1e-6) {
			t.Fatalf("triangulated point %d = %v, want %v", i, got[i], world[i])
		}
	}
}

// frob3full compares two rotation matrices directly (no sign folding), returning
// the Frobenius distance.
func frob3full(a, b [3][3]float64) float64 {
	var s float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			d := a[i][j] - b[i][j]
			s += d * d
		}
	}
	return math.Sqrt(s)
}

// silence unused import if core drops out during edits.
var _ = core.Point2d{}
