package stitch

import (
	"math"
	"testing"
)

func matchesFromHomography(h Homography, src []PointF) []Match {
	ms := make([]Match, len(src))
	for i, p := range src {
		ms[i] = Match{Src: p, Dst: h.Apply(p)}
	}
	return ms
}

func TestIdentityAndTranslationApply(t *testing.T) {
	id := IdentityHomography()
	p := PointF{3, 4}
	if got := id.Apply(p); got != p {
		t.Fatalf("identity apply = %v, want %v", got, p)
	}
	tr := TranslationHomography(5, -3)
	if got := tr.Apply(p); got.X != 8 || got.Y != 1 {
		t.Fatalf("translation apply = %v, want {8 1}", got)
	}
}

func TestHomographyInverseAndMul(t *testing.T) {
	h := Homography{{1.2, 0.1, 5}, {-0.05, 1.1, -3}, {0.0002, 0.0001, 1}}
	inv, ok := h.Inverse()
	if !ok {
		t.Fatal("expected invertible homography")
	}
	prod := h.Mul(inv).Normalize()
	id := IdentityHomography()
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(prod[i][j]-id[i][j]) > 1e-9 {
				t.Fatalf("H·H⁻¹[%d][%d] = %g, want %g", i, j, prod[i][j], id[i][j])
			}
		}
	}
}

func TestEstimateHomographyDLTTranslation(t *testing.T) {
	h := TranslationHomography(7, -4)
	src := []PointF{{0, 0}, {10, 0}, {0, 10}, {10, 10}, {5, 5}}
	ms := matchesFromHomography(h, src)
	est, ok := EstimateHomographyDLT(ms)
	if !ok {
		t.Fatal("DLT failed")
	}
	if e := MeanReprojectionError(est, ms); e > 1e-6 {
		t.Fatalf("reprojection error %g too large", e)
	}
	// A fresh point must also map correctly.
	got := est.Apply(PointF{3, 8})
	if math.Abs(got.X-10) > 1e-6 || math.Abs(got.Y-4) > 1e-6 {
		t.Fatalf("mapped fresh point = %v, want {10 4}", got)
	}
}

func TestEstimateHomographyDLTProjective(t *testing.T) {
	h := Homography{{1.2, 0.1, 5}, {-0.05, 1.1, -3}, {0.0002, 0.0001, 1}}
	src := []PointF{{0, 0}, {10, 0}, {0, 10}, {10, 10}, {5, 5}, {7, 3}, {2, 9}}
	ms := matchesFromHomography(h, src)
	est, ok := EstimateHomographyDLT(ms)
	if !ok {
		t.Fatal("DLT failed")
	}
	if e := MeanReprojectionError(est, ms); e > 1e-5 {
		t.Fatalf("reprojection error %g too large", e)
	}
	hn := h.Normalize()
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(est[i][j]-hn[i][j]) > 1e-4 {
				t.Fatalf("H[%d][%d] = %g, want %g", i, j, est[i][j], hn[i][j])
			}
		}
	}
}

func TestEstimateHomographyRANSAC(t *testing.T) {
	h := TranslationHomography(4, 7)
	src := []PointF{{0, 0}, {10, 0}, {0, 10}, {10, 10}, {5, 5}, {3, 8}, {9, 2}, {6, 6}}
	ms := matchesFromHomography(h, src)
	// Inject outliers.
	ms = append(ms,
		Match{Src: PointF{1, 1}, Dst: PointF{99, 99}},
		Match{Src: PointF{2, 2}, Dst: PointF{-40, 60}},
		Match{Src: PointF{4, 4}, Dst: PointF{80, -20}},
	)
	est, inliers, ok := EstimateHomographyRANSAC(ms, 1.0, 300, 42)
	if !ok {
		t.Fatal("RANSAC failed")
	}
	if len(inliers) < 8 {
		t.Fatalf("found %d inliers, want >= 8", len(inliers))
	}
	got := est.Apply(PointF{5, 5})
	if math.Abs(got.X-9) > 1e-3 || math.Abs(got.Y-12) > 1e-3 {
		t.Fatalf("RANSAC model maps {5 5} to %v, want ~{9 12}", got)
	}
}

func TestPointFDistance(t *testing.T) {
	if d := (PointF{0, 0}).Distance(PointF{3, 4}); math.Abs(d-5) > 1e-12 {
		t.Fatalf("distance = %g, want 5", d)
	}
}
