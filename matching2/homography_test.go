package matching2

import (
	"testing"

	"github.com/malcolmston/opencv/core"
)

var hGT = [3][3]float64{
	{1.2, 0.1, 30},
	{-0.05, 1.1, 20},
	{0.0002, 0.0001, 1},
}

func hSrcPoints() []core.Point2d {
	return []core.Point2d{
		{X: 0, Y: 0}, {X: 100, Y: 0}, {X: 0, Y: 100}, {X: 100, Y: 100},
		{X: 50, Y: 20}, {X: 20, Y: 80}, {X: 75, Y: 60},
	}
}

func TestFindHomographyDLT(t *testing.T) {
	src := hSrcPoints()
	dst := make([]core.Point2d, len(src))
	for i, p := range src {
		dst[i] = ApplyHomography(hGT, p)
	}
	H, ok := FindHomographyDLT(src, dst)
	if !ok {
		t.Fatal("FindHomographyDLT failed")
	}
	if d := frob(H, hGT); d > 1e-6 {
		t.Fatalf("recovered H differs from ground truth, frob=%v\nH=%v", d, H)
	}
	// Round-trip reprojection is essentially exact.
	for i, p := range src {
		if e := HomographyReprojectionError(H, p, dst[i]); e > 1e-6 {
			t.Fatalf("reprojection error %v at point %d", e, i)
		}
	}
}

func TestSymmetricTransferError(t *testing.T) {
	src := hSrcPoints()
	dst := make([]core.Point2d, len(src))
	for i, p := range src {
		dst[i] = ApplyHomography(hGT, p)
	}
	// Exact correspondences have ~zero symmetric transfer error.
	if e := SymmetricTransferError(hGT, src[0], dst[0]); e > 1e-6 {
		t.Fatalf("symmetric transfer error = %v, want ~0", e)
	}
	// A grossly wrong correspondence has large error.
	if e := SymmetricTransferError(hGT, src[0], core.Point2d{X: 500, Y: 500}); e < 100 {
		t.Fatalf("expected large error for outlier, got %v", e)
	}
}

func TestFindHomographyRANSAC(t *testing.T) {
	// 10 inliers plus 3 outliers.
	src := []core.Point2d{
		{X: 0, Y: 0}, {X: 100, Y: 0}, {X: 0, Y: 100}, {X: 100, Y: 100},
		{X: 50, Y: 20}, {X: 20, Y: 80}, {X: 75, Y: 60}, {X: 33, Y: 44},
		{X: 90, Y: 10}, {X: 15, Y: 95},
		{X: 60, Y: 60}, {X: 25, Y: 25}, {X: 80, Y: 40},
	}
	dst := make([]core.Point2d, len(src))
	for i, p := range src {
		dst[i] = ApplyHomography(hGT, p)
	}
	// Corrupt the last three destinations into outliers.
	dst[10] = core.Point2d{X: 5, Y: 500}
	dst[11] = core.Point2d{X: 400, Y: 3}
	dst[12] = core.Point2d{X: 250, Y: 250}

	res := FindHomographyRANSAC(src, dst, 2.0, 500, DefaultRANSACSeed)
	if !res.Ok {
		t.Fatal("RANSAC homography failed")
	}
	if res.NumInliers != 10 {
		t.Fatalf("inliers = %d, want 10", res.NumInliers)
	}
	for i := 0; i < 10; i++ {
		if !res.Inliers[i] {
			t.Fatalf("point %d should be an inlier", i)
		}
	}
	for i := 10; i < 13; i++ {
		if res.Inliers[i] {
			t.Fatalf("point %d should be an outlier", i)
		}
	}
	if d := frob(res.Model, hGT); d > 1e-4 {
		t.Fatalf("RANSAC H differs from ground truth, frob=%v", d)
	}
}

func TestPerspectiveTransformRoundTrip(t *testing.T) {
	src := hSrcPoints()
	fwd := PerspectiveTransform(hGT, src)
	Hinv, ok := Mat3Inverse(hGT)
	if !ok {
		t.Fatal("H not invertible")
	}
	back := PerspectiveTransform(Hinv, fwd)
	for i := range src {
		if !approx(back[i].X, src[i].X, 1e-6) || !approx(back[i].Y, src[i].Y, 1e-6) {
			t.Fatalf("round trip failed at %d: %v vs %v", i, back[i], src[i])
		}
	}
}
