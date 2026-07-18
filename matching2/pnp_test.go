package matching2

import (
	"math"
	"testing"

	"github.com/malcolmston/opencv/core"
)

func pnpWorld() []core.Point3d {
	return []core.Point3d{
		{X: -1, Y: -1, Z: 0.2}, {X: 1, Y: -1, Z: -0.3}, {X: 1, Y: 1, Z: 0.4},
		{X: -1, Y: 1, Z: -0.1}, {X: 0, Y: 0, Z: 0.6}, {X: 0.5, Y: -0.5, Z: 0.1},
		{X: -0.5, Y: 0.5, Z: -0.4}, {X: 0.8, Y: 0.2, Z: 0.3},
	}
}

func TestSolvePnPDLT(t *testing.T) {
	world := pnpWorld()
	Rgt := RodriguesToMatrix([3]float64{0.1, -0.2, 0.15})
	tgt := [3]float64{0.3, -0.2, 8}
	img := ProjectPoints(world, Rgt, tgt, testK)

	R, tvec, ok := SolvePnPDLT(world, img, testK)
	if !ok {
		t.Fatal("SolvePnPDLT failed")
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if !approx(R[i][j], Rgt[i][j], 1e-4) {
				t.Fatalf("R differs at %d,%d: %v vs %v", i, j, R[i][j], Rgt[i][j])
			}
		}
		if !approx(tvec[i], tgt[i], 1e-4) {
			t.Fatalf("t differs at %d: %v vs %v", i, tvec[i], tgt[i])
		}
	}
	if e := MeanReprojectionError(world, img, R, tvec, testK); e > 1e-4 {
		t.Fatalf("mean reprojection error = %v", e)
	}
}

func TestSolvePnPRansac(t *testing.T) {
	world := append(pnpWorld(),
		core.Point3d{X: 0.3, Y: -0.9, Z: 0.5},
		core.Point3d{X: -0.7, Y: -0.2, Z: -0.2},
		core.Point3d{X: 0.9, Y: 0.9, Z: 0.1},
	)
	Rgt := RodriguesToMatrix([3]float64{0.05, 0.1, -0.05})
	tgt := [3]float64{0.1, 0.2, 7}
	img := ProjectPoints(world, Rgt, tgt, testK)
	// Corrupt two observations.
	img[9] = core.Point2d{X: 10, Y: 10}
	img[10] = core.Point2d{X: 600, Y: 5}

	res := SolvePnPRansac(world, img, testK, 2.0, 800, DefaultRANSACSeed)
	if !res.Ok {
		t.Fatal("SolvePnPRansac failed")
	}
	if res.Inliers[9] || res.Inliers[10] {
		t.Fatalf("outliers flagged as inliers")
	}
	if res.NumInliers < 9 {
		t.Fatalf("inliers = %d, want >= 9", res.NumInliers)
	}
}

func TestRodriguesRoundTrip(t *testing.T) {
	vecs := [][3]float64{
		{0, 0, 0},
		{0.1, -0.2, 0.3},
		{math.Pi / 2, 0, 0},
		{0, 0, math.Pi - 0.01},
	}
	for _, v := range vecs {
		R := RodriguesToMatrix(v)
		if !approx(Mat3Det(R), 1, 1e-9) {
			t.Fatalf("Rodrigues R det = %v, want 1", Mat3Det(R))
		}
		back := MatrixToRodrigues(R)
		R2 := RodriguesToMatrix(back)
		// Compare rotation matrices rather than vectors (axis-angle is not unique).
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				if !approx(R[i][j], R2[i][j], 1e-6) {
					t.Fatalf("Rodrigues round trip failed for %v", v)
				}
			}
		}
	}
}
