package structured_light

import (
	"math"
	"testing"
)

func TestJacobiEigenSymmetric(t *testing.T) {
	a := [][]float64{
		{4, 1, 2},
		{1, 3, 0},
		{2, 0, 5},
	}
	vals, vecs := jacobiEigenSymmetric(a)
	for k := range vals {
		// A v must equal λ v.
		for i := 0; i < 3; i++ {
			var av float64
			for j := 0; j < 3; j++ {
				av += a[i][j] * vecs[k][j]
			}
			if e := math.Abs(av - vals[k]*vecs[k][i]); e > 1e-9 {
				t.Fatalf("eigenpair %d fails A v = λ v at row %d: %.3e", k, i, e)
			}
		}
	}
}

// testRig returns a camera at the origin and a projector translated by baseline
// B along +x, both with intrinsics (f, cx, cy).
func testRig(f, cx, cy, baseline float64) (cam, proj CameraMatrix) {
	k := [3][3]float64{{f, 0, cx}, {0, f, cy}, {0, 0, 1}}
	id := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	cam = NewPinhole(k, id, [3]float64{0, 0, 0})
	proj = NewPinhole(k, id, [3]float64{-baseline, 0, 0})
	return cam, proj
}

func TestTriangulatePointExact(t *testing.T) {
	cam, proj := testRig(800, 320, 240, 0.2)
	want := [3]float64{0.1, -0.05, 1.2}
	uc, vc := cam.Project(want)
	up, vp := proj.Project(want)
	got := TriangulatePoint(cam, proj, uc, vc, up, vp)
	for i := 0; i < 3; i++ {
		if e := math.Abs(got[i] - want[i]); e > 1e-6 {
			t.Fatalf("triangulated coord %d = %.8f, want %.8f", i, got[i], want[i])
		}
	}
}

func TestTriangulatePlanarDepth(t *testing.T) {
	f, cx, cy, baseline := 800.0, 320.0, 240.0, 0.2
	cam, proj := testRig(f, cx, cy, baseline)
	const z0 = 1.5

	// Build a Decoded over a small camera-pixel grid by back-projecting each
	// integer camera pixel onto the plane Z=z0 and projecting to the projector.
	rows, cols := 240, 400
	dec := &Decoded{Rows: rows, Cols: cols, Col: make([]int, rows*cols), Row: make([]int, rows*cols), Mask: make([]bool, rows*cols)}
	for i := range dec.Col {
		dec.Col[i], dec.Row[i] = -1, -1
	}
	for y := 180; y < 200; y += 5 {
		for x := 200; x < 240; x += 5 {
			// Camera ray hits the plane at this world point.
			X := [3]float64{(float64(x) - cx) / f * z0, (float64(y) - cy) / f * z0, z0}
			up, vp := proj.Project(X)
			i := y*cols + x
			dec.Col[i] = int(math.Round(up))
			dec.Row[i] = int(math.Round(vp))
			dec.Mask[i] = true
		}
	}

	pc := Triangulate(dec, cam, proj)
	if pc.Len() == 0 {
		t.Fatal("no points triangulated")
	}
	maxZErr := 0.0
	for _, p := range pc.Points {
		if e := math.Abs(p[2] - z0); e > maxZErr {
			maxZErr = e
		}
	}
	if maxZErr > 0.05 {
		t.Fatalf("planar depth max Z error %.4f exceeds tolerance (should reproduce Z=%.2f)", maxZErr, z0)
	}
}
