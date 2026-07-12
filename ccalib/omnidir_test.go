package ccalib

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func testModel() OmniModel {
	return OmniModel{Fx: 280, Fy: 285, Cx: 320, Cy: 240, Xi: 1.05, K1: -0.02, K2: 0.01, P1: 0.001, P2: -0.0008}
}

func TestOmniModelHelpers(t *testing.T) {
	m := testModel()
	k := m.K()
	if k[0][0] != 280 || k[1][1] != 285 || k[0][2] != 320 || k[1][2] != 240 {
		t.Fatalf("K mismatch: %+v", k)
	}
	d := m.Dist()
	if len(d) != 4 || d[0] != -0.02 || d[3] != -0.0008 {
		t.Fatalf("Dist mismatch: %+v", d)
	}
	m2 := NewOmniModel(k, m.Xi, d)
	if math.Abs(m2.Fx-m.Fx) > 1e-9 || math.Abs(m2.Xi-m.Xi) > 1e-9 || math.Abs(m2.K2-m.K2) > 1e-9 {
		t.Fatalf("round-trip through NewOmniModel changed the model: %+v", m2)
	}
}

func TestProjectLiftRoundTrip(t *testing.T) {
	m := testModel()
	K := m.K()
	obj := [][3]float64{
		{0, 0, 4},
		{1.2, 0.6, 5},
		{-1.1, 0.8, 4.5},
		{0.4, -0.9, 3.5},
		{-0.7, -0.5, 6},
	}
	pix := Omnidir.ProjectPoints(obj, [3]float64{0, 0, 0}, [3]float64{0, 0, 0}, K, m.Xi, m.Dist())
	for i, p := range pix {
		if math.IsNaN(p[0]) {
			t.Fatalf("point %d unexpectedly not imaged", i)
		}
		dir, ok := liftToSphere(p[0], p[1], m.Xi, m.Fx, m.Fy, m.Cx, m.Cy, m.Skew, m.K1, m.K2, m.P1, m.P2)
		if !ok {
			t.Fatalf("lift failed for point %d", i)
		}
		want, _ := normalize3(obj[i])
		got, _ := normalize3(dir)
		ang := math.Acos(math.Min(1, math.Max(-1, dot3(want, got))))
		if ang > 1e-6 {
			t.Fatalf("point %d: lifted ray off by %g rad", i, ang)
		}
	}
}

func TestProjectPointsBehindMirror(t *testing.T) {
	m := testModel()
	// A point far behind the camera cannot be imaged by a forward mirror.
	pix := Omnidir.ProjectPoints([][3]float64{{0, 0, -1}}, [3]float64{0, 0, 0}, [3]float64{0, 0, 0}, m.K(), 0.0, nil)
	if !math.IsNaN(pix[0][0]) {
		t.Fatalf("expected NaN for a point behind a pinhole-like mirror, got %v", pix[0])
	}
}

func TestUndistortToPerspective(t *testing.T) {
	m := testModel()
	// Build a set of rays with a positive Z component (visible to a perspective
	// rectification), project them, then undistort and confirm we recover the
	// perspective pixels implied by the same rays through Knew.
	Knew := [3][3]float64{{250, 0, 320}, {0, 250, 240}, {0, 0, 1}}
	R := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	obj := [][3]float64{
		{0, 0, 4},
		{0.5, 0.3, 4},
		{-0.6, 0.2, 4},
		{0.2, -0.4, 4},
	}
	pix := Omnidir.ProjectPoints(obj, [3]float64{0, 0, 0}, [3]float64{0, 0, 0}, m.K(), m.Xi, m.Dist())
	undist := Omnidir.Undistort(pix, m.K(), m.Xi, m.Dist(), Knew, R)
	for i, X := range obj {
		wantU := Knew[0][0]*(X[0]/X[2]) + Knew[0][2]
		wantV := Knew[1][1]*(X[1]/X[2]) + Knew[1][2]
		if math.Abs(undist[i][0]-wantU) > 0.5 || math.Abs(undist[i][1]-wantV) > 0.5 {
			t.Fatalf("point %d: undistort gave (%.2f,%.2f) want (%.2f,%.2f)", i, undist[i][0], undist[i][1], wantU, wantV)
		}
	}
}

func TestInitUndistortRectifyMapConsistency(t *testing.T) {
	m := testModel()
	Knew := [3][3]float64{{240, 0, 160}, {0, 240, 120}, {0, 0, 1}}
	R := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	w, h := 320, 240
	mapX, mapY := Omnidir.InitUndistortRectifyMap(m.K(), m.Xi, m.Dist(), R, Knew, w, h, RectifyPerspective)
	if mapX.Rows != h || mapX.Cols != w || mapY.Rows != h {
		t.Fatalf("map dimensions wrong")
	}
	// For a central output pixel the source should be the forward optical axis,
	// i.e. near the omni principal point.
	cxOut, cyOut := 160, 120
	sx := mapX.At(cyOut, cxOut)
	sy := mapY.At(cyOut, cxOut)
	if math.Abs(sx-m.Cx) > 1.0 || math.Abs(sy-m.Cy) > 1.0 {
		t.Fatalf("centre maps to (%.2f,%.2f), want near (%.1f,%.1f)", sx, sy, m.Cx, m.Cy)
	}
}

func TestUndistortImageRuns(t *testing.T) {
	m := testModel()
	src := cv.NewMat(120, 160, 1)
	for y := 0; y < 120; y++ {
		for x := 0; x < 160; x++ {
			src.Set(y, x, 0, uint8((x*7+y*5)%256))
		}
	}
	Knew := [3][3]float64{{120, 0, 100}, {0, 120, 75}, {0, 0, 1}}
	R := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	for _, flag := range []RectifyFlag{RectifyPerspective, RectifyCylindrical, RectifyLongLat} {
		out := Omnidir.UndistortImage(src, m.K(), m.Xi, m.Dist(), Knew, R, 200, 150, flag)
		if out.Rows != 150 || out.Cols != 200 {
			t.Fatalf("flag %d: output is %dx%d, want 150x200", flag, out.Rows, out.Cols)
		}
	}
}
