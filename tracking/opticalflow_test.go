package tracking

import (
	"math"
	"testing"
)

func TestCalcOpticalFlowLKTranslation(t *testing.T) {
	prev := synthTexture(48, 48, 0, 0)
	next := synthTexture(48, 48, 1, 0) // content shifted +1 in x
	pts := []Point2f{{X: 24, Y: 24}, {X: 20, Y: 28}}
	params := LKParams{WindowRadius: 6, MaxIterations: 30, Epsilon: 1e-3}
	nextPts, status := CalcOpticalFlowLK(prev, next, pts, params)
	for i, p := range pts {
		requireTrue(t, status[i], "point %d not tracked", i)
		du := nextPts[i].X - p.X
		dv := nextPts[i].Y - p.Y
		requireTrue(t, approx(du, 1.0, 0.2), "point %d du = %v, want ~1", i, du)
		requireTrue(t, approx(dv, 0.0, 0.2), "point %d dv = %v, want ~0", i, dv)
	}
}

func TestCalcOpticalFlowPyrLKLargeMotion(t *testing.T) {
	prev := synthTexture(64, 64, 0, 0)
	next := synthTexture(64, 64, 4, 3) // large-ish shift needs the pyramid
	pts := []Point2f{{X: 32, Y: 32}}
	nextPts, status := CalcOpticalFlowPyrLK(prev, next, pts, 3, DefaultLKParams())
	requireTrue(t, status[0], "point not tracked")
	du := nextPts[0].X - pts[0].X
	dv := nextPts[0].Y - pts[0].Y
	requireTrue(t, approx(du, 4.0, 0.6), "du = %v, want ~4", du)
	requireTrue(t, approx(dv, 3.0, 0.6), "dv = %v, want ~3", dv)
}

func TestBuildOpticalFlowPyramid(t *testing.T) {
	img := synthTexture(32, 40, 0, 0)
	pyr := BuildOpticalFlowPyramid(img, 3)
	requireTrue(t, len(pyr) == 4, "expected 4 levels, got %d", len(pyr))
	// Each level halves the size (rounding up).
	requireTrue(t, pyr[0].Rows == 32 && pyr[0].Cols == 40, "level 0 wrong size")
	requireTrue(t, pyr[1].Rows == 16 && pyr[1].Cols == 20, "level 1 wrong size")
	requireTrue(t, pyr[2].Rows == 8 && pyr[2].Cols == 10, "level 2 wrong size")
	requireTrue(t, pyr[3].Rows == 4 && pyr[3].Cols == 5, "level 3 wrong size")
}

func TestFlowFieldStats(t *testing.T) {
	f := NewFlowField(2, 2)
	f.SetFlow(0, 0, 3, 4)
	f.SetFlow(1, 1, 0, 0)
	if !approx(f.MagnitudeAt(0, 0), 5, 1e-9) {
		t.Fatalf("magnitude wrong")
	}
	if !approx(f.MaxMagnitude(), 5, 1e-9) {
		t.Fatalf("max magnitude wrong")
	}
	m := f.MeanFlow()
	if !approx(m.X, 0.75, 1e-9) || !approx(m.Y, 1.0, 1e-9) {
		t.Fatalf("mean flow = %v, want (0.75, 1)", m)
	}
	if got := f.At(0, 0); !approx(got.X, 3, 1e-9) || !approx(got.Y, 4, 1e-9) {
		t.Fatalf("At wrong: %v", got)
	}
}

func TestFlowFieldPanicsOutOfRange(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on out-of-range access")
		}
	}()
	f := NewFlowField(2, 2)
	_ = f.At(5, 5)
	_ = math.Inf(1)
}
