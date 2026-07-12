package tracking

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// motionSequence describes a synthetic clip: the object starts at (startX,
// startY) and moves by (dx, dy) each frame.
type motionSequence struct {
	w, h, half     int
	startX, startY int
	dx, dy         int
	frames         int
}

func (s motionSequence) center(i int) (int, int) {
	return s.startX + s.dx*i, s.startY + s.dy*i
}

func (s motionSequence) initBox() cv.Rect {
	return cv.Rect{X: s.startX - s.half, Y: s.startY - s.half, Width: 2 * s.half, Height: 2 * s.half}
}

// runAppearanceTracker drives an appearance tracker (gray frames) over a motion
// sequence and asserts that every reported box centre stays within tol pixels of
// ground truth.
func runAppearanceTracker(t *testing.T, name string, tr Tracker, s motionSequence, tol float64) {
	t.Helper()
	cx, cy := s.center(0)
	tr.Init(grayFrame(s.w, s.h, cx, cy, s.half), s.initBox())
	for i := 1; i < s.frames; i++ {
		gx, gy := s.center(i)
		box, ok := tr.Update(grayFrame(s.w, s.h, gx, gy, s.half))
		if !ok {
			t.Fatalf("%s: frame %d reported low confidence", name, i)
		}
		bx, by := boxCenter(box)
		if err := math.Hypot(bx-float64(gx), by-float64(gy)); err > tol {
			t.Errorf("%s: frame %d centre (%.1f,%.1f) off ground truth (%d,%d) by %.2f px (tol %.1f)",
				name, i, bx, by, gx, gy, err, tol)
		}
	}
}

func TestTrackerTemplateFollowsPatch(t *testing.T) {
	s := motionSequence{w: 50, h: 50, half: 8, startX: 16, startY: 16, dx: 2, dy: 1, frames: 6}
	runAppearanceTracker(t, "TrackerTemplate", NewTrackerTemplate(), s, 2.0)
}

func TestTrackerKCFFollowsPatch(t *testing.T) {
	s := motionSequence{w: 50, h: 50, half: 8, startX: 16, startY: 16, dx: 2, dy: 1, frames: 6}
	runAppearanceTracker(t, "TrackerKCF", NewTrackerKCF(), s, 2.0)
}

func TestTrackerMedianFlowFollowsPatch(t *testing.T) {
	s := motionSequence{w: 50, h: 50, half: 8, startX: 16, startY: 16, dx: 2, dy: 1, frames: 6}
	runAppearanceTracker(t, "TrackerMedianFlow", NewTrackerMedianFlow(), s, 3.0)
}

func TestTrackerMedianFlowHandlesScale(t *testing.T) {
	tr := NewTrackerMedianFlow()
	const w, h, cx, cy = 60, 60, 30, 30
	// Object grows from half 10 to half 12 (about +20%) between two frames.
	tr.Init(grayFrame(w, h, cx, cy, 10), cv.Rect{X: cx - 10, Y: cy - 10, Width: 20, Height: 20})
	box, ok := tr.Update(grayFrame(w, h, cx, cy, 12))
	if !ok {
		t.Fatal("TrackerMedianFlow: scale frame reported low confidence")
	}
	bx, by := boxCenter(box)
	if err := math.Hypot(bx-cx, by-cy); err > 3 {
		t.Errorf("centre (%.1f,%.1f) drifted from (%d,%d) by %.2f px", bx, by, cx, cy, err)
	}
	scale := float64(box.Width) / 20.0
	if scale < 1.05 || scale > 1.6 {
		t.Errorf("estimated scale %.3f not in [1.05,1.6] for a ~20%% enlargement", scale)
	}
}

func TestMeanShiftConvergesToMode(t *testing.T) {
	prob := blobProb(60, 60, 40, 25, 5, 5)
	// Start the window off-centre; it should walk to the blob's mode.
	win := MeanShift(prob, cv.Rect{X: 20, Y: 30, Width: 16, Height: 16}, 20)
	cx, cy := boxCenter(win)
	if err := math.Hypot(cx-40, cy-25); err > 2 {
		t.Errorf("MeanShift converged to (%.1f,%.1f), want near (40,25); off by %.2f px", cx, cy, err)
	}
}

func TestMeanShiftEmptyProbabilityStays(t *testing.T) {
	prob := cv.NewMat(30, 30, 1) // all zero
	start := cv.Rect{X: 5, Y: 5, Width: 8, Height: 8}
	win := MeanShift(prob, start, 10)
	if win != start {
		t.Errorf("MeanShift on empty probability moved the window: got %+v, want %+v", win, start)
	}
}

func TestCamShiftAdaptsWindow(t *testing.T) {
	// Blob elongated along x, so the fitted rectangle's centre matches the mode,
	// the major axis exceeds the minor axis, and the window is non-degenerate.
	prob := blobProb(60, 60, 30, 30, 7, 3)
	rr, win := CamShift(prob, cv.Rect{X: 20, Y: 24, Width: 20, Height: 12}, 20)
	if err := math.Hypot(rr.CenterX-30, rr.CenterY-30); err > 2.5 {
		t.Errorf("CamShift centre (%.1f,%.1f), want near (30,30); off by %.2f px", rr.CenterX, rr.CenterY, err)
	}
	if rr.Width <= 0 || rr.Height <= 0 {
		t.Errorf("CamShift produced a degenerate rectangle: %+v", rr)
	}
	// The major axis (Width) should exceed the minor axis for an elongated blob.
	if rr.Width <= rr.Height {
		t.Errorf("CamShift major axis %.2f not larger than minor axis %.2f", rr.Width, rr.Height)
	}
	wcx, wcy := boxCenter(win)
	if err := math.Hypot(wcx-30, wcy-30); err > 3 {
		t.Errorf("CamShift window centre (%.1f,%.1f) off mode by %.2f px", wcx, wcy, err)
	}
}

func TestMeanShiftTrackerFollowsColorBlob(t *testing.T) {
	tr := NewMeanShiftTracker()
	const w, h, half = 50, 50, 7
	sx, sy := 12, 12
	tr.Init(colorFrame(w, h, sx, sy, half), cv.Rect{X: sx - half, Y: sy - half, Width: 2 * half, Height: 2 * half})
	for i := 1; i < 6; i++ {
		gx := sx + 3*i
		gy := sy + 2*i
		box, ok := tr.Update(colorFrame(w, h, gx, gy, half))
		if !ok {
			t.Fatalf("MeanShiftTracker: frame %d reported no probability mass", i)
		}
		bx, by := boxCenter(box)
		if err := math.Hypot(bx-float64(gx), by-float64(gy)); err > 3 {
			t.Errorf("MeanShiftTracker: frame %d centre (%.1f,%.1f) off (%d,%d) by %.2f px", i, bx, by, gx, gy, err)
		}
	}
}

func TestCamShiftTrackerFollowsColorBlob(t *testing.T) {
	tr := NewCamShiftTracker()
	const w, h, half = 50, 50, 7
	sx, sy := 12, 12
	tr.Init(colorFrame(w, h, sx, sy, half), cv.Rect{X: sx - half, Y: sy - half, Width: 2 * half, Height: 2 * half})
	for i := 1; i < 6; i++ {
		gx := sx + 3*i
		gy := sy + 2*i
		box, ok := tr.Update(colorFrame(w, h, gx, gy, half))
		if !ok {
			t.Fatalf("CamShiftTracker: frame %d reported no probability mass", i)
		}
		bx, by := boxCenter(box)
		if err := math.Hypot(bx-float64(gx), by-float64(gy)); err > 3.5 {
			t.Errorf("CamShiftTracker: frame %d centre (%.1f,%.1f) off (%d,%d) by %.2f px", i, bx, by, gx, gy, err)
		}
	}
	rr := tr.RotatedRect()
	if rr.Width <= 0 || rr.Height <= 0 {
		t.Errorf("CamShiftTracker rotated rect degenerate: %+v", rr)
	}
}

func TestLKTrackRecoversTranslation(t *testing.T) {
	prev := grayFrame(40, 40, 20, 20, 8)
	cur := grayFrame(40, 40, 22, 21, 8) // shifted by (2,1)
	// A point on a strong corner feature (top-left feature at ~ -0.55*half).
	px := 20 + int(math.Round(-0.55*8))
	py := 20 + int(math.Round(-0.55*8))
	fx, fy, ok := lkTrack(prev, cur, float64(px), float64(py), 5, 30)
	if !ok {
		t.Fatal("lkTrack reported ill-conditioned window on a corner feature")
	}
	if err := math.Hypot(fx-float64(px+2), fy-float64(py+1)); err > 0.75 {
		t.Errorf("lkTrack found (%.2f,%.2f), want near (%d,%d); off by %.2f px", fx, fy, px+2, py+1, err)
	}
}

func TestLKTrackRejectsFlatRegion(t *testing.T) {
	prev := grayFrame(40, 40, 20, 20, 8)
	cur := grayFrame(40, 40, 22, 21, 8)
	// The far corner of the image is uniform background: no gradient.
	if _, _, ok := lkTrack(prev, cur, 3, 3, 4, 20); ok {
		t.Error("lkTrack should report ill-conditioned on a flat background region")
	}
}

func TestUpdateBeforeInitPanics(t *testing.T) {
	trackers := map[string]Tracker{
		"template":   NewTrackerTemplate(),
		"kcf":        NewTrackerKCF(),
		"medianflow": NewTrackerMedianFlow(),
		"meanshift":  NewMeanShiftTracker(),
		"camshift":   NewCamShiftTracker(),
	}
	for name, tr := range trackers {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: Update before Init did not panic", name)
				}
			}()
			tr.Update(grayFrame(20, 20, 10, 10, 4))
		}()
	}
}

func TestMedianFlowReportsFailureOnBlankFrame(t *testing.T) {
	tr := NewTrackerMedianFlow()
	tr.Init(grayFrame(40, 40, 20, 20, 8), cv.Rect{X: 12, Y: 12, Width: 16, Height: 16})
	blank := cv.NewMat(40, 40, 1) // featureless: no point survives
	box, ok := tr.Update(blank)
	if ok {
		t.Error("MedianFlow should report failure when no points can be tracked")
	}
	if (box != cv.Rect{X: 12, Y: 12, Width: 16, Height: 16}) {
		t.Errorf("MedianFlow should freeze the box on failure, got %+v", box)
	}
}

func TestHueTrackerRejectsGrayscale(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("MeanShiftTracker.Init should panic on a grayscale frame")
		}
	}()
	NewMeanShiftTracker().Init(cv.NewMat(20, 20, 1), cv.Rect{X: 5, Y: 5, Width: 5, Height: 5})
}

func TestMedianEvenAndEmpty(t *testing.T) {
	if got := median(nil); got != 0 {
		t.Errorf("median(nil) = %v, want 0", got)
	}
	if got := median([]float64{4, 1, 3, 2}); got != 2.5 {
		t.Errorf("median = %v, want 2.5", got)
	}
}

func TestClampRect(t *testing.T) {
	got := clampRect(cv.Rect{X: -5, Y: -5, Width: 100, Height: 100}, 40, 40)
	want := cv.Rect{X: 0, Y: 0, Width: 40, Height: 40}
	if got != want {
		t.Errorf("clampRect = %+v, want %+v", got, want)
	}
	got = clampRect(cv.Rect{X: 38, Y: 38, Width: 10, Height: 10}, 40, 40)
	want = cv.Rect{X: 30, Y: 30, Width: 10, Height: 10}
	if got != want {
		t.Errorf("clampRect = %+v, want %+v", got, want)
	}
}
