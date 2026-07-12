package tracking

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// runConfTracker drives a ConfidenceTracker over a motion sequence, asserting
// every centre stays within tol of ground truth and every reported confidence is
// finite. It returns the confidence of the final frame.
func runConfTracker(t *testing.T, name string, tr ConfidenceTracker, s motionSequence, tol float64) float64 {
	t.Helper()
	cx, cy := s.center(0)
	tr.Init(grayFrame(s.w, s.h, cx, cy, s.half), s.initBox())
	var lastConf float64
	for i := 1; i < s.frames; i++ {
		gx, gy := s.center(i)
		box, conf := tr.UpdateConfidence(grayFrame(s.w, s.h, gx, gy, s.half))
		lastConf = conf
		if math.IsNaN(conf) || math.IsInf(conf, 0) {
			t.Fatalf("%s: frame %d non-finite confidence %v", name, i, conf)
		}
		bx, by := boxCenter(box)
		if err := math.Hypot(bx-float64(gx), by-float64(gy)); err > tol {
			t.Errorf("%s: frame %d centre (%.1f,%.1f) off ground truth (%d,%d) by %.2f px (tol %.1f)",
				name, i, bx, by, gx, gy, err, tol)
		}
	}
	return lastConf
}

// bigSeq is the standard synthetic clip used by the new trackers: a well-textured
// patch on an 80×80 frame moving (2,1) px/frame for 8 frames.
func bigSeq() motionSequence {
	return motionSequence{w: 80, h: 80, half: 10, startX: 24, startY: 24, dx: 2, dy: 1, frames: 8}
}

func TestTrackerMOSSEFollowsPatch(t *testing.T) {
	runConfTracker(t, "TrackerMOSSE", NewTrackerMOSSE(), bigSeq(), 2.0)
}

func TestTrackerDCFFollowsPatch(t *testing.T) {
	runConfTracker(t, "TrackerDCF", NewTrackerDCF(), bigSeq(), 2.0)
}

func TestTrackerKCFHOGFollowsPatch(t *testing.T) {
	runConfTracker(t, "TrackerKCFHOG", NewTrackerKCFHOG(), bigSeq(), 2.0)
}

func TestTrackerCSRTFollowsPatch(t *testing.T) {
	runConfTracker(t, "TrackerCSRT", NewTrackerCSRT(), bigSeq(), 2.0)
}

func TestTrackerBoostingFollowsPatch(t *testing.T) {
	runConfTracker(t, "TrackerBoosting", NewTrackerBoosting(), bigSeq(), 2.0)
}

func TestTrackerMILFollowsPatch(t *testing.T) {
	// Random generalised-Haar features localise more coarsely, so allow 4 px.
	runConfTracker(t, "TrackerMIL", NewTrackerMIL(), bigSeq(), 4.0)
}

func TestTrackerTLDFollowsPatch(t *testing.T) {
	runConfTracker(t, "TrackerTLD", NewTrackerTLD(), bigSeq(), 2.0)
}

// TestTrackerDCFHandlesScale checks that the FFT tracker enlarges its box when
// the object grows, while keeping its centre.
func TestTrackerDCFHandlesScale(t *testing.T) {
	tr := NewTrackerDCF()
	const w, h, cx, cy = 120, 120, 60, 60
	tr.Init(grayFrame(w, h, cx, cy, 12), cv.Rect{X: cx - 12, Y: cy - 12, Width: 24, Height: 24})
	var box cv.Rect
	for i := 1; i <= 8; i++ {
		half := 12 + i // grows from 12 to 20 (~1.67×)
		var ok bool
		box, ok = tr.Update(grayFrame(w, h, cx, cy, half))
		if !ok {
			t.Fatalf("TrackerDCF: scale frame %d low confidence", i)
		}
		bx, by := boxCenter(box)
		if err := math.Hypot(bx-cx, by-cy); err > 4 {
			t.Errorf("TrackerDCF: scale frame %d centre (%.1f,%.1f) drifted from (%d,%d) by %.2f", i, bx, by, cx, cy, err)
		}
	}
	if scale := float64(box.Width) / 24.0; scale < 1.15 {
		t.Errorf("TrackerDCF: box grew to scale %.3f, want >= 1.15 for a growing object", scale)
	}
}

// TestTrackerKCFHOGHandlesScale checks the HOG tracker also adapts scale upward.
func TestTrackerKCFHOGHandlesScale(t *testing.T) {
	tr := NewTrackerKCFHOG()
	const w, h, cx, cy = 120, 120, 60, 60
	tr.Init(grayFrame(w, h, cx, cy, 12), cv.Rect{X: cx - 12, Y: cy - 12, Width: 24, Height: 24})
	var box cv.Rect
	for i := 1; i <= 8; i++ {
		half := 12 + i
		box, _ = tr.Update(grayFrame(w, h, cx, cy, half))
		bx, by := boxCenter(box)
		if err := math.Hypot(bx-cx, by-cy); err > 5 {
			t.Errorf("TrackerKCFHOG: scale frame %d centre off by %.2f", i, err)
		}
	}
	if scale := float64(box.Width) / 24.0; scale < 1.1 {
		t.Errorf("TrackerKCFHOG: box grew to scale %.3f, want >= 1.1", scale)
	}
}

// TestTrackerTLDReDetects verifies the detector re-acquires the object after an
// occluding blank frame and reappearance elsewhere — the defining TLD behaviour.
func TestTrackerTLDReDetects(t *testing.T) {
	tr := NewTrackerTLD()
	const w, h, half = 110, 110, 10
	tr.Init(grayFrame(w, h, 30, 30, half), cv.Rect{X: 20, Y: 20, Width: 20, Height: 20})
	for i := 1; i <= 5; i++ {
		tr.Update(grayFrame(w, h, 30+3*i, 30+2*i, half))
	}
	// Occlusion: uniform frame; the tracker should report failure.
	blank := cv.NewMat(h, w, 1)
	for i := range blank.Data {
		blank.Data[i] = 30
	}
	if _, ok := tr.Update(blank); ok {
		t.Error("TrackerTLD: expected failure on a blank (occluded) frame")
	}
	// Reappear far from the last position; the detector must re-detect it.
	box, ok := tr.Update(grayFrame(w, h, 75, 65, half))
	if !ok {
		t.Fatal("TrackerTLD: failed to re-detect the reappeared object")
	}
	bx, by := boxCenter(box)
	if err := math.Hypot(bx-75, by-65); err > 6 {
		t.Errorf("TrackerTLD: re-detected at (%.1f,%.1f), want near (75,65); off %.2f px", bx, by, err)
	}
}

// TestMultiTracker follows two objects of different tracker types at once.
func TestMultiTracker(t *testing.T) {
	w, h := 100, 100
	frame := func(ax, ay, bx, by int) *cv.Mat {
		m := cv.NewMat(h, w, 1)
		for i := range m.Data {
			m.Data[i] = 30
		}
		stamp := func(cx, cy int) {
			p := grayFrame(w, h, cx, cy, 8)
			for y := cy - 12; y <= cy+12; y++ {
				for x := cx - 12; x <= cx+12; x++ {
					if x >= 0 && x < w && y >= 0 && y < h {
						m.Data[y*w+x] = p.Data[y*w+x]
					}
				}
			}
		}
		stamp(ax, ay)
		stamp(bx, by)
		return m
	}
	mt := NewMultiTracker()
	mt.Add(NewTrackerMOSSE(), frame(25, 25, 70, 70), cv.Rect{X: 17, Y: 17, Width: 16, Height: 16})
	mt.Add(NewTrackerCSRT(), frame(25, 25, 70, 70), cv.Rect{X: 62, Y: 62, Width: 16, Height: 16})
	if mt.Len() != 2 {
		t.Fatalf("MultiTracker.Len = %d, want 2", mt.Len())
	}
	for i := 1; i <= 4; i++ {
		ax, ay := 25+2*i, 25+i
		bx, by := 70-2*i, 70-i
		boxes, confs := mt.UpdateConfidence(frame(ax, ay, bx, by))
		if len(boxes) != 2 || len(confs) != 2 {
			t.Fatalf("MultiTracker returned %d boxes, %d confs", len(boxes), len(confs))
		}
		c0x, c0y := boxCenter(boxes[0])
		if err := math.Hypot(c0x-float64(ax), c0y-float64(ay)); err > 3 {
			t.Errorf("object 0 frame %d off by %.2f", i, err)
		}
		c1x, c1y := boxCenter(boxes[1])
		if err := math.Hypot(c1x-float64(bx), c1y-float64(by)); err > 3 {
			t.Errorf("object 1 frame %d off by %.2f", i, err)
		}
	}
	if len(mt.Boxes()) != 2 {
		t.Errorf("MultiTracker.Boxes returned %d entries", len(mt.Boxes()))
	}
}

// TestMultiTrackerPlainTracker checks the confidence fallback for a plain
// Tracker (no ConfidenceTracker) yields 1/0 confidences.
func TestMultiTrackerPlainTracker(t *testing.T) {
	mt := NewMultiTracker()
	s := bigSeq()
	cx, cy := s.center(0)
	mt.Add(NewTrackerTemplate(), grayFrame(s.w, s.h, cx, cy, s.half), s.initBox())
	gx, gy := s.center(1)
	_, confs := mt.UpdateConfidence(grayFrame(s.w, s.h, gx, gy, s.half))
	if confs[0] != 0 && confs[0] != 1 {
		t.Errorf("plain-tracker confidence = %v, want 0 or 1", confs[0])
	}
}

// TestFFTRoundTrip checks IFFT2(FFT2(x)) recovers x.
func TestFFTRoundTrip(t *testing.T) {
	const n = 8
	src := NewComplexMat(n, n)
	for i := range src.Data {
		src.Data[i] = complex(float64((i*7)%13), float64((i*3)%5))
	}
	got := IFFT2(FFT2(src))
	for i := range src.Data {
		if d := got.Data[i] - src.Data[i]; math.Hypot(real(d), imag(d)) > 1e-9 {
			t.Fatalf("FFT round-trip mismatch at %d: got %v want %v", i, got.Data[i], src.Data[i])
		}
	}
}

// TestFFTMatchesDFT checks the fast transform against a direct DFT for a 1D row.
func TestFFTMatchesDFT(t *testing.T) {
	const n = 4
	m := NewComplexMat(1, n)
	vals := []float64{1, 2, 3, 4}
	for i, v := range vals {
		m.Data[i] = complex(v, 0)
	}
	f := FFT2(m)
	for k := 0; k < n; k++ {
		var acc complex128
		for j := 0; j < n; j++ {
			ang := -2 * math.Pi * float64(k*j) / float64(n)
			acc += complex(vals[j], 0) * complex(math.Cos(ang), math.Sin(ang))
		}
		if d := f.Data[k] - acc; math.Hypot(real(d), imag(d)) > 1e-9 {
			t.Errorf("FFT[%d] = %v, direct DFT = %v", k, f.Data[k], acc)
		}
	}
}

func TestNextPow2(t *testing.T) {
	cases := map[int]int{1: 1, 2: 2, 3: 4, 5: 8, 16: 16, 17: 32, 20: 32}
	for in, want := range cases {
		if got := NextPow2(in); got != want {
			t.Errorf("NextPow2(%d) = %d, want %d", in, got, want)
		}
	}
	if isPow2(0) || !isPow2(64) || isPow2(48) {
		t.Error("isPow2 classification wrong")
	}
}

func TestFFT2RejectsNonPow2(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("FFT2 should panic on non-power-of-two dimensions")
		}
	}()
	FFT2(NewComplexMat(3, 4))
}

// TestHannWindow checks the window peaks at the centre and vanishes at the edge.
func TestHannWindow(t *testing.T) {
	h := HannWindow2D(9, 9)
	if h[0] != 0 {
		t.Errorf("Hann corner = %v, want 0", h[0])
	}
	if c := h[4*9+4]; c < 0.99 {
		t.Errorf("Hann centre = %v, want ~1", c)
	}
}

// TestGaussianResponse checks the target peaks (value 1) at the grid centre for
// an odd size (integer centre).
func TestGaussianResponse(t *testing.T) {
	g := GaussianResponse(15, 15, 2)
	px, py, peak := peakLoc(g, 15, 15)
	if math.Abs(peak-1) > 1e-9 {
		t.Errorf("Gaussian peak = %v, want 1", peak)
	}
	if px != 7 || py != 7 {
		t.Errorf("Gaussian peak at (%d,%d), want centre (7,7)", px, py)
	}
}

// TestHOGCells checks a horizontal edge concentrates energy in one orientation
// bin and the histogram is unit-normalised.
func TestHOGCells(t *testing.T) {
	m := cv.NewMat(8, 8, 1)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			if y >= 4 {
				m.Data[y*8+x] = 200
			}
		}
	}
	planes, cr, cc := HOGCells(m, 4, 9)
	if cr != 2 || cc != 2 {
		t.Fatalf("HOG grid = %dx%d, want 2x2", cr, cc)
	}
	// L2 norm of each cell's histogram is ~1 (or 0 for a flat cell).
	for idx := 0; idx < cr*cc; idx++ {
		var ss float64
		for b := range planes {
			ss += planes[b][idx] * planes[b][idx]
		}
		if ss > 1e-9 && math.Abs(math.Sqrt(ss)-1) > 1e-3 {
			t.Errorf("cell %d L2 norm = %v, want ~1", idx, math.Sqrt(ss))
		}
	}
}

// TestNewTrackersUpdateBeforeInitPanic checks the new trackers panic if Update
// runs before Init.
func TestNewTrackersUpdateBeforeInitPanic(t *testing.T) {
	trackers := map[string]Tracker{
		"mosse":    NewTrackerMOSSE(),
		"dcf":      NewTrackerDCF(),
		"kcfhog":   NewTrackerKCFHOG(),
		"csrt":     NewTrackerCSRT(),
		"mil":      NewTrackerMIL(),
		"boosting": NewTrackerBoosting(),
		"tld":      NewTrackerTLD(),
	}
	for name, tr := range trackers {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s: Update before Init did not panic", name)
				}
			}()
			tr.Update(grayFrame(40, 40, 20, 20, 6))
		}()
	}
}

// TestCSRTChannelWeights checks the reliability weights are a valid distribution.
func TestCSRTChannelWeights(t *testing.T) {
	tr := NewTrackerCSRT()
	s := bigSeq()
	cx, cy := s.center(0)
	tr.Init(grayFrame(s.w, s.h, cx, cy, s.half), s.initBox())
	w := tr.ChannelWeights()
	if len(w) != csrtChannels {
		t.Fatalf("ChannelWeights len = %d, want %d", len(w), csrtChannels)
	}
	var sum float64
	for _, v := range w {
		if v < 0 {
			t.Errorf("negative channel weight %v", v)
		}
		sum += v
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Errorf("channel weights sum = %v, want 1", sum)
	}
}
