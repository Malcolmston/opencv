package videoproc

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// grayFrom builds a single-channel Mat from row-major byte data.
func grayFrom(rows, cols int, data []uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	copy(m.Data, data)
	return m
}

// constGray builds a rows×cols single-channel Mat filled with v.
func constGray(rows, cols int, v uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	m.SetTo(v)
	return m
}

// noiseGray builds a deterministic pseudo-random single-channel Mat (a linear
// congruential sequence) whose local patches are unique, so block matching has
// a well-defined optimum.
func noiseGray(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	s := uint32(12345)
	for i := range m.Data {
		s = s*1664525 + 1013904223
		m.Data[i] = uint8(s >> 24)
	}
	return m
}

func TestAbsDiff(t *testing.T) {
	a := grayFrom(1, 3, []uint8{10, 200, 30})
	b := grayFrom(1, 3, []uint8{20, 100, 30})
	got := AbsDiff(a, b)
	want := []uint8{10, 100, 0}
	for i, w := range want {
		if got.Data[i] != w {
			t.Fatalf("AbsDiff[%d]=%d want %d", i, got.Data[i], w)
		}
	}
}

func TestFrameDifferenceAndThree(t *testing.T) {
	prev := grayFrom(1, 3, []uint8{0, 100, 100})
	cur := grayFrom(1, 3, []uint8{0, 100, 200})
	m := FrameDifference(prev, cur, 20)
	want := []uint8{0, 0, 255}
	for i, w := range want {
		if m.Data[i] != w {
			t.Fatalf("FrameDifference[%d]=%d want %d", i, m.Data[i], w)
		}
	}
	// three-frame: pixel 2 changes prev->cur but not cur->next; pixel 0 changes
	// in both steps.
	next := grayFrom(1, 3, []uint8{200, 100, 200})
	tf := ThreeFrameDifference(prev, cur, next, 20)
	// prev->cur diff: {0,0,255}; cur->next diff: {255,0,0}; AND -> {0,0,0}
	for i := 0; i < 3; i++ {
		if tf.Data[i] != 0 {
			t.Fatalf("ThreeFrameDifference[%d]=%d want 0", i, tf.Data[i])
		}
	}
}

func TestAccumulateAndMask(t *testing.T) {
	acc := cv.NewFloatMat(1, 2)
	Accumulate(grayFrom(1, 2, []uint8{10, 20}), acc)
	Accumulate(grayFrom(1, 2, []uint8{30, 40}), acc)
	if acc.Data[0] != 40 || acc.Data[1] != 60 {
		t.Fatalf("Accumulate acc=%v want [40 60]", acc.Data)
	}
	mean := AccumulatorToMat(acc, 2)
	if mean.Data[0] != 20 || mean.Data[1] != 30 {
		t.Fatalf("AccumulatorToMat=%v want [20 30]", mean.Data)
	}
	sq := cv.NewFloatMat(1, 1)
	AccumulateSquare(grayFrom(1, 1, []uint8{10}), sq)
	if sq.Data[0] != 100 {
		t.Fatalf("AccumulateSquare=%v want 100", sq.Data)
	}
	// weighted: start 0, alpha 0.5, add 100 -> 50
	w := cv.NewFloatMat(1, 1)
	AccumulateWeighted(grayFrom(1, 1, []uint8{100}), w, 0.5)
	if w.Data[0] != 50 {
		t.Fatalf("AccumulateWeighted=%v want 50", w.Data)
	}
}

func TestMotionEnergy(t *testing.T) {
	mask := grayFrom(1, 4, []uint8{0, 255, 0, 255})
	if n := CountMotionPixels(mask); n != 2 {
		t.Fatalf("CountMotionPixels=%d want 2", n)
	}
	if e := MotionEnergy(mask); e != 0.5 {
		t.Fatalf("MotionEnergy=%v want 0.5", e)
	}
}

func TestRunningAverageSubtractor(t *testing.T) {
	s := NewRunningAverageSubtractor(0.5, 20)
	a := constGray(2, 2, 50)
	m0 := s.Apply(a) // init -> all zero
	if CountMotionPixels(m0) != 0 {
		t.Fatalf("first mask not empty")
	}
	m1 := s.Apply(a) // identical -> zero
	if CountMotionPixels(m1) != 0 {
		t.Fatalf("identical mask not empty")
	}
	b := a.Clone()
	b.Set(0, 0, 0, 200) // one changed pixel
	m2 := s.Apply(b)
	if m2.At(0, 0, 0) != 255 {
		t.Fatalf("changed pixel not foreground")
	}
	if CountMotionPixels(m2) != 1 {
		t.Fatalf("expected exactly one foreground pixel, got %d", CountMotionPixels(m2))
	}
	if s.Background() == nil {
		t.Fatalf("nil background")
	}
}

func TestMedianBackgroundSubtractor(t *testing.T) {
	s := NewMedianBackgroundSubtractor(3, 10)
	a := constGray(2, 2, 50)
	s.Apply(a)
	s.Apply(a)
	s.Apply(a)
	if !s.Ready() {
		t.Fatalf("not ready")
	}
	bg := s.Background()
	if bg.At(0, 0, 0) != 50 {
		t.Fatalf("median background=%d want 50", bg.At(0, 0, 0))
	}
	b := a.Clone()
	b.Set(0, 0, 0, 200)
	m := s.Apply(b)
	if m.At(0, 0, 0) != 255 {
		t.Fatalf("changed pixel not foreground in median subtractor")
	}
	if CountMotionPixels(m) != 1 {
		t.Fatalf("expected one foreground pixel, got %d", CountMotionPixels(m))
	}
}

func TestMOGBackgroundSubtractor(t *testing.T) {
	s := NewMOGBackgroundSubtractor()
	a := constGray(3, 3, 50)
	for i := 0; i < 5; i++ {
		s.Apply(a)
	}
	// identical frame -> background everywhere
	m := s.Apply(a)
	if CountMotionPixels(m) != 0 {
		t.Fatalf("static frame produced %d foreground pixels", CountMotionPixels(m))
	}
	b := a.Clone()
	b.Set(1, 1, 0, 200)
	m2 := s.Apply(b)
	if m2.At(1, 1, 0) != 255 {
		t.Fatalf("changed pixel not foreground in MOG")
	}
	if s.Background().At(0, 0, 0) != 50 {
		t.Fatalf("MOG background=%d want 50", s.Background().At(0, 0, 0))
	}
}

func TestMotionHistory(t *testing.T) {
	h := NewMotionHistory(2, 2, 3)
	sil := grayFrom(2, 2, []uint8{255, 0, 0, 0})
	h.Update(sil, 1.0)
	if h.Image().At(0, 0) != 1.0 {
		t.Fatalf("mhi[0,0]=%v want 1.0", h.Image().At(0, 0))
	}
	if h.EnergyImage().At(0, 0, 0) != 255 {
		t.Fatalf("MEI missing motion")
	}
	// advance far past duration with no motion -> the stamp decays to zero
	h.Update(constGray(2, 2, 0), 10.0)
	if h.Image().At(0, 0) != 0 {
		t.Fatalf("mhi did not decay, got %v", h.Image().At(0, 0))
	}
}

func TestMotionGradientOrientation(t *testing.T) {
	// mhi(y,x) = x -> horizontal gradient, orientation 0 degrees.
	mhi := cv.NewFloatMat(5, 5)
	for y := 0; y < 5; y++ {
		for x := 0; x < 5; x++ {
			mhi.Data[y*5+x] = float64(x)
		}
	}
	orient, mask := MotionGradient(mhi, 0.4, 2.0)
	// interior pixel gradient magnitude is 1, orientation 0.
	if mask.At(2, 2, 0) != 255 {
		t.Fatalf("interior gradient not flagged valid")
	}
	if math.Abs(orient.At(2, 2)) > 1e-9 {
		t.Fatalf("orientation=%v want 0", orient.At(2, 2))
	}
	g, ok := GlobalMotionOrientation(orient, mask)
	if !ok {
		t.Fatalf("global orientation reported no valid gradients")
	}
	if math.Abs(g) > 1e-6 {
		t.Fatalf("global orientation=%v want ~0", g)
	}
}

func TestTemporalStats(t *testing.T) {
	f1 := grayFrom(1, 1, []uint8{10})
	f2 := grayFrom(1, 1, []uint8{20})
	f3 := grayFrom(1, 1, []uint8{30})
	frames := []*cv.Mat{f1, f2, f3}
	if v := TemporalAverage(frames).Data[0]; v != 20 {
		t.Fatalf("TemporalAverage=%d want 20", v)
	}
	if v := TemporalMedian(frames).Data[0]; v != 20 {
		t.Fatalf("TemporalMedian=%d want 20", v)
	}
	if v := TemporalMinimum(frames).Data[0]; v != 10 {
		t.Fatalf("TemporalMinimum=%d want 10", v)
	}
	if v := TemporalMaximum(frames).Data[0]; v != 30 {
		t.Fatalf("TemporalMaximum=%d want 30", v)
	}
	if v := TemporalGaussian(frames, 1.0).Data[0]; v != 20 {
		t.Fatalf("TemporalGaussian=%d want 20", v)
	}
}

func TestExponentialMovingAverage(t *testing.T) {
	e := NewExponentialMovingAverage(0.5)
	if v := e.Update(grayFrom(1, 1, []uint8{100})).Data[0]; v != 100 {
		t.Fatalf("EMA init=%d want 100", v)
	}
	if v := e.Update(grayFrom(1, 1, []uint8{200})).Data[0]; v != 150 {
		t.Fatalf("EMA second=%d want 150", v)
	}
}

func TestMovingWindowFilter(t *testing.T) {
	w := NewMovingWindowFilter(2)
	if v := w.Push(grayFrom(1, 1, []uint8{10})).Data[0]; v != 10 {
		t.Fatalf("window push1=%d want 10", v)
	}
	if v := w.Push(grayFrom(1, 1, []uint8{20})).Data[0]; v != 15 {
		t.Fatalf("window push2=%d want 15", v)
	}
	if !w.Full() {
		t.Fatalf("window should be full")
	}
	if v := w.Push(grayFrom(1, 1, []uint8{30})).Data[0]; v != 25 {
		t.Fatalf("window push3=%d want 25", v)
	}
}

func TestShotBoundaryMetrics(t *testing.T) {
	a := constGray(4, 4, 0)
	b := constGray(4, 4, 255)
	if d := HistogramL1Difference(a, b); math.Abs(d-2) > 1e-9 {
		t.Fatalf("HistogramL1Difference=%v want 2", d)
	}
	if d := HistogramL1Difference(a, a); d != 0 {
		t.Fatalf("identical L1=%v want 0", d)
	}
	if d := HistogramChiSquare(a, b); math.Abs(d-2) > 1e-9 {
		t.Fatalf("HistogramChiSquare=%v want 2", d)
	}
	if r := PixelDifferenceRatio(a, b, 10); r != 1 {
		t.Fatalf("PixelDifferenceRatio=%v want 1", r)
	}
	if d := MeanIntensityDelta(a, b); d != 255 {
		t.Fatalf("MeanIntensityDelta=%v want 255", d)
	}
}

func TestEdgeChangeRatio(t *testing.T) {
	// left half dark, right half bright -> a vertical edge.
	edged := cv.NewMat(4, 4, 1)
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			if x >= 2 {
				edged.Set(y, x, 0, 255)
			}
		}
	}
	if r := EdgeChangeRatio(edged, edged, 50); r != 0 {
		t.Fatalf("identical edge frames ratio=%v want 0", r)
	}
	uniform := constGray(4, 4, 0)
	if r := EdgeChangeRatio(edged, uniform, 50); r != 1 {
		t.Fatalf("edge->uniform ratio=%v want 1", r)
	}
}

func TestDetectShotBoundaries(t *testing.T) {
	a := constGray(3, 3, 0)
	b := constGray(3, 3, 255)
	frames := []*cv.Mat{a, a, b, a}
	cuts := DetectShotBoundaries(frames, 1.0, nil)
	if len(cuts) != 2 || cuts[0] != 2 || cuts[1] != 3 {
		t.Fatalf("cuts=%v want [2 3]", cuts)
	}
}

func TestBlendAndCrossFade(t *testing.T) {
	a := grayFrom(1, 1, []uint8{0})
	b := grayFrom(1, 1, []uint8{100})
	if v := BlendFrames(a, b, 0).Data[0]; v != 0 {
		t.Fatalf("blend t=0 =%d want 0", v)
	}
	if v := BlendFrames(a, b, 1).Data[0]; v != 100 {
		t.Fatalf("blend t=1 =%d want 100", v)
	}
	if v := BlendFrames(a, b, 0.5).Data[0]; v != 50 {
		t.Fatalf("blend t=0.5 =%d want 50", v)
	}
	cf := CrossFade(a, b, 1)
	if len(cf) != 1 || cf[0].Data[0] != 50 {
		t.Fatalf("crossfade=%v want single 50", cf)
	}
}

func TestWarpByFlowAndInterpolate(t *testing.T) {
	a := noiseGray(8, 8)
	b := noiseGray(8, 8)
	zero := NewFlowField(8, 8)
	// zero flow -> identity
	w := WarpByFlow(a, zero, 1.0)
	for i := range a.Data {
		if w.Data[i] != a.Data[i] {
			t.Fatalf("zero-flow warp changed pixel %d", i)
		}
	}
	// zero flow interpolation == plain blend
	got := InterpolateFlow(a, b, zero, 0.5)
	want := BlendFrames(a, b, 0.5)
	for i := range got.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("zero-flow interpolate != blend at %d", i)
		}
	}
}

func TestWarpTranslate(t *testing.T) {
	src := grayFrom(1, 3, []uint8{10, 20, 30})
	out := WarpTranslate(src, 1, 0) // out(x)=src(x-1) clamped
	want := []uint8{10, 10, 20}
	for i, wv := range want {
		if out.Data[i] != wv {
			t.Fatalf("WarpTranslate[%d]=%d want %d", i, out.Data[i], wv)
		}
	}
}

func TestSmoothTrajectory(t *testing.T) {
	traj := []PointF{{X: 0}, {X: 3}, {X: 0}}
	sm := SmoothTrajectory(traj, 1)
	// index1 window = (0+3+0)/3 = 1
	if math.Abs(sm[1].X-1) > 1e-9 {
		t.Fatalf("smoothed[1].X=%v want 1", sm[1].X)
	}
	// endpoints clamp: index0 = (0+3)/2 = 1.5
	if math.Abs(sm[0].X-1.5) > 1e-9 {
		t.Fatalf("smoothed[0].X=%v want 1.5", sm[0].X)
	}
}

func TestSampleDenseGrid(t *testing.T) {
	pts := SampleDenseGrid(10, 10, 5)
	// y in {2,7}, x in {2,7} -> 4 points
	if len(pts) != 4 {
		t.Fatalf("grid points=%d want 4", len(pts))
	}
}

func TestTrackPointsRecoversShift(t *testing.T) {
	prev := noiseGray(40, 40)
	cur := WarpTranslate(prev, 2, 0) // content moves +2 in x
	pts := []PointF{{X: 20, Y: 20}}
	tracked, valid := TrackPoints(prev, cur, pts, 5, 3, 25)
	if !valid[0] {
		t.Fatalf("point not tracked")
	}
	if math.Abs(tracked[0].X-22) > 1e-9 || math.Abs(tracked[0].Y-20) > 1e-9 {
		t.Fatalf("tracked=%v want (22,20)", tracked[0])
	}
}

func TestEstimateGlobalTranslation(t *testing.T) {
	prev := noiseGray(48, 48)
	cur := WarpTranslate(prev, 2, 0)
	dx, dy, ok := EstimateGlobalTranslation(prev, cur, 100, 5)
	if !ok {
		t.Fatalf("estimation failed")
	}
	if math.Abs(dx-2) > 0.5 || math.Abs(dy) > 0.5 {
		t.Fatalf("estimated (%.2f,%.2f) want ~(2,0)", dx, dy)
	}
}

func TestTrajectoryMethods(t *testing.T) {
	tr := &Trajectory{Points: []PointF{{X: 0, Y: 0}, {X: 3, Y: 4}}}
	if tr.Length() != 2 {
		t.Fatalf("length=%d want 2", tr.Length())
	}
	dx, dy := tr.Displacement()
	if dx != 3 || dy != 4 {
		t.Fatalf("displacement=(%v,%v) want (3,4)", dx, dy)
	}
	if math.Abs(tr.TotalLength()-5) > 1e-9 {
		t.Fatalf("total length=%v want 5", tr.TotalLength())
	}
}

func TestDenseTrajectorySampler(t *testing.T) {
	prev := noiseGray(40, 40)
	cur := WarpTranslate(prev, 2, 0)
	s := NewDenseTrajectorySampler(8, 2)
	if s.Feed(prev) != nil {
		t.Fatalf("first feed should return nil")
	}
	completed := s.Feed(cur)
	if len(completed) == 0 {
		t.Fatalf("no completed trajectories")
	}
	// at least one interior trajectory should show the +2 x displacement.
	found := false
	for _, tr := range completed {
		dx, dy := tr.Displacement()
		if math.Abs(dx-2) < 1e-9 && math.Abs(dy) < 1e-9 {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("no trajectory recovered the (2,0) shift")
	}
}

func TestFlowFieldSetAt(t *testing.T) {
	f := NewFlowField(3, 3)
	f.Set(1, 2, 1.5, -2.5)
	dx, dy := f.At(1, 2)
	if dx != 1.5 || dy != -2.5 {
		t.Fatalf("flow At=(%v,%v) want (1.5,-2.5)", dx, dy)
	}
}

// BenchmarkStabilizer exercises the heaviest routine: the online stabilizer,
// which per frame runs corner detection, block-matching feature tracking and a
// bilinear compensation warp.
func BenchmarkStabilizer(b *testing.B) {
	base := noiseGray(120, 120)
	frames := make([]*cv.Mat, 8)
	for i := range frames {
		frames[i] = WarpTranslate(base, float64(i%3), float64((i % 2)))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := NewStabilizer(0.5)
		for _, f := range frames {
			s.Stabilize(f)
		}
	}
}
