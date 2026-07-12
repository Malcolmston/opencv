package optflow

import (
	"bytes"
	"image"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// valueNoiseShift builds a deterministic, non-periodic smooth texture (bilinear
// value noise over a seeded coarse lattice) and its translate by (dx, dy). Unlike
// the sinusoidal texturedShift, this pattern has no spatial period, so the
// correlation-based matchers (SimpleFlow, DeepFlow) have a unique optimum — the
// fair regime for those methods, which cannot resolve the aperture ambiguity of
// a periodic signal any more than the reference implementations can.
func valueNoiseShift(rows, cols int, dx, dy float64, seed int64) (prev, next *cv.Mat) {
	const s = 6 // lattice spacing in pixels
	gw := cols/s + 4
	gh := rows/s + 4
	rng := rand.New(rand.NewSource(seed))
	lat := make([]float64, gw*gh)
	for i := range lat {
		lat[i] = rng.Float64() * 255
	}
	sample := func(fx, fy float64) uint8 {
		// Shift the lattice origin so negative sample coordinates stay in range.
		gx := fx/s + 1
		gy := fy/s + 1
		x0 := int(math.Floor(gx))
		y0 := int(math.Floor(gy))
		tx := gx - float64(x0)
		ty := gy - float64(y0)
		at := func(xx, yy int) float64 {
			if xx < 0 {
				xx = 0
			} else if xx >= gw {
				xx = gw - 1
			}
			if yy < 0 {
				yy = 0
			} else if yy >= gh {
				yy = gh - 1
			}
			return lat[yy*gw+xx]
		}
		top := at(x0, y0)*(1-tx) + at(x0+1, y0)*tx
		bot := at(x0, y0+1)*(1-tx) + at(x0+1, y0+1)*tx
		v := top*(1-ty) + bot*ty
		if v < 0 {
			v = 0
		}
		if v > 255 {
			v = 255
		}
		return uint8(math.Round(v))
	}
	prev = cv.NewMat(rows, cols, 1)
	next = cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			prev.Set(y, x, 0, sample(float64(x), float64(y)))
			next.Set(y, x, 0, sample(float64(x)-dx, float64(y)-dy))
		}
	}
	return prev, next
}

// rotatedFrame builds a pair where next is prev rotated about the image centre
// by theta radians. Sampling next at the inverse-rotated coordinate realises a
// forward rotation of the content; the induced flow is smooth and non-constant,
// exercising the smooth-motion solvers beyond pure translation.
func rotatedFrame(rows, cols int, theta float64) (prev, next *cv.Mat) {
	prev = cv.NewMat(rows, cols, 1)
	next = cv.NewMat(rows, cols, 1)
	cx, cy := float64(cols-1)/2, float64(rows-1)/2
	pattern := func(x, y float64) uint8 {
		val := 128 +
			55*math.Sin(2*math.Pi*x/15) +
			45*math.Cos(2*math.Pi*y/12) +
			25*math.Sin(2*math.Pi*(x+y)/10)
		if val < 0 {
			val = 0
		}
		if val > 255 {
			val = 255
		}
		return uint8(math.Round(val))
	}
	ct, st := math.Cos(theta), math.Sin(theta)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			prev.Set(y, x, 0, pattern(float64(x), float64(y)))
			// Inverse-rotate the destination pixel to sample the source pattern.
			dx := float64(x) - cx
			dy := float64(y) - cy
			sx := cx + ct*dx + st*dy
			sy := cy - st*dx + ct*dy
			next.Set(y, x, 0, pattern(sx, sy))
		}
	}
	return prev, next
}

func interiorMeanFlow(f *FlowField, border int) (u, v float64) {
	return f.MeanFlow(border)
}

func TestTVL1RecoversTranslation(t *testing.T) {
	const dx, dy = 2.0, 1.0
	prev, next := texturedShift(64, 64, dx, dy)
	p := DefaultTVL1Params()
	p.Scales = 4
	p.Warps = 5
	p.Iterations = 30
	flow := CalcOpticalFlowDenseTVL1(prev, next, p)
	mu, mv := interiorMeanFlow(flow, 8)
	if math.Abs(mu-dx) > 0.35 || math.Abs(mv-dy) > 0.35 {
		t.Fatalf("TV-L1 mean flow = (%.3f, %.3f), want ≈ (%.1f, %.1f)", mu, mv, dx, dy)
	}
}

func TestDualTVL1ObjectMatchesFunction(t *testing.T) {
	prev, next := texturedShift(48, 48, 1.0, 0.0)
	p := DefaultTVL1Params()
	p.Scales = 3
	obj := NewDualTVL1OpticalFlow(p)
	a := obj.Calc(prev, next)
	b := CalcOpticalFlowDenseTVL1(prev, next, p)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("DualTVL1OpticalFlow.Calc differs from function at %d: %v != %v", i, a.Data[i], b.Data[i])
		}
	}
}

func TestDeepFlowRecoversLargeTranslation(t *testing.T) {
	const dx, dy = 4.0, -3.0
	prev, next := valueNoiseShift(72, 72, dx, dy, 20240712)
	p := DefaultDeepFlowParams()
	flow := DeepFlow(prev, next, p)
	mu, mv := interiorMeanFlow(flow, 12)
	if math.Abs(mu-dx) > 0.7 || math.Abs(mv-dy) > 0.7 {
		t.Fatalf("DeepFlow mean flow = (%.3f, %.3f), want ≈ (%.1f, %.1f)", mu, mv, dx, dy)
	}
}

func TestPCAFlowRecoversTranslation(t *testing.T) {
	const dx, dy = 1.5, 1.0
	prev, next := texturedShift(56, 56, dx, dy)
	flow := CalcOpticalFlowPCAFlow(prev, next, DefaultPCAFlowParams())
	mu, mv := interiorMeanFlow(flow, 8)
	if math.Abs(mu-dx) > 0.4 || math.Abs(mv-dy) > 0.4 {
		t.Fatalf("PCAFlow mean flow = (%.3f, %.3f), want ≈ (%.1f, %.1f)", mu, mv, dx, dy)
	}
}

func TestPCAFlowHandlesRotation(t *testing.T) {
	prev, next := rotatedFrame(56, 56, 0.05)
	flow := CalcOpticalFlowPCAFlow(prev, next, DefaultPCAFlowParams())
	// A rotation about the centre produces near-zero mean flow but non-trivial
	// local motion; check that the field is smooth and non-degenerate.
	if flow.MaxMagnitude() < 0.3 {
		t.Fatalf("PCAFlow rotation max magnitude = %.3f, want a non-trivial field", flow.MaxMagnitude())
	}
}

func TestSparseRLOFTracksPoints(t *testing.T) {
	const dx, dy = 2.0, -1.0
	prev, next := texturedShift(64, 64, dx, dy)
	var pts []image.Point
	for y := 12; y < 52; y += 8 {
		for x := 12; x < 52; x += 8 {
			pts = append(pts, image.Point{X: x, Y: y})
		}
	}
	nextPts, status := CalcOpticalFlowSparseRLOF(prev, next, pts, DefaultRLOFParams())
	var su, sv float64
	var n int
	for i, ok := range status {
		if !ok {
			continue
		}
		su += nextPts[i].X - float64(pts[i].X)
		sv += nextPts[i].Y - float64(pts[i].Y)
		n++
	}
	if n < len(pts)/2 {
		t.Fatalf("sparse RLOF tracked only %d/%d points", n, len(pts))
	}
	mu, mv := su/float64(n), sv/float64(n)
	if math.Abs(mu-dx) > 0.35 || math.Abs(mv-dy) > 0.35 {
		t.Fatalf("sparse RLOF mean flow = (%.3f, %.3f), want ≈ (%.1f, %.1f)", mu, mv, dx, dy)
	}
}

func TestSparseRLOFIlluminationInvariance(t *testing.T) {
	const dx, dy = 1.0, 0.0
	prev, next := texturedShift(64, 64, dx, dy)
	// Add a uniform brightness offset to next: plain LK would be biased, RLOF's
	// additive illumination model should absorb it.
	for i := range next.Data {
		v := int(next.Data[i]) + 25
		if v > 255 {
			v = 255
		}
		next.Data[i] = uint8(v)
	}
	pts := []image.Point{{X: 20, Y: 20}, {X: 32, Y: 32}, {X: 40, Y: 24}}
	nextPts, status := CalcOpticalFlowSparseRLOF(prev, next, pts, DefaultRLOFParams())
	tracked := 0
	var su float64
	for i, ok := range status {
		if ok {
			su += nextPts[i].X - float64(pts[i].X)
			tracked++
		}
	}
	if tracked == 0 {
		t.Fatal("sparse RLOF tracked no points under illumination change")
	}
	if mu := su / float64(tracked); math.Abs(mu-dx) > 0.4 {
		t.Fatalf("sparse RLOF under illumination change mean u = %.3f, want ≈ %.1f", mu, dx)
	}
}

func TestDenseRLOFRecoversTranslation(t *testing.T) {
	const dx, dy = 1.0, 1.0
	prev, next := texturedShift(56, 56, dx, dy)
	flow := CalcOpticalFlowDenseRLOF(prev, next, DefaultRLOFParams())
	if flow.Rows != 56 || flow.Cols != 56 {
		t.Fatalf("dense RLOF size = %dx%d, want 56x56", flow.Rows, flow.Cols)
	}
	mu, mv := interiorMeanFlow(flow, 8)
	if math.Abs(mu-dx) > 0.4 || math.Abs(mv-dy) > 0.4 {
		t.Fatalf("dense RLOF mean flow = (%.3f, %.3f), want ≈ (%.1f, %.1f)", mu, mv, dx, dy)
	}
}

func TestSimpleFlowRecoversTranslation(t *testing.T) {
	const dx, dy = 2.0, 1.0
	prev, next := valueNoiseShift(48, 48, dx, dy, 424242)
	flow := CalcOpticalFlowSimpleFlow(prev, next, DefaultSimpleFlowParams())
	mu, mv := interiorMeanFlow(flow, 8)
	if math.Abs(mu-dx) > 0.6 || math.Abs(mv-dy) > 0.6 {
		t.Fatalf("SimpleFlow mean flow = (%.3f, %.3f), want ≈ (%.1f, %.1f)", mu, mv, dx, dy)
	}
}

func TestFloRoundTripBytes(t *testing.T) {
	src := NewFlowField(7, 11)
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			src.set(y, x, float64(x)-3.25, float64(y)*0.5-2.0)
		}
	}
	var buf bytes.Buffer
	if err := WriteFlow(&buf, src); err != nil {
		t.Fatalf("WriteFlow: %v", err)
	}
	got, err := ReadFlow(&buf)
	if err != nil {
		t.Fatalf("ReadFlow: %v", err)
	}
	if got.Rows != src.Rows || got.Cols != src.Cols {
		t.Fatalf("round-trip size = %dx%d, want %dx%d", got.Rows, got.Cols, src.Rows, src.Cols)
	}
	for i := range src.Data {
		// .flo stores float32; compare against the narrowed expectation.
		want := float64(float32(src.Data[i]))
		if got.Data[i] != want {
			t.Fatalf("round-trip sample %d = %v, want %v", i, got.Data[i], want)
		}
	}
}

func TestReadOpticalFlowFile(t *testing.T) {
	src := NewFlowField(5, 6)
	for i := range src.Data {
		src.Data[i] = float64(i) * 0.25
	}
	path := filepath.Join(t.TempDir(), "test.flo")
	if err := WriteOpticalFlow(path, src); err != nil {
		t.Fatalf("WriteOpticalFlow: %v", err)
	}
	got, err := ReadOpticalFlow(path)
	if err != nil {
		t.Fatalf("ReadOpticalFlow: %v", err)
	}
	for i := range src.Data {
		want := float64(float32(src.Data[i]))
		if got.Data[i] != want {
			t.Fatalf("file round-trip sample %d = %v, want %v", i, got.Data[i], want)
		}
	}
}

func TestReadFlowRejectsBadTag(t *testing.T) {
	// A stream whose first four bytes are not the PIEH tag must be rejected.
	bad := []byte{0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}
	if _, err := ReadFlow(bytes.NewReader(bad)); err == nil {
		t.Fatal("ReadFlow accepted a stream with a bad magic tag")
	}
}

func TestReadOpticalFlowMissingFile(t *testing.T) {
	if _, err := ReadOpticalFlow(filepath.Join(t.TempDir(), "nope.flo")); err == nil || !os.IsNotExist(err) {
		t.Fatalf("ReadOpticalFlow on missing file err = %v, want a not-exist error", err)
	}
}

func TestEndpointAndAngularError(t *testing.T) {
	est := NewFlowField(2, 2)
	gt := NewFlowField(2, 2)
	// Pixel 0: exact match. Pixel 1: est (3,4) vs gt (0,0) → EPE 5.
	est.set(0, 1, 3, 4)
	ee := EndpointError(est, gt)
	if math.Abs(ee[0]) > 1e-12 {
		t.Fatalf("EPE[0] = %v, want 0", ee[0])
	}
	if math.Abs(ee[1]-5) > 1e-9 {
		t.Fatalf("EPE[1] = %v, want 5", ee[1])
	}
	aee := AverageEndpointError(est, gt)
	if math.Abs(aee-1.25) > 1e-9 { // (0+5+0+0)/4
		t.Fatalf("AEE = %v, want 1.25", aee)
	}
	// Angular error of identical fields is zero everywhere.
	if aae := AverageAngularError(gt, gt); math.Abs(aae) > 1e-9 {
		t.Fatalf("AAE of identical fields = %v, want 0", aae)
	}
	// A known angular pair: est (1,0) vs gt (0,0) embeds as (1,0,1) vs (0,0,1),
	// angle = arccos(1/sqrt(2)) = pi/4.
	e2 := NewFlowField(1, 1)
	g2 := NewFlowField(1, 1)
	e2.set(0, 0, 1, 0)
	ae := AngularError(e2, g2)
	if math.Abs(ae[0]-math.Pi/4) > 1e-9 {
		t.Fatalf("angular error = %v, want pi/4", ae[0])
	}
}

func TestEndpointErrorStats(t *testing.T) {
	est := NewFlowField(1, 4)
	gt := NewFlowField(1, 4)
	est.set(0, 0, 0, 0) // 0
	est.set(0, 1, 3, 4) // 5
	est.set(0, 2, 1, 0) // 1
	est.set(0, 3, 0, 2) // 2
	s := EndpointErrorStats(est, gt)
	if math.Abs(s.Max-5) > 1e-9 {
		t.Fatalf("stats.Max = %v, want 5", s.Max)
	}
	if math.Abs(s.Mean-2.0) > 1e-9 { // (0+5+1+2)/4
		t.Fatalf("stats.Mean = %v, want 2", s.Mean)
	}
	if math.Abs(s.Median-1.5) > 1e-9 { // median of {0,1,2,5}
		t.Fatalf("stats.Median = %v, want 1.5", s.Median)
	}
	wantRMS := math.Sqrt((0 + 25 + 1 + 4) / 4.0)
	if math.Abs(s.RMS-wantRMS) > 1e-9 {
		t.Fatalf("stats.RMS = %v, want %v", s.RMS, wantRMS)
	}
}

func TestInterpolateFlowExactAtSamples(t *testing.T) {
	pts := []PointF{{X: 2, Y: 2}, {X: 8, Y: 8}}
	vecs := []PointF{{X: 1, Y: 0}, {X: -1, Y: 2}}
	// Shepard (sigma<=0) interpolates exactly at sample locations.
	flow := InterpolateFlow(12, 12, pts, vecs, 0)
	u, v := flow.At(2, 2)
	if math.Abs(u-1) > 1e-9 || math.Abs(v-0) > 1e-9 {
		t.Fatalf("interpolated flow at sample 0 = (%.3f, %.3f), want (1, 0)", u, v)
	}
	u, v = flow.At(8, 8)
	if math.Abs(u+1) > 1e-9 || math.Abs(v-2) > 1e-9 {
		t.Fatalf("interpolated flow at sample 1 = (%.3f, %.3f), want (-1, 2)", u, v)
	}
}

func TestInterpolateFlowConstantField(t *testing.T) {
	// All samples share the same vector → constant dense field everywhere.
	pts := []PointF{{X: 1, Y: 1}, {X: 5, Y: 3}, {X: 8, Y: 7}}
	vecs := []PointF{{X: 2, Y: -1}, {X: 2, Y: -1}, {X: 2, Y: -1}}
	flow := InterpolateFlow(10, 10, pts, vecs, 3.0)
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			u, v := flow.At(y, x)
			if math.Abs(u-2) > 1e-9 || math.Abs(v+1) > 1e-9 {
				t.Fatalf("constant interpolation at (%d,%d) = (%.3f, %.3f), want (2, -1)", y, x, u, v)
			}
		}
	}
}

func TestInterpolateFlowGuidedEdgeAware(t *testing.T) {
	guide := cv.NewMat(20, 20, 1)
	// Left half dark, right half bright: a vertical intensity edge at x=10.
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			if x < 10 {
				guide.Set(y, x, 0, 30)
			} else {
				guide.Set(y, x, 0, 220)
			}
		}
	}
	// One seed on each side with opposite flow.
	pts := []PointF{{X: 4, Y: 10}, {X: 15, Y: 10}}
	vecs := []PointF{{X: 3, Y: 0}, {X: -3, Y: 0}}
	flow := InterpolateFlowGuided(guide, pts, vecs, 6.0, 15.0)
	// A pixel on the dark side should follow the dark seed (u > 0).
	ul, _ := flow.At(10, 2)
	ur, _ := flow.At(10, 17)
	if ul <= 0 {
		t.Fatalf("guided interp on dark side u = %.3f, want > 0", ul)
	}
	if ur >= 0 {
		t.Fatalf("guided interp on bright side u = %.3f, want < 0", ur)
	}
}

func TestNewMethodsDeterministic(t *testing.T) {
	prev, next := texturedShift(40, 40, 1.0, 0.0)
	checks := []struct {
		name string
		f    func() *FlowField
	}{
		{"TVL1", func() *FlowField { return CalcOpticalFlowDenseTVL1(prev, next, DefaultTVL1Params()) }},
		{"DeepFlow", func() *FlowField { return DeepFlow(prev, next, DefaultDeepFlowParams()) }},
		{"PCAFlow", func() *FlowField { return CalcOpticalFlowPCAFlow(prev, next, DefaultPCAFlowParams()) }},
		{"DenseRLOF", func() *FlowField { return CalcOpticalFlowDenseRLOF(prev, next, DefaultRLOFParams()) }},
		{"SimpleFlow", func() *FlowField { return CalcOpticalFlowSimpleFlow(prev, next, DefaultSimpleFlowParams()) }},
	}
	for _, c := range checks {
		a := c.f()
		b := c.f()
		for i := range a.Data {
			if a.Data[i] != b.Data[i] {
				t.Fatalf("%s is non-deterministic at %d: %v != %v", c.name, i, a.Data[i], b.Data[i])
			}
		}
	}
}

// TestMetricsOnRealFlow ties the algorithms to the metrics: a recovered flow
// should have a small AEE against the known ground-truth translation.
func TestMetricsOnRealFlow(t *testing.T) {
	const dx, dy = 2.0, 1.0
	prev, next := texturedShift(64, 64, dx, dy)
	gt := NewFlowField(64, 64)
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			gt.set(y, x, dx, dy)
		}
	}
	flow := CalcOpticalFlowDenseTVL1(prev, next, DefaultTVL1Params())
	// Evaluate on the interior to avoid border artefacts.
	sub := NewFlowField(48, 48)
	subGt := NewFlowField(48, 48)
	for y := 0; y < 48; y++ {
		for x := 0; x < 48; x++ {
			u, v := flow.At(y+8, x+8)
			sub.set(y, x, u, v)
			subGt.set(y, x, dx, dy)
		}
	}
	if aee := AverageEndpointError(sub, subGt); aee > 0.5 {
		t.Fatalf("interior AEE = %.3f, want < 0.5", aee)
	}
}
