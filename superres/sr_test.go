package superres

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// sinusoidMat builds a smooth, well-conditioned single-channel test image.
func sinusoidMat(h, w int) *cv.Mat {
	m := cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := 128 + 60*math.Sin(2*math.Pi*float64(x)/17) + 40*math.Sin(2*math.Pi*float64(y)/23)
			m.Data[y*w+x] = uint8(v)
		}
	}
	return m
}

func TestShiftIdentityAndInteger(t *testing.T) {
	src := sinusoidMat(12, 12)
	// Zero shift is the identity.
	same := Shift(src, 0, 0)
	for i := range src.Data {
		if same.Data[i] != src.Data[i] {
			t.Fatalf("zero shift changed sample %d", i)
		}
	}
	// A whole-pixel sub-pixel shift matches the integer shifter exactly.
	a := Shift(src, 1, 0)
	b := IntegerShift(src, 1, 0)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("Shift vs IntegerShift differ at %d: %d != %d", i, a.Data[i], b.Data[i])
		}
	}
}

func TestIntegerShiftKnownAnswer(t *testing.T) {
	src := gray1(1, 4, []uint8{10, 20, 30, 40})
	got := IntegerShift(src, 1, 0) // content moves right by one, left border replicates
	want := []uint8{10, 10, 20, 30}
	for i, w := range want {
		if got.Data[i] != w {
			t.Errorf("shift[%d] = %d, want %d", i, got.Data[i], w)
		}
	}
}

func TestEstimateShiftSubpixel(t *testing.T) {
	base := sinusoidMat(40, 40)
	cases := []Shift2D{{0.3, -0.2}, {0.7, 0.5}, {-0.4, 0.9}}
	for _, c := range cases {
		moved := Shift(base, c.Dx, c.Dy)
		est := EstimateShiftRefine(base, moved, 12)
		if !approx(est.Dx, c.Dx, 0.05) || !approx(est.Dy, c.Dy, 0.05) {
			t.Errorf("EstimateShiftRefine = (%.3f, %.3f), want (%.3f, %.3f)", est.Dx, est.Dy, c.Dx, c.Dy)
		}
	}
}

func TestPhaseCorrelateInteger(t *testing.T) {
	h, w := 16, 16
	p := cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			p.Data[y*w+x] = uint8((x*7 + y*13) % 200)
		}
	}
	dx, dy := 3, -2
	mv := cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			sy := ((y-dy)%h + h) % h
			sx := ((x-dx)%w + w) % w
			mv.Data[y*w+x] = p.Data[sy*w+sx]
		}
	}
	est := PhaseCorrelateShift(p, mv)
	if !approx(est.Dx, float64(dx), 0.01) || !approx(est.Dy, float64(dy), 0.01) {
		t.Errorf("PhaseCorrelateShift = (%.3f, %.3f), want (%d, %d)", est.Dx, est.Dy, dx, dy)
	}
}

func TestNEDIConstantAndDims(t *testing.T) {
	src := constMat(8, 8, 3, 123)
	out := NEDIDouble(src)
	if out.Rows != 16 || out.Cols != 16 || out.Channels != 3 {
		t.Fatalf("NEDIDouble dims = %dx%dx%d, want 16x16x3", out.Rows, out.Cols, out.Channels)
	}
	for i, v := range out.Data {
		if v != 123 {
			t.Fatalf("NEDIDouble changed constant at %d: %d", i, v)
		}
	}
	four := NEDI(src, 4)
	if four.Rows != 32 || four.Cols != 32 {
		t.Errorf("NEDI x4 dims = %dx%d, want 32x32", four.Rows, four.Cols)
	}
}

func TestNEDIRampMidpoints(t *testing.T) {
	// A horizontal ramp should be interpolated close to the true midpoints.
	ramp := cv.NewMat(8, 8, 1)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			ramp.Data[y*8+x] = uint8(10 * x)
		}
	}
	out := NEDIDouble(ramp)
	// Interior odd column of an interior row: true value ~ midpoint.
	row := 4
	for col := 3; col <= 11; col += 2 {
		got := float64(out.Data[row*16+col])
		want := 5.0 * float64(col) // 10*x with x=col/2 => 5*col
		if !approx(got, want, 3) {
			t.Errorf("NEDI ramp at (%d,%d) = %v, want ~%v", row, col, got, want)
		}
	}
}

func TestEdgeDirectedResizeDims(t *testing.T) {
	src := sinusoidMat(10, 10)
	out := EdgeDirectedResize(src, 25, 25)
	if out.Rows != 25 || out.Cols != 25 {
		t.Errorf("EdgeDirectedResize dims = %dx%d, want 25x25", out.Rows, out.Cols)
	}
}

func TestBackProjectionConstantAndConsistency(t *testing.T) {
	src := constMat(10, 10, 1, 88)
	out := BackProjectionSR(src, 2, 6)
	if out.Rows != 20 || out.Cols != 20 {
		t.Fatalf("dims = %dx%d, want 20x20", out.Rows, out.Cols)
	}
	for i, v := range out.Data {
		if v != 88 {
			t.Fatalf("back-projection changed constant at %d: %d", i, v)
		}
	}

	// On a real image, iterating must not increase the reprojection error and
	// should improve over the plain bicubic initial estimate.
	low := sinusoidMat(16, 16)
	bic := BicubicResize(low, 32, 32)
	sr := BackProjectionSR(low, 2, 10)
	reBic := GaussianDownscale(bic, 2, superresGaussianSigmaFor(2))
	reSR := GaussianDownscale(sr, 2, superresGaussianSigmaFor(2))
	if MSE(reSR, low) > MSE(reBic, low) {
		t.Errorf("back-projection did not reduce reprojection error: SR=%v bicubic=%v",
			MSE(reSR, low), MSE(reBic, low))
	}
}

func TestExampleFreeSR(t *testing.T) {
	src := constMat(8, 8, 1, 100)
	out := ExampleFreeSR(src, 2, 4)
	if out.Rows != 16 || out.Cols != 16 {
		t.Fatalf("dims = %dx%d, want 16x16", out.Rows, out.Cols)
	}
	for i, v := range out.Data {
		if v != 100 {
			t.Fatalf("ExampleFreeSR changed constant at %d: %d", i, v)
		}
	}
	// Non-power-of-two scale falls back to bicubic base; just check dims.
	odd := ExampleFreeSR(sinusoidMat(8, 8), 3, 2)
	if odd.Rows != 24 || odd.Cols != 24 {
		t.Errorf("ExampleFreeSR x3 dims = %dx%d, want 24x24", odd.Rows, odd.Cols)
	}
}

func TestProgressiveUpscale(t *testing.T) {
	src := constMat(10, 10, 3, 70)
	out := ProgressiveUpscale(src, 3, 0.3)
	if out.Rows != 30 || out.Cols != 30 {
		t.Fatalf("dims = %dx%d, want 30x30", out.Rows, out.Cols)
	}
	for _, v := range out.Data {
		if v != 70 {
			t.Fatal("ProgressiveUpscale changed constant image")
		}
	}
}

func TestRegisterFramesRecoversShifts(t *testing.T) {
	base := sinusoidMat(40, 40)
	f1 := Shift(base, 0.5, 0.3)
	f2 := Shift(base, -0.6, 0.4)
	shifts := RegisterFrames([]*cv.Mat{base, f1, f2})
	if len(shifts) != 3 {
		t.Fatalf("got %d shifts, want 3", len(shifts))
	}
	if shifts[0] != (Shift2D{}) {
		t.Errorf("reference shift = %v, want zero", shifts[0])
	}
	if !approx(shifts[1].Dx, 0.5, 0.05) || !approx(shifts[1].Dy, 0.3, 0.05) {
		t.Errorf("shift[1] = %v, want ~(0.5, 0.3)", shifts[1])
	}
	if !approx(shifts[2].Dx, -0.6, 0.05) || !approx(shifts[2].Dy, 0.4, 0.05) {
		t.Errorf("shift[2] = %v, want ~(-0.6, 0.4)", shifts[2])
	}
}

func TestFuseAverageAndMedianIdentical(t *testing.T) {
	img := sinusoidMat(12, 12)
	frames := []*cv.Mat{img, img.Clone(), img.Clone()}
	avg := FuseAverage(frames, nil)
	med := FuseMedian(frames, nil)
	for i := range img.Data {
		if avg.Data[i] != img.Data[i] {
			t.Fatalf("FuseAverage of identical frames differs at %d", i)
		}
		if med.Data[i] != img.Data[i] {
			t.Fatalf("FuseMedian of identical frames differs at %d", i)
		}
	}
}

func TestFuseMedianRejectsOutlier(t *testing.T) {
	base := constMat(4, 4, 1, 100)
	// One frame is corrupted; the median must ignore it while the mean would not.
	bad := constMat(4, 4, 1, 100)
	bad.Data[5] = 250
	frames := []*cv.Mat{base, bad, base.Clone()}
	med := FuseMedian(frames, nil)
	if med.Data[5] != 100 {
		t.Errorf("median failed to reject outlier: %d, want 100", med.Data[5])
	}
}

func TestShiftAndAddConstant(t *testing.T) {
	frames := []*cv.Mat{constMat(8, 8, 1, 111), constMat(8, 8, 1, 111)}
	shifts := []Shift2D{{0, 0}, {0.5, 0.5}}
	out := ShiftAndAddSR(frames, shifts, 2)
	if out.Rows != 16 || out.Cols != 16 {
		t.Fatalf("dims = %dx%d, want 16x16", out.Rows, out.Cols)
	}
	for i, v := range out.Data {
		if v != 111 {
			t.Fatalf("ShiftAndAddSR changed constant at %d: %d", i, v)
		}
	}
}

func TestMultiFrameSRDims(t *testing.T) {
	base := sinusoidMat(16, 16)
	frames := []*cv.Mat{base, Shift(base, 0.5, 0), Shift(base, 0, 0.5), Shift(base, 0.5, 0.5)}
	out := MultiFrameSR(frames, 2)
	if out.Rows != 32 || out.Cols != 32 {
		t.Errorf("MultiFrameSR dims = %dx%d, want 32x32", out.Rows, out.Cols)
	}
}

// BenchmarkIterativeBackProjection exercises the heaviest routine: each
// iteration performs a full-resolution Gaussian blur plus two bicubic resizes
// per channel.
func BenchmarkIterativeBackProjection(b *testing.B) {
	low := sinusoidMat(32, 32)
	params := DefaultBackProjectionParams(2)
	params.Iterations = 8
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = IterativeBackProjection(low, params)
	}
}
