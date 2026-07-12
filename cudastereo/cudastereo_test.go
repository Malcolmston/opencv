package cudastereo

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// buildPair makes a tiny rectified pair whose right half is shifted right by a
// known disparity, so the matchers should recover that disparity there. It
// mirrors the fixture used by the sibling stereo package's tests.
func buildPair(w, h, disp int) (left, right *cv.Mat) {
	tex := func(x, y int) uint8 { return uint8((x*167 + y*83 + (x*x)%91) % 256) }
	right = cv.NewMat(h, w, 1)
	left = cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			right.Data[y*w+x] = tex(x, y)
			sx := x
			if x >= w/2 {
				sx = x - disp
			}
			if sx < 0 {
				sx = 0
			}
			left.Data[y*w+x] = tex(sx, y)
		}
	}
	return left, right
}

// buildColorPair wraps buildPair as three-channel RGB images by replicating the
// gray value into each channel, exercising the RGB->gray conversion path.
func buildColorPair(w, h, disp int) (left, right *cv.Mat) {
	gl, gr := buildPair(w, h, disp)
	toRGB := func(g *cv.Mat) *cv.Mat {
		out := cv.NewMat(g.Rows, g.Cols, 3)
		for i := 0; i < g.Total(); i++ {
			out.Data[i*3+0] = g.Data[i]
			out.Data[i*3+1] = g.Data[i]
			out.Data[i*3+2] = g.Data[i]
		}
		return out
	}
	return toRGB(gl), toRGB(gr)
}

func mustPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s: expected panic, got none", name)
		}
	}()
	fn()
}

// near reports whether recovered is within tol of want.
func near(recovered, want, tol int) bool {
	d := recovered - want
	if d < 0 {
		d = -d
	}
	return d <= tol
}

func TestGpuMatLifecycle(t *testing.T) {
	m := cv.NewMat(4, 6, 1)
	m.SetTo(42)

	g := NewGpuMatFromMat(m)
	if g.Empty() {
		t.Fatal("GpuMat should not be empty after upload")
	}
	if r, c := g.Size(); r != 4 || c != 6 {
		t.Fatalf("Size = %d,%d want 4,6", r, c)
	}
	if g.Rows() != 4 || g.Cols() != 6 || g.Channels() != 1 {
		t.Fatalf("Rows/Cols/Channels = %d,%d,%d", g.Rows(), g.Cols(), g.Channels())
	}

	// Upload is a deep copy: mutating the source must not affect the GpuMat.
	m.SetTo(7)
	if g.Download().Data[0] != 42 {
		t.Fatal("Upload was not a deep copy")
	}

	// Mat aliases; Clone is independent.
	clone := g.Clone()
	g.Mat().Data[0] = 9
	if clone.Download().Data[0] != 42 {
		t.Fatal("Clone should be independent")
	}

	g.Release()
	if !g.Empty() {
		t.Fatal("GpuMat should be empty after Release")
	}
	if g.Rows() != 0 || g.Cols() != 0 || g.Channels() != 0 {
		t.Fatal("empty GpuMat dims should be zero")
	}
	if g.Clone().Empty() != true {
		t.Fatal("clone of empty should be empty")
	}
}

func TestGpuMatPanics(t *testing.T) {
	mustPanic(t, "NewGpuMat zero dim", func() { NewGpuMat(0, 4, 1) })
	mustPanic(t, "Upload nil", func() { (&GpuMat{}).Upload(nil) })
	mustPanic(t, "Download empty", func() { (&GpuMat{}).Download() })
	mustPanic(t, "matOf empty", func() { matOf(&GpuMat{}, "x") })
	mustPanic(t, "grayGrid unsupported channels", func() { grayGrid(cv.NewMat(2, 2, 2)) })
}

func TestStream(t *testing.T) {
	s := NewStream()
	s.WaitForCompletion()
	var nilStream *Stream
	nilStream.WaitForCompletion() // must be safe on nil
}

func TestStereoBM(t *testing.T) {
	left, right := buildPair(64, 24, 8)
	bm := CreateStereoBM(16, 7)
	d := bm.Compute(NewGpuMatFromMat(left), NewGpuMatFromMat(right), nil).Download()
	if got := int(d.Data[12*64+50]); !near(got, 8, 1) {
		t.Fatalf("StereoBM recovered %d, want ~8", got)
	}
}

func TestStereoBMDefaults(t *testing.T) {
	bm := CreateStereoBM(0, 0)
	if bm.NumDisparities != 64 || bm.BlockSize != 19 {
		t.Fatalf("defaults = %d,%d want 64,19", bm.NumDisparities, bm.BlockSize)
	}
}

func TestStereoSGM(t *testing.T) {
	left, right := buildPair(64, 24, 8)
	for _, mode := range []SGMMode{ModeHH4, ModeHH} {
		sgm := CreateStereoSGM(0, 16, 0, 0, 0, mode)
		sgm.BlockSize = 5
		d := sgm.Compute(NewGpuMatFromMat(left), NewGpuMatFromMat(right), NewStream()).Download()
		if got := int(d.Data[12*64+50]); !near(got, 8, 1) {
			t.Fatalf("StereoSGM (mode %d) recovered %d, want ~8", mode, got)
		}
	}
}

func TestStereoSGMDefaults(t *testing.T) {
	sgm := CreateStereoSGM(0, 0, 0, 0, 0, ModeHH4)
	if sgm.NumDisparities != 128 || sgm.UniquenessRatio != 5 {
		t.Fatalf("defaults = %d,%d", sgm.NumDisparities, sgm.UniquenessRatio)
	}
}

func TestStereoBeliefPropagation(t *testing.T) {
	left, right := buildPair(64, 24, 8)
	bp := CreateStereoBeliefPropagation(16, 5, 3)
	d := bp.Compute(NewGpuMatFromMat(left), NewGpuMatFromMat(right), nil).Download()
	// Sample several interior pixels of the shifted (disparity 8) region.
	for _, x := range []int{40, 48, 56} {
		if got := int(d.Data[12*64+x]); !near(got, 8, 1) {
			t.Fatalf("BP recovered %d at x=%d, want ~8", got, x)
		}
	}
}

func TestStereoBeliefPropagationColorInput(t *testing.T) {
	left, right := buildColorPair(64, 24, 8)
	bp := CreateStereoBeliefPropagation(16, 5, 1) // single-scale loopy BP
	d := bp.Compute(NewGpuMatFromMat(left), NewGpuMatFromMat(right), nil).Download()
	if got := int(d.Data[12*64+50]); !near(got, 8, 1) {
		t.Fatalf("BP (color) recovered %d, want ~8", got)
	}
}

func TestStereoBeliefPropagationDefaultsAndEstimate(t *testing.T) {
	bp := CreateStereoBeliefPropagation(0, 0, 0)
	if bp.NumDisparities != 64 || bp.NumIters != 5 || bp.NumLevels != 5 {
		t.Fatalf("defaults = %d,%d,%d", bp.NumDisparities, bp.NumIters, bp.NumLevels)
	}
	ndisp, iters, levels := bp.EstimateRecommendedParams(640, 480)
	if ndisp != 160 || iters != 8 || levels < 1 {
		t.Fatalf("estimate = %d,%d,%d", ndisp, iters, levels)
	}
	if bp.NumDisparities != ndisp || bp.NumIters != iters || bp.NumLevels != levels {
		t.Fatal("EstimateRecommendedParams should update the receiver")
	}
}

func TestStereoBeliefPropagationSizeMismatch(t *testing.T) {
	l := NewGpuMat(10, 10, 1)
	r := NewGpuMat(10, 12, 1)
	bp := CreateStereoBeliefPropagation(8, 2, 1)
	mustPanic(t, "BP size mismatch", func() { bp.Compute(l, r, nil) })
}

func TestStereoConstantSpaceBP(t *testing.T) {
	left, right := buildPair(64, 24, 8)
	csbp := CreateStereoConstantSpaceBP(16, 8, 4, 4)
	d := csbp.Compute(NewGpuMatFromMat(left), NewGpuMatFromMat(right), NewStream()).Download()
	for _, x := range []int{40, 48, 56} {
		if got := int(d.Data[12*64+x]); !near(got, 8, 1) {
			t.Fatalf("CSBP recovered %d at x=%d, want ~8", got, x)
		}
	}
}

func TestStereoConstantSpaceBPDefaultsAndEstimate(t *testing.T) {
	csbp := CreateStereoConstantSpaceBP(0, 0, 0, 0)
	if csbp.NumDisparities != 128 || csbp.NrPlane != 4 {
		t.Fatalf("defaults = %d,%d", csbp.NumDisparities, csbp.NrPlane)
	}
	ndisp, iters, levels, nrPlane := csbp.EstimateRecommendedParams(640, 480)
	if ndisp < 1 || iters < 1 || levels < 1 || nrPlane < 1 {
		t.Fatalf("estimate = %d,%d,%d,%d", ndisp, iters, levels, nrPlane)
	}
	if csbp.NrPlane != nrPlane {
		t.Fatal("EstimateRecommendedParams should update the receiver")
	}
}

func TestStereoConstantSpaceBPSizeMismatch(t *testing.T) {
	l := NewGpuMat(10, 10, 1)
	r := NewGpuMat(12, 10, 1)
	csbp := CreateStereoConstantSpaceBP(8, 2, 1, 2)
	mustPanic(t, "CSBP size mismatch", func() { csbp.Compute(l, r, nil) })
}

func TestDisparityBilateralFilter(t *testing.T) {
	left, right := buildPair(64, 24, 8)
	guide := NewGpuMatFromMat(left)
	disp := CreateStereoBM(16, 7).Compute(guide, NewGpuMatFromMat(right), nil)

	f := CreateDisparityBilateralFilter(16, 3, 2)
	out := f.Apply(disp, guide, NewStream()).Download()
	if out.Rows != 24 || out.Cols != 64 || out.Channels != 1 {
		t.Fatalf("filtered shape = %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
	// A pixel that matched at disparity ~8 should remain near 8 after filtering.
	if got := int(out.Data[12*64+50]); !near(got, 8, 2) {
		t.Fatalf("filtered disparity = %d, want ~8", got)
	}
}

func TestDisparityBilateralFilterPreservesHoles(t *testing.T) {
	disp := NewGpuMat(8, 8, 1)
	disp.Mat().SetTo(10)
	disp.Mat().Data[3*8+3] = 0 // a hole
	guide := NewGpuMat(8, 8, 1)
	guide.Mat().SetTo(100)

	f := CreateDisparityBilateralFilter(16, 2, 1)
	out := f.Apply(disp, guide, nil).Download()
	if out.Data[3*8+3] != 0 {
		t.Fatalf("hole should be preserved, got %d", out.Data[3*8+3])
	}
	if !near(int(out.Data[0]), 10, 0) {
		t.Fatalf("flat region should stay 10, got %d", out.Data[0])
	}
}

func TestDisparityBilateralFilterPanics(t *testing.T) {
	f := CreateDisparityBilateralFilter(16, 2, 1)
	badDisp := NewGpuMat(4, 4, 3) // multi-channel disparity
	mustPanic(t, "multi-channel disparity", func() { f.Apply(badDisp, NewGpuMat(4, 4, 1), nil) })
	mustPanic(t, "size mismatch", func() { f.Apply(NewGpuMat(4, 4, 1), NewGpuMat(4, 6, 1), nil) })
}

func TestReprojectImageTo3D(t *testing.T) {
	d := NewGpuMat(2, 3, 1)
	d.Mat().SetTo(8)
	Q := [4][4]float64{
		{1, 0, 0, -1},
		{0, 1, 0, -1},
		{0, 0, 0, 500},
		{0, 0, 10, 0},
	}
	pts := ReprojectImageTo3D(d, Q, NewStream())
	if len(pts) != 6 {
		t.Fatalf("expected 6 points, got %d", len(pts))
	}
	for _, p := range pts {
		if p[2] < 6.24 || p[2] > 6.26 {
			t.Fatalf("constant disparity should give constant Z~6.25, got %v", p[2])
		}
	}
}

func TestDrawColorDisp(t *testing.T) {
	disp := NewGpuMat(4, 4, 1)
	disp.Mat().SetTo(8)
	disp.Mat().Data[0] = 0 // invalid -> black

	out := DrawColorDisp(disp, 16, nil).Download()
	if out.Channels != 3 {
		t.Fatalf("expected 3-channel output, got %d", out.Channels)
	}
	if out.Data[0] != 0 || out.Data[1] != 0 || out.Data[2] != 0 {
		t.Fatalf("invalid pixel should be black, got %d,%d,%d", out.Data[0], out.Data[1], out.Data[2])
	}
	// A valid pixel should be coloured (non-zero somewhere).
	i := (1) * 3
	if out.Data[i] == 0 && out.Data[i+1] == 0 && out.Data[i+2] == 0 {
		t.Fatal("valid pixel should not be black")
	}

	// ndisp inference path (ndisp <= 0).
	inferred := DrawColorDisp(disp, 0, nil).Download()
	if inferred.Channels != 3 {
		t.Fatal("inferred ndisp output should be 3-channel")
	}
}

func TestDrawColorDispPanic(t *testing.T) {
	mustPanic(t, "multi-channel disparity", func() { DrawColorDisp(NewGpuMat(4, 4, 3), 16, nil) })
}
