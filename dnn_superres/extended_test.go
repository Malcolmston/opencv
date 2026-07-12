package dnn_superres

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// extendedMethods lists the new arbitrary-scale upsamplers with a conservative
// reconstruction-PSNR floor each must clear on the blurred synthetic pattern.
var extendedMethods = []struct {
	name    string
	fn      UpsampleFunc
	minPSNR float64
}{
	{"gaussian", UpsampleGaussian, 13},
	{"bspline", UpsampleBSpline, 13},
	{"hermite", UpsampleHermite, 13},
	{"mitchell", UpsampleMitchell, 13},
	{"lanczos3", UpsampleLanczos3, 13},
	{"scale-bicubic", UpsampleScale, 13},
	{"espcn", UpsampleESPCN, 13},
	{"lapsrn", UpsampleLapSRN, 13},
	{"nedi", UpsampleNEDI, 13},
	{"dcci", UpsampleDCCI, 13},
	{"gradient-profile", UpsampleGradientProfile, 12},
	{"ibp", UpsampleIBP, 12},
}

func TestExtendedSizesAndChannels(t *testing.T) {
	for _, ch := range []int{1, 3} {
		src := synthPattern(10, 12, ch)
		for _, m := range extendedMethods {
			for _, scale := range []int{2, 3, 5} {
				out, err := m.fn(src, scale)
				if err != nil {
					t.Fatalf("%s x%d ch%d: %v", m.name, scale, ch, err)
				}
				wantH, wantW := src.Rows*scale, src.Cols*scale
				if out.Rows != wantH || out.Cols != wantW {
					t.Errorf("%s x%d ch%d: size = %dx%d, want %dx%d",
						m.name, scale, ch, out.Rows, out.Cols, wantH, wantW)
				}
				if out.Channels != ch {
					t.Errorf("%s x%d: channels = %d, want %d", m.name, scale, out.Channels, ch)
				}
			}
		}
	}
}

func TestExtendedReconstructionPSNR(t *testing.T) {
	for _, scale := range []int{2, 3} {
		hi := cv.GaussianBlur(synthPattern(24, 24, 3), 3, 0.8)
		lo := boxDownsample(hi, scale)
		for _, m := range extendedMethods {
			out, err := m.fn(lo, scale)
			if err != nil {
				t.Fatalf("%s x%d: %v", m.name, scale, err)
			}
			ref := hi.Region(0, 0, out.Rows, out.Cols)
			p, err := PSNR(out, ref)
			if err != nil {
				t.Fatal(err)
			}
			if p < m.minPSNR {
				t.Errorf("%s x%d: PSNR %.2f dB below threshold %.2f", m.name, scale, p, m.minPSNR)
			}
		}
	}
}

func TestExtendedConstantPreserved(t *testing.T) {
	for _, m := range extendedMethods {
		src := cv.NewMat(6, 8, 3)
		src.SetTo(140)
		out, err := m.fn(src, 4)
		if err != nil {
			t.Fatal(err)
		}
		for i := range out.Data {
			if out.Data[i] != 140 {
				t.Fatalf("%s: constant not preserved, got %d at %d", m.name, out.Data[i], i)
				break
			}
		}
	}
}

func TestExtendedDeterminism(t *testing.T) {
	src := synthPattern(9, 11, 3)
	for _, m := range extendedMethods {
		a, err := m.fn(src, 3)
		if err != nil {
			t.Fatal(err)
		}
		b, err := m.fn(src, 3)
		if err != nil {
			t.Fatal(err)
		}
		for i := range a.Data {
			if a.Data[i] != b.Data[i] {
				t.Fatalf("%s: nondeterministic at %d", m.name, i)
			}
		}
	}
}

func TestIterativeBackProjectionConverges(t *testing.T) {
	// More iterations must not worsen the reconstruction residual (the LR
	// simulated from HR should match the true LR increasingly well).
	src := cv.GaussianBlur(synthPattern(20, 20, 1), 3, 0.8)
	lo := boxDownsample(src, 2)
	prev := math.Inf(1)
	for _, iters := range []int{1, 4, 12} {
		hr, err := IterativeBackProjection(lo, 2, iters)
		if err != nil {
			t.Fatal(err)
		}
		simLo := boxDownsample(hr, 2)
		mse, err := MSE(simLo, lo)
		if err != nil {
			t.Fatal(err)
		}
		if mse > prev+1e-6 {
			t.Errorf("IBP residual grew: iters=%d MSE=%.4f prev=%.4f", iters, mse, prev)
		}
		prev = mse
	}
}

func TestNEDIandDCCIReconstruct(t *testing.T) {
	// The edge-directed methods must produce genuine reconstructions (well above
	// the ~8 dB a random image scores) on a pattern with strong diagonal edges.
	hi := cv.GaussianBlur(synthPattern(24, 24, 1), 3, 0.6)
	lo := boxDownsample(hi, 2)
	ref := hi.Region(0, 0, lo.Rows*2, lo.Cols*2)
	for _, m := range []struct {
		name string
		fn   UpsampleFunc
	}{{"nedi", UpsampleNEDI}, {"dcci", UpsampleDCCI}} {
		out, _ := m.fn(lo, 2)
		p, _ := PSNR(out, ref)
		if p < 15 {
			t.Errorf("%s PSNR %.2f below reconstruction floor 15", m.name, p)
		}
	}
}

func TestSSIMIdenticalAndDegraded(t *testing.T) {
	src := synthPattern(20, 20, 3)
	s, err := SSIM(src, src.Clone())
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(s-1.0) > 1e-9 {
		t.Errorf("SSIM of identical images = %v, want 1.0", s)
	}
	// A blurred copy must score below 1 but well above 0.
	blur := cv.GaussianBlur(src, 5, 2.0)
	s2, err := SSIM(src, blur)
	if err != nil {
		t.Fatal(err)
	}
	if s2 >= 1.0 || s2 <= 0.0 {
		t.Errorf("SSIM of blurred copy = %v, want in (0,1)", s2)
	}
}

func TestBenchmarkOrderingAndDefaults(t *testing.T) {
	hi := cv.GaussianBlur(synthPattern(24, 24, 3), 3, 0.8)
	res, err := Benchmark(hi, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != len(DefaultUpsamplers()) {
		t.Fatalf("results = %d, want %d", len(res), len(DefaultUpsamplers()))
	}
	// Sorted best-first by PSNR.
	for i := 1; i < len(res); i++ {
		if res[i-1].PSNR < res[i].PSNR {
			t.Errorf("results not sorted: %v before %v", res[i-1], res[i])
		}
	}
	// Best reconstruction must be a genuine one, not noise-level.
	if res[0].PSNR < 15 {
		t.Errorf("best PSNR %.2f too low to be a real reconstruction", res[0].PSNR)
	}
	// Every method must report a sane SSIM.
	for _, r := range res {
		if r.SSIM <= 0 || r.SSIM > 1.0001 {
			t.Errorf("%s: SSIM %.4f out of range", r.Name, r.SSIM)
		}
	}
}

func TestBenchmarkOrderingSaneVsNearest(t *testing.T) {
	// A proper interpolator must beat nearest-neighbour on a smooth reference.
	hi := cv.GaussianBlur(synthPattern(24, 24, 3), 3, 1.0)
	methods := []NamedUpsampler{
		{"nearest", UpsampleNearest},
		{"bicubic", UpsampleScale},
	}
	res, err := Benchmark(hi, 2, methods)
	if err != nil {
		t.Fatal(err)
	}
	if res[0].Name != "bicubic" {
		t.Errorf("expected bicubic to rank above nearest, got order %s, %s", res[0].Name, res[1].Name)
	}
}

func TestLumaOnlyAndPerChannel(t *testing.T) {
	rgb := synthPattern(10, 10, 3)
	out, err := UpsampleLumaOnly(rgb, 3, UpsampleScale)
	if err != nil {
		t.Fatal(err)
	}
	if out.Rows != 30 || out.Cols != 30 || out.Channels != 3 {
		t.Errorf("luma-only size = %dx%dx%d, want 30x30x3", out.Rows, out.Cols, out.Channels)
	}
	// Grayscale luma-only must equal the direct upscale.
	gray := synthPattern(8, 8, 1)
	lo, _ := UpsampleLumaOnly(gray, 2, UpsampleScale)
	direct, _ := UpsampleScale(gray, 2)
	for i := range lo.Data {
		if lo.Data[i] != direct.Data[i] {
			t.Fatalf("gray luma-only differs from direct at %d", i)
		}
	}
	// Per-channel on RGB must match the method's own multi-channel output.
	pc, err := UpsamplePerChannel(rgb, 2, UpsampleScale)
	if err != nil {
		t.Fatal(err)
	}
	whole, _ := UpsampleScale(rgb, 2)
	for i := range pc.Data {
		if pc.Data[i] != whole.Data[i] {
			t.Fatalf("per-channel differs from whole-image at %d", i)
		}
	}
}

func TestExtendedErrorPaths(t *testing.T) {
	good := synthPattern(6, 6, 3)

	if _, err := UpsampleMitchell(good, 1); err == nil {
		t.Error("expected error for scale 1")
	}
	if _, err := UpsampleNEDI(nil, 2); err == nil {
		t.Error("expected error for nil image")
	}
	if _, err := IterativeBackProjection(good, 2, 0); err == nil {
		t.Error("expected error for zero iterations")
	}
	if _, err := EdgeGuidedUpscale(good, 1, 1.0); err == nil {
		t.Error("expected error for bad scale")
	}
	if _, err := UpsampleLumaOnly(good, 2, nil); err == nil {
		t.Error("expected error for nil UpsampleFunc")
	}
	if _, err := UpsamplePerChannel(good, 2, nil); err == nil {
		t.Error("expected error for nil UpsampleFunc")
	}
	if _, err := UpsampleLumaOnly(good, 1, UpsampleScale); err == nil {
		t.Error("expected error for bad luma scale")
	}
	// SSIM mismatch and empty.
	if _, err := SSIM(good, synthPattern(8, 8, 3)); err == nil {
		t.Error("expected SSIM shape mismatch error")
	}
	if _, err := SSIM(nil, good); err == nil {
		t.Error("expected SSIM nil error")
	}
	// Benchmark errors.
	if _, err := Benchmark(nil, 2, nil); err == nil {
		t.Error("expected Benchmark empty error")
	}
	if _, err := Benchmark(good, 1, nil); err == nil {
		t.Error("expected Benchmark bad scale error")
	}
	tiny := cv.NewMat(2, 2, 1)
	if _, err := Benchmark(tiny, 4, nil); err == nil {
		t.Error("expected Benchmark too-small error")
	}
	// Benchmark surfacing a method error.
	bad := []NamedUpsampler{{"broken", func(_ *cv.Mat, _ int) (*cv.Mat, error) {
		return nil, errIterations
	}}}
	if _, err := Benchmark(good, 2, bad); err == nil {
		t.Error("expected Benchmark to surface method error")
	}
}
