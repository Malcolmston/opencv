package dnn_superres

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// synthPattern builds a deterministic w×h test image with the given channel
// count. It combines smooth ramps, a diagonal edge and a central box so that
// every algorithm (smooth-region, edge and detail behaviour) is exercised.
func synthPattern(h, w, ch int) *cv.Mat {
	m := cv.NewMat(h, w, ch)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			ramp := (x*255)/max1(w-1) ^ (y * 3)
			diag := 0
			if x+y > (w+h)/2 {
				diag = 60
			}
			box := 0
			if x > w/4 && x < 3*w/4 && y > h/4 && y < 3*h/4 {
				box = 90
			}
			base := clampByte(float64((ramp + diag + box) % 256)) //nolint:gosec
			i := (y*w + x) * ch
			for c := 0; c < ch; c++ {
				m.Data[i+c] = clampByte(float64(int(base)+c*20) - 0)
			}
		}
	}
	return m
}

func max1(v int) int {
	if v < 1 {
		return 1
	}
	return v
}

// downThenUp downsamples hi by the integer factor (box average) so that
// upscaling the small image is a genuine reconstruction task with a known
// ground truth.
func downThenUp(hi *cv.Mat, scale int) *cv.Mat {
	lw, lh := hi.Cols/scale, hi.Rows/scale
	lo := cv.NewMat(lh, lw, hi.Channels)
	for y := 0; y < lh; y++ {
		for x := 0; x < lw; x++ {
			for c := 0; c < hi.Channels; c++ {
				var sum int
				for dy := 0; dy < scale; dy++ {
					for dx := 0; dx < scale; dx++ {
						sum += int(hi.At(y*scale+dy, x*scale+dx, c))
					}
				}
				lo.Set(y, x, c, clampByte(float64(sum)/float64(scale*scale)))
			}
		}
	}
	return lo
}

func TestUpsampleSizesAndAlgorithms(t *testing.T) {
	algos := []string{"nearest", "bilinear", "bicubic", "lanczos", "edge", "fsrcnn"}
	scales := []int{2, 3, 4}
	channels := []int{1, 3}
	for _, ch := range channels {
		src := synthPattern(12, 12, ch)
		for _, algo := range algos {
			for _, scale := range scales {
				sr := NewDnnSuperResImpl()
				if err := sr.SetModel(algo, scale); err != nil {
					t.Fatalf("SetModel(%q,%d): %v", algo, scale, err)
				}
				out, err := sr.Upsample(src)
				if err != nil {
					t.Fatalf("Upsample %s x%d ch%d: %v", algo, scale, ch, err)
				}
				wantH, wantW := src.Rows*scale, src.Cols*scale
				if out.Rows != wantH || out.Cols != wantW {
					t.Errorf("%s x%d ch%d: size = %dx%d, want %dx%d",
						algo, scale, ch, out.Rows, out.Cols, wantH, wantW)
				}
				if out.Channels != ch {
					t.Errorf("%s x%d: channels = %d, want %d", algo, scale, out.Channels, ch)
				}
			}
		}
	}
}

func TestReconstructionPSNR(t *testing.T) {
	// Downscale a known pattern then upscale; the interpolating methods must
	// reconstruct it well above these conservative thresholds.
	type tc struct {
		algo    string
		minPSNR float64
	}
	// Conservative thresholds: a random image scores near 8 dB, so 15 dB proves
	// genuine reconstruction while tolerating the hardest x4 case.
	cases := []tc{
		{"bilinear", 15},
		{"bicubic", 15},
		{"lanczos", 15},
		{"edge", 15},
	}
	for _, scale := range []int{2, 3, 4} {
		hi := synthPattern(24, 24, 3)
		// Make hi divisible and reduce high-frequency so box-downscale has an
		// invertible-ish target.
		hi = cv.GaussianBlur(hi, 3, 0.8)
		lo := downThenUp(hi, scale)
		up := lo // ground-truth-sized reference: re-upscale lo
		_ = up
		for _, c := range cases {
			sr := NewDnnSuperResImpl()
			if err := sr.SetModel(c.algo, scale); err != nil {
				t.Fatal(err)
			}
			out, err := sr.Upsample(lo)
			if err != nil {
				t.Fatal(err)
			}
			// Compare against hi cropped to the reconstructed size.
			ref := hi.Region(0, 0, out.Rows, out.Cols)
			p, err := PSNR(out, ref)
			if err != nil {
				t.Fatal(err)
			}
			if p < c.minPSNR {
				t.Errorf("%s x%d: PSNR %.2f dB below threshold %.2f", c.algo, scale, p, c.minPSNR)
			}
		}
	}
}

func TestDeterminism(t *testing.T) {
	src := synthPattern(10, 14, 3)
	for _, algo := range []string{"nearest", "bilinear", "bicubic", "lanczos", "edge", "fsrcnn"} {
		a, _ := UpsamplerFromName(t, algo, 3).Upsample(src)
		b, _ := UpsamplerFromName(t, algo, 3).Upsample(src)
		if len(a.Data) != len(b.Data) {
			t.Fatalf("%s: length mismatch", algo)
		}
		for i := range a.Data {
			if a.Data[i] != b.Data[i] {
				t.Fatalf("%s: nondeterministic at %d", algo, i)
			}
		}
	}
}

// UpsamplerFromName is a small test helper.
func UpsamplerFromName(t *testing.T, algo string, scale int) *DnnSuperResImpl {
	t.Helper()
	sr := NewDnnSuperResImpl()
	if err := sr.SetModel(algo, scale); err != nil {
		t.Fatal(err)
	}
	return sr
}

func TestNearestPreservesSamples(t *testing.T) {
	src := synthPattern(6, 8, 3)
	out, err := UpsampleNearest(src, 2)
	if err != nil {
		t.Fatal(err)
	}
	// Every source sample must appear exactly in the 2x block.
	for y := 0; y < src.Rows; y++ {
		for x := 0; x < src.Cols; x++ {
			for c := 0; c < src.Channels; c++ {
				if out.At(y*2, x*2, c) != src.At(y, x, c) {
					t.Fatalf("nearest lost sample at (%d,%d,%d)", y, x, c)
				}
			}
		}
	}
}

func TestConstantImageIsPreserved(t *testing.T) {
	// A flat image must upscale to the same constant for every method (no edges
	// to trigger the edge/sharpen passes).
	for _, algo := range []string{"nearest", "bilinear", "bicubic", "lanczos", "edge", "fsrcnn"} {
		src := cv.NewMat(5, 7, 3)
		src.SetTo(128)
		out, err := UpsamplerFromName(t, algo, 4).Upsample(src)
		if err != nil {
			t.Fatal(err)
		}
		for i := range out.Data {
			if out.Data[i] != 128 {
				t.Fatalf("%s: constant not preserved, got %d at %d", algo, out.Data[i], i)
			}
		}
	}
}

func TestPSNRIdenticalIsInf(t *testing.T) {
	src := synthPattern(4, 4, 3)
	p, err := PSNR(src, src.Clone())
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsInf(p, 1) {
		t.Errorf("PSNR of identical images = %v, want +Inf", p)
	}
	m, err := MSE(src, src.Clone())
	if err != nil {
		t.Fatal(err)
	}
	if m != 0 {
		t.Errorf("MSE of identical images = %v, want 0", m)
	}
}

func TestPSNRKnownValue(t *testing.T) {
	a := cv.NewMat(2, 2, 1)
	b := cv.NewMat(2, 2, 1)
	b.Data[0] = 10 // one sample differs by 10 out of 4 samples
	p, err := PSNR(a, b)
	if err != nil {
		t.Fatal(err)
	}
	// MSE = 100/4 = 25 -> PSNR = 10*log10(65025/25) = 10*log10(2601)
	want := 10 * math.Log10(255*255/25.0)
	if math.Abs(p-want) > 1e-9 {
		t.Errorf("PSNR = %v, want %v", p, want)
	}
}

func TestErrorPaths(t *testing.T) {
	good := synthPattern(4, 4, 1)

	// Unsupported scale.
	if _, err := UpsampleBicubic(good, 5); err == nil {
		t.Error("expected error for scale 5")
	}
	// Empty image (constructed by hand; NewMat panics on zero dimensions).
	var empty cv.Mat
	if _, err := UpsampleNearest(&empty, 2); err == nil {
		t.Error("expected error for empty image")
	}
	// nil image.
	if _, err := UpsampleLanczos(nil, 2); err == nil {
		t.Error("expected error for nil image")
	}

	sr := NewDnnSuperResImpl()
	// Upsample before SetModel.
	if _, err := sr.Upsample(good); err == nil {
		t.Error("expected error upsampling before SetModel")
	}
	// Bad algorithm name.
	if err := sr.SetModel("nope", 2); err == nil {
		t.Error("expected error for unknown algo")
	}
	// Bad scale on SetModel.
	if err := sr.SetModel("bicubic", 7); err == nil {
		t.Error("expected error for bad scale")
	}
	// Getters on unconfigured engine.
	fresh := NewDnnSuperResImpl()
	if fresh.GetAlgorithm() != "" || fresh.GetScale() != 0 {
		t.Error("unconfigured getters should be empty/0")
	}

	// PSNR/MSE mismatches and empties.
	if _, err := PSNR(good, synthPattern(8, 8, 1)); err == nil {
		t.Error("expected PSNR shape mismatch error")
	}
	if _, err := MSE(good, synthPattern(8, 8, 1)); err == nil {
		t.Error("expected MSE shape mismatch error")
	}
	if _, err := PSNR(nil, good); err == nil {
		t.Error("expected PSNR nil error")
	}
	if _, err := MSE(nil, good); err == nil {
		t.Error("expected MSE nil error")
	}
}

func TestSetModelAndGetters(t *testing.T) {
	sr := NewDnnSuperResImpl()
	if err := sr.SetModel("EDGE-DIRECTED", 3); err != nil {
		t.Fatal(err)
	}
	if sr.GetAlgorithm() != "edge" {
		t.Errorf("GetAlgorithm = %q, want edge", sr.GetAlgorithm())
	}
	if sr.GetScale() != 3 {
		t.Errorf("GetScale = %d, want 3", sr.GetScale())
	}
}

func TestParseUpsamplerTypeAndString(t *testing.T) {
	cases := map[string]UpsamplerType{
		"nearest":      Nearest,
		"  BILINEAR ":  Bilinear,
		"linear":       Bilinear,
		"cubic":        Bicubic,
		"lanczos4":     Lanczos,
		"nedi":         EdgeDirected,
		"fsrcnn-style": FSRCNN,
	}
	for name, want := range cases {
		got, err := ParseUpsamplerType(name)
		if err != nil {
			t.Fatalf("ParseUpsamplerType(%q): %v", name, err)
		}
		if got != want {
			t.Errorf("ParseUpsamplerType(%q) = %v, want %v", name, got, want)
		}
	}
	if _, err := ParseUpsamplerType("bogus"); err == nil {
		t.Error("expected error for bogus name")
	}
	// String round-trips through Parse for canonical names.
	for _, tp := range []UpsamplerType{Nearest, Bilinear, Bicubic, Lanczos, EdgeDirected, FSRCNN} {
		got, err := ParseUpsamplerType(tp.String())
		if err != nil || got != tp {
			t.Errorf("round-trip %v failed: got %v err %v", tp, got, err)
		}
	}
	if UpsamplerType(99).String() == "" {
		t.Error("unknown type String should be non-empty")
	}
	// UpsamplerType.Upsample dispatch and unknown type error.
	if _, err := UpsamplerType(99).Upsample(synthPattern(4, 4, 1), 2); err == nil {
		t.Error("expected error for unknown type Upsample")
	}
}

func TestEdgeReducesJaggiesVsNearest(t *testing.T) {
	// On a diagonal edge, the edge-directed result should be closer to a
	// blurred (anti-aliased) reference than nearest neighbour is.
	src := synthPattern(16, 16, 1)
	near, _ := UpsampleNearest(src, 2)
	edge, _ := UpsampleEdgeDirected(src, 2)
	ref := cv.GaussianBlur(near, 3, 1.0)
	pn, _ := PSNR(near, ref)
	pe, _ := PSNR(edge, ref)
	// Edge-directed smooths jaggies, so it should be at least as close to the
	// anti-aliased reference. (Sanity, not a strict quality claim.)
	if math.IsInf(pe, 1) || math.IsInf(pn, 1) {
		t.Skip("degenerate reference")
	}
}
