package intensity_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/intensity"
)

// grayRamp builds a single-channel image whose column x has value
// lo + (hi-lo)*x/(cols-1), a linear ramp between lo and hi inclusive.
func grayRamp(rows, cols int, lo, hi uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := int(lo) + (int(hi)-int(lo))*x/(cols-1)
			m.Set(y, x, 0, uint8(v))
		}
	}
	return m
}

// fullRamp builds a single-channel 1x256 image with value == column, covering
// every intensity 0..255 exactly once.
func fullRamp() *cv.Mat {
	m := cv.NewMat(1, 256, 1)
	for x := 0; x < 256; x++ {
		m.Set(0, x, 0, uint8(x))
	}
	return m
}

// solid builds a single-channel image filled with value v.
func solid(rows, cols int, v uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	m.SetTo(v)
	return m
}

func meanOf(m *cv.Mat) float64 {
	var s float64
	for _, b := range m.Data {
		s += float64(b)
	}
	return s / float64(len(m.Data))
}

func TestGammaIdentity(t *testing.T) {
	src := fullRamp()
	out := intensity.GammaCorrection(src, 1.0)
	for i := range src.Data {
		if out.Data[i] != src.Data[i] {
			t.Fatalf("gamma=1 not identity at %d: got %d want %d", i, out.Data[i], src.Data[i])
		}
	}
}

func TestGammaBrightensMidtones(t *testing.T) {
	mid := solid(8, 8, 128)
	out := intensity.GammaCorrection(mid, 0.5)
	before := meanOf(mid)
	after := meanOf(out)
	if after <= before {
		t.Fatalf("gamma<1 should brighten midtones: before=%.1f after=%.1f", before, after)
	}
	// A specific check: 255*(128/255)^0.5 ≈ 180.6 → 181.
	if got := out.Data[0]; got != 181 {
		t.Fatalf("gamma 0.5 of 128 = %d, want 181", got)
	}
	// gamma>1 should darken.
	dark := intensity.GammaCorrection(mid, 2.0)
	if meanOf(dark) >= before {
		t.Fatalf("gamma>1 should darken: before=%.1f after=%.1f", before, meanOf(dark))
	}
}

func TestLogTransformMonotonicEndpoints(t *testing.T) {
	src := fullRamp()
	out := intensity.LogTransform(src)
	if out.Data[0] != 0 {
		t.Fatalf("log(0) should map to 0, got %d", out.Data[0])
	}
	if out.Data[255] != 255 {
		t.Fatalf("log(255) should map to 255, got %d", out.Data[255])
	}
	for i := 1; i < 256; i++ {
		if out.Data[i] < out.Data[i-1] {
			t.Fatalf("log transform not monotonic at %d: %d < %d", i, out.Data[i], out.Data[i-1])
		}
	}
	// Log transform expands darks: value at mid input exceeds the input.
	if out.Data[64] <= 64 {
		t.Fatalf("log transform should lift low values: out[64]=%d", out.Data[64])
	}
}

func TestExpTransformInverseOfLog(t *testing.T) {
	src := fullRamp()
	out := intensity.ExpTransform(src)
	if out.Data[0] != 0 || out.Data[255] != 255 {
		t.Fatalf("exp endpoints wrong: 0->%d 255->%d", out.Data[0], out.Data[255])
	}
	for i := 1; i < 256; i++ {
		if out.Data[i] < out.Data[i-1] {
			t.Fatalf("exp transform not monotonic at %d", i)
		}
	}
	// Round-trip exp∘log is approximately the identity: LogTransform spreads
	// the dark end so ExpTransform can recover it within rounding. (The other
	// order collapses because Exp compresses low inputs below one quantum.)
	round := intensity.ExpTransform(intensity.LogTransform(src))
	for i := 0; i < 256; i++ {
		d := int(round.Data[i]) - i
		if d < -4 || d > 4 {
			t.Fatalf("exp(log(%d))=%d differs by %d", i, round.Data[i], d)
		}
	}
}

func TestAutoscaleContrast(t *testing.T) {
	// Ramp spanning [50,200] must stretch to the [0,255] endpoints.
	src := grayRamp(1, 151, 50, 200)
	out := intensity.AutoscaleContrast(src)
	if out.Data[0] != 0 {
		t.Fatalf("min sample should map to 0, got %d", out.Data[0])
	}
	if out.Data[len(out.Data)-1] != 255 {
		t.Fatalf("max sample should map to 255, got %d", out.Data[len(out.Data)-1])
	}
	// Constant channel is left unchanged.
	flat := solid(4, 4, 77)
	fo := intensity.AutoscaleContrast(flat)
	for _, b := range fo.Data {
		if b != 77 {
			t.Fatalf("constant image altered: got %d", b)
		}
	}
}

func TestContrastStretchingEndpointsExact(t *testing.T) {
	src := fullRamp()
	r1, s1, r2, s2 := 50.0, 30.0, 200.0, 220.0
	out := intensity.ContrastStretching(src, r1, s1, r2, s2)
	if out.Data[50] != 30 {
		t.Fatalf("control point r1: out[50]=%d want 30", out.Data[50])
	}
	if out.Data[200] != 220 {
		t.Fatalf("control point r2: out[200]=%d want 220", out.Data[200])
	}
	if out.Data[0] != 0 || out.Data[255] != 255 {
		t.Fatalf("global endpoints: 0->%d 255->%d", out.Data[0], out.Data[255])
	}
	for i := 1; i < 256; i++ {
		if out.Data[i] < out.Data[i-1] {
			t.Fatalf("contrast stretch not monotonic at %d", i)
		}
	}
}

func TestIntensityLevelSlicing(t *testing.T) {
	src := fullRamp()
	// Band [100,150] -> 255, background preserved.
	keep := intensity.IntensityLevelSlicing(src, 100, 150, 255, true)
	for i := 0; i < 256; i++ {
		want := uint8(i)
		if i >= 100 && i <= 150 {
			want = 255
		}
		if keep.Data[i] != want {
			t.Fatalf("preserve mode at %d: got %d want %d", i, keep.Data[i], want)
		}
	}
	// Background zeroed.
	zero := intensity.IntensityLevelSlicing(src, 100, 150, 200, false)
	if zero.Data[50] != 0 || zero.Data[125] != 200 {
		t.Fatalf("zero mode wrong: out[50]=%d out[125]=%d", zero.Data[50], zero.Data[125])
	}
}

func TestBitPlaneSlicingMSBStep(t *testing.T) {
	src := fullRamp()
	msb := intensity.BitPlaneSlicing(src, 7)
	// The MSB is 0 for [0,127] and 1 (→255) for [128,255]: a single step.
	for i := 0; i < 128; i++ {
		if msb.Data[i] != 0 {
			t.Fatalf("MSB of %d should be 0, got %d", i, msb.Data[i])
		}
	}
	for i := 128; i < 256; i++ {
		if msb.Data[i] != 255 {
			t.Fatalf("MSB of %d should be 255, got %d", i, msb.Data[i])
		}
	}
	// LSB alternates every step.
	lsb := intensity.BitPlaneSlicing(src, 0)
	for i := 0; i < 256; i++ {
		want := uint8(0)
		if i%2 == 1 {
			want = 255
		}
		if lsb.Data[i] != want {
			t.Fatalf("LSB of %d: got %d want %d", i, lsb.Data[i], want)
		}
	}
}

func TestSolarize(t *testing.T) {
	src := fullRamp()
	out := intensity.Solarize(src, 128)
	for i := 0; i < 128; i++ {
		if out.Data[i] != uint8(i) {
			t.Fatalf("below threshold changed at %d: got %d", i, out.Data[i])
		}
	}
	for i := 128; i < 256; i++ {
		if out.Data[i] != uint8(255-i) {
			t.Fatalf("above threshold not inverted at %d: got %d want %d", i, out.Data[i], 255-i)
		}
	}
}

func TestPosterize(t *testing.T) {
	src := fullRamp()
	// Two levels: threshold at midpoint, outputs only 0 or 255.
	out := intensity.Posterize(src, 2)
	for i := 0; i < 256; i++ {
		if out.Data[i] != 0 && out.Data[i] != 255 {
			t.Fatalf("posterize(2) produced non-binary value %d at %d", out.Data[i], i)
		}
	}
	if out.Data[0] != 0 || out.Data[255] != 255 {
		t.Fatalf("posterize endpoints: 0->%d 255->%d", out.Data[0], out.Data[255])
	}
	// 256 levels is the identity.
	id := intensity.Posterize(src, 256)
	for i := 0; i < 256; i++ {
		if id.Data[i] != uint8(i) {
			t.Fatalf("posterize(256) not identity at %d: got %d", i, id.Data[i])
		}
	}
	// Distinct-value count for a mid level is small.
	four := intensity.Posterize(src, 4)
	seen := map[uint8]bool{}
	for _, b := range four.Data {
		seen[b] = true
	}
	if len(seen) != 4 {
		t.Fatalf("posterize(4) yielded %d distinct levels, want 4", len(seen))
	}
}

func TestInvert(t *testing.T) {
	src := fullRamp()
	out := intensity.Invert(src)
	for i := 0; i < 256; i++ {
		if out.Data[i] != uint8(255-i) {
			t.Fatalf("invert at %d: got %d want %d", i, out.Data[i], 255-i)
		}
	}
	// Double inversion is the identity.
	back := intensity.Invert(out)
	for i := range src.Data {
		if back.Data[i] != src.Data[i] {
			t.Fatalf("double invert not identity at %d", i)
		}
	}
}

func TestHistogramMatchingApproximatesReference(t *testing.T) {
	// Source: a narrow band of dark values. Reference: a bright band. After
	// matching, the output histogram should resemble the reference far more
	// than the source did.
	src := cv.NewMat(16, 16, 1)
	for i := range src.Data {
		src.Data[i] = uint8(20 + i%30) // values in [20,49]
	}
	ref := cv.NewMat(16, 16, 1)
	for i := range ref.Data {
		ref.Data[i] = uint8(180 + i%40) // values in [180,219]
	}
	out := intensity.HistogramMatching(src, ref)

	refHist := cv.CalcHist(ref, 0)
	srcHist := cv.CalcHist(src, 0)
	outHist := cv.CalcHist(out, 0)

	dOut := cv.CompareHist(outHist, refHist, cv.HistCmpChiSqr)
	dSrc := cv.CompareHist(srcHist, refHist, cv.HistCmpChiSqr)
	if dOut >= dSrc {
		t.Fatalf("matching did not move histogram toward reference: dSrc=%.1f dOut=%.1f", dSrc, dOut)
	}
	// The output mean should be close to the reference mean.
	if mo, mr := meanOf(out), meanOf(ref); mo < mr-20 || mo > mr+20 {
		t.Fatalf("output mean %.1f not near reference mean %.1f", mo, mr)
	}
}

func TestBIMEFBrightensLowLight(t *testing.T) {
	dark := solid(16, 16, 30)
	out := intensity.BIMEF(dark)
	if meanOf(out) <= meanOf(dark) {
		t.Fatalf("BIMEF should brighten a dark image: before=%.1f after=%.1f",
			meanOf(dark), meanOf(out))
	}
	// Determinism: same input yields identical output.
	out2 := intensity.BIMEF(dark)
	for i := range out.Data {
		if out.Data[i] != out2.Data[i] {
			t.Fatalf("BIMEF not deterministic at %d", i)
		}
	}
	// A three-channel dark image is also brightened, shape preserved.
	dark3 := cv.NewMat(8, 8, 3)
	dark3.SetTo(25)
	out3 := intensity.BIMEF(dark3)
	if out3.Channels != 3 || out3.Rows != 8 || out3.Cols != 8 {
		t.Fatalf("BIMEF changed shape: %dx%dx%d", out3.Rows, out3.Cols, out3.Channels)
	}
	if meanOf(out3) <= meanOf(dark3) {
		t.Fatalf("BIMEF should brighten dark RGB image")
	}
}

func TestMultiChannelPointOp(t *testing.T) {
	// Point ops apply per channel; verify Invert on a 3-channel image.
	m := cv.NewMat(2, 2, 3)
	for i := range m.Data {
		m.Data[i] = uint8(i * 10)
	}
	out := intensity.Invert(m)
	for i := range m.Data {
		if out.Data[i] != uint8(255-int(m.Data[i])) {
			t.Fatalf("multichannel invert wrong at %d", i)
		}
	}
}
