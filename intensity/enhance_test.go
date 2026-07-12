package intensity_test

import (
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/intensity"
)

// clampByte clamps an int into the [0,255] byte range.
func clampByte(v int) uint8 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// --- shared test helpers -------------------------------------------------

// stdOf returns the population standard deviation of every sample of m.
func stdOf(m *cv.Mat) float64 {
	mean := meanOf(m)
	var sq float64
	for _, b := range m.Data {
		d := float64(b) - mean
		sq += d * d
	}
	return math.Sqrt(sq / float64(len(m.Data)))
}

// lumaOf returns the Rec. 601 luminance of pixel p of a 1- or 3-channel Mat.
func lumaOf(m *cv.Mat, p int) float64 {
	if m.Channels == 1 {
		return float64(m.Data[p])
	}
	b := p * m.Channels
	return 0.299*float64(m.Data[b]) + 0.587*float64(m.Data[b+1]) + 0.114*float64(m.Data[b+2])
}

// lumaCorr returns the Pearson correlation between the per-pixel luminance of a
// and b, which must share dimensions.
func lumaCorr(a, b *cv.Mat) float64 {
	n := a.Total()
	la := make([]float64, n)
	lb := make([]float64, n)
	for p := 0; p < n; p++ {
		la[p] = lumaOf(a, p)
		lb[p] = lumaOf(b, p)
	}
	var ma, mb float64
	for p := 0; p < n; p++ {
		ma += la[p]
		mb += lb[p]
	}
	ma /= float64(n)
	mb /= float64(n)
	var num, da, db float64
	for p := 0; p < n; p++ {
		x := la[p] - ma
		y := lb[p] - mb
		num += x * y
		da += x * x
		db += y * y
	}
	if da == 0 || db == 0 {
		return 0
	}
	return num / math.Sqrt(da*db)
}

// darkTextured builds a dark, structured single-channel image with a peaked
// (Gaussian) histogram centred near 24: most pixels are dark, with a spread of
// levels, so exposure-ratio entropy maximisation has a genuine interior optimum.
// The seed is fixed, so the image is deterministic.
func darkTextured(rows, cols int) *cv.Mat {
	rng := rand.New(rand.NewSource(9))
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := 24 + int(math.Round(rng.NormFloat64()*8))
			m.Set(y, x, 0, clampByte(v))
		}
	}
	return m
}

// lowContrastGray builds a single-channel image with a peaked histogram in a
// narrow, mid-dark band centred near 50 (roughly [30,75]): low contrast but not
// flat, and non-uniform. The seed is fixed.
func lowContrastGray(rows, cols int) *cv.Mat {
	rng := rand.New(rand.NewSource(3))
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := 50 + int(math.Round(rng.NormFloat64()*7))
			m.Set(y, x, 0, clampByte(v))
		}
	}
	return m
}

// darkColor builds a dark, structured three-channel image with peaked per-channel
// histograms. The seed is fixed.
func darkColor(rows, cols int) *cv.Mat {
	rng := rand.New(rand.NewSource(17))
	m := cv.NewMat(rows, cols, 3)
	center := [3]int{26, 20, 16}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			px := make([]uint8, 3)
			for c := 0; c < 3; c++ {
				px[c] = clampByte(center[c] + int(math.Round(rng.NormFloat64()*7)))
			}
			m.SetPixel(y, x, px)
		}
	}
	return m
}

func sameShape(t *testing.T, a, b *cv.Mat) {
	t.Helper()
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		t.Fatalf("shape mismatch: %dx%dx%d vs %dx%dx%d",
			a.Rows, a.Cols, a.Channels, b.Rows, b.Cols, b.Channels)
	}
}

func sameData(a, b *cv.Mat) bool {
	if len(a.Data) != len(b.Data) {
		return false
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			return false
		}
	}
	return true
}

// --- LUT builders --------------------------------------------------------

func TestGammaLUTMatchesGammaCorrection(t *testing.T) {
	if intensity.GammaLUT(1.0)[200] != 200 {
		t.Fatalf("gamma=1 LUT not identity at 200")
	}
	lut := intensity.GammaLUT(2.0)
	src := fullRamp()
	viaLUT := intensity.ApplyLUT(src, lut)
	viaGamma := intensity.GammaCorrection(src, 2.0)
	if !sameData(viaLUT, viaGamma) {
		t.Fatalf("ApplyLUT(GammaLUT) disagrees with GammaCorrection")
	}
}

func TestApplyLUTPanicsOnBadTable(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("ApplyLUT should panic on non-256 table")
		}
	}()
	intensity.ApplyLUT(fullRamp(), make([]uint8, 10))
}

// --- adaptive gamma ------------------------------------------------------

func TestAutoGammaBrightensAndDarkens(t *testing.T) {
	dark := solid(8, 8, 50)
	bright := solid(8, 8, 200)
	if meanOf(intensity.AutoGamma(dark)) <= meanOf(dark) {
		t.Fatalf("AutoGamma should brighten a dark image")
	}
	if meanOf(intensity.AutoGamma(bright)) >= meanOf(bright) {
		t.Fatalf("AutoGamma should darken a bright image")
	}
	if g := intensity.AutoGammaValue(solid(4, 4, 128)); g < 0.9 || g > 1.1 {
		t.Fatalf("AutoGammaValue of mid-grey = %.3f, want ~1", g)
	}
}

func TestAGCWDImprovesLowContrast(t *testing.T) {
	src := lowContrastGray(16, 16)
	out := intensity.AGCWD(src, 0.5)
	if meanOf(out) <= meanOf(src) {
		t.Fatalf("AGCWD should brighten a dark low-contrast image: %.1f -> %.1f",
			meanOf(src), meanOf(out))
	}
	if stdOf(out) <= stdOf(src) {
		t.Fatalf("AGCWD should increase contrast: std %.2f -> %.2f", stdOf(src), stdOf(out))
	}
	// Determinism.
	out2 := intensity.AGCWD(src, 0.5)
	if !sameData(out, out2) {
		t.Fatalf("AGCWD not deterministic")
	}
	// Flat image is returned unchanged.
	flat := solid(8, 8, 90)
	if !sameData(intensity.AGCWD(flat, 0.5), flat) {
		t.Fatalf("AGCWD altered a flat image")
	}
}

// --- retinex -------------------------------------------------------------

func TestSingleScaleRetinexBrightensPreservesStructure(t *testing.T) {
	src := darkTextured(24, 24)
	out := intensity.SingleScaleRetinex(src, 20)
	sameShape(t, src, out)
	if meanOf(out) <= meanOf(src) {
		t.Fatalf("SSR should brighten a dark image: %.1f -> %.1f", meanOf(src), meanOf(out))
	}
	if c := lumaCorr(src, out); c < 0.5 {
		t.Fatalf("SSR should preserve structure, correlation=%.3f", c)
	}
	if !sameData(out, intensity.SingleScaleRetinex(src, 20)) {
		t.Fatalf("SSR not deterministic")
	}
}

func TestMultiScaleRetinexColor(t *testing.T) {
	src := darkColor(24, 24)
	scales := []float64{5, 15, 40}
	out := intensity.MultiScaleRetinex(src, scales)
	sameShape(t, src, out)
	if meanOf(out) <= meanOf(src) {
		t.Fatalf("MSR should brighten a dark image")
	}
	if c := lumaCorr(src, out); c < 0.4 {
		t.Fatalf("MSR should preserve structure, correlation=%.3f", c)
	}
	if !sameData(out, intensity.MultiScaleRetinex(src, scales)) {
		t.Fatalf("MSR not deterministic")
	}
}

func TestMSRCRBrightensColor(t *testing.T) {
	src := darkColor(24, 24)
	out := intensity.MSRCR(src, intensity.DefaultRetinexScales())
	sameShape(t, src, out)
	if meanOf(out) <= meanOf(src) {
		t.Fatalf("MSRCR should brighten a dark image")
	}
	if !sameData(out, intensity.MSRCR(src, intensity.DefaultRetinexScales())) {
		t.Fatalf("MSRCR not deterministic")
	}
}

func TestRetinexPanicsOnBadScales(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on empty scales")
		}
	}()
	intensity.MultiScaleRetinex(darkTextured(4, 4), nil)
}

// --- auto contrast / levels ---------------------------------------------

func TestAutoContrastStretchesEndpoints(t *testing.T) {
	src := grayRamp(1, 151, 50, 200)
	out := intensity.AutoContrast(src, 0)
	if out.Data[0] != 0 || out.Data[len(out.Data)-1] != 255 {
		t.Fatalf("AutoContrast endpoints: %d..%d", out.Data[0], out.Data[len(out.Data)-1])
	}
}

func TestAutoLevelsStretches(t *testing.T) {
	src := grayRamp(1, 151, 50, 200)
	out := intensity.AutoLevels(src, 0, 0)
	if out.Data[0] != 0 || out.Data[len(out.Data)-1] != 255 {
		t.Fatalf("AutoLevels endpoints: %d..%d", out.Data[0], out.Data[len(out.Data)-1])
	}
}

func TestContrastLimitedStretchCapsGain(t *testing.T) {
	src := lowContrastGray(16, 16) // span ~12
	limited := intensity.ContrastLimitedStretch(src, 0, 3)
	full := intensity.ContrastLimitedStretch(src, 0, 1000)
	rng := func(m *cv.Mat) int {
		lo, hi := 255, 0
		for _, b := range m.Data {
			if int(b) < lo {
				lo = int(b)
			}
			if int(b) > hi {
				hi = int(b)
			}
		}
		return hi - lo
	}
	if rng(limited) >= rng(full) {
		t.Fatalf("gain cap did not limit the stretch: limited=%d full=%d", rng(limited), rng(full))
	}
	if rng(full) < 200 {
		t.Fatalf("uncapped stretch should span most of the range, got %d", rng(full))
	}
}

// --- tone curve ----------------------------------------------------------

func TestToneCurveControlPointsExact(t *testing.T) {
	pts := []intensity.CurvePoint{{In: 0, Out: 0}, {In: 64, Out: 32}, {In: 192, Out: 224}, {In: 255, Out: 255}}
	lut := intensity.ToneCurveLUT(pts)
	for _, p := range pts {
		if got := lut[int(p.In)]; math.Abs(float64(got)-p.Out) > 0.5 {
			t.Fatalf("tone curve at %v = %d, want %v", p.In, got, p.Out)
		}
	}
	// Monotonic non-decreasing for this monotone control set.
	for i := 1; i < 256; i++ {
		if lut[i] < lut[i-1] {
			t.Fatalf("tone curve not monotonic at %d", i)
		}
	}
	// Applied per channel.
	out := intensity.ToneCurve(fullRamp(), pts)
	if out.Data[64] != 32 || out.Data[192] != 224 {
		t.Fatalf("ToneCurve did not hit control points: %d %d", out.Data[64], out.Data[192])
	}
}

func TestToneCurvePanicsOnNonIncreasing(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on non-increasing control points")
		}
	}()
	intensity.ToneCurveLUT([]intensity.CurvePoint{{In: 100, Out: 0}, {In: 100, Out: 255}})
}

// --- dodge and burn ------------------------------------------------------

func TestDodgeAndBurnFlattensIllumination(t *testing.T) {
	src := grayRamp(16, 64, 30, 220)
	out := intensity.DodgeAndBurn(src, 1.0, 5)
	sameShape(t, src, out)
	leftBefore := meanOf(src.Region(0, 0, 16, 16))
	rightBefore := meanOf(src.Region(0, 48, 16, 16))
	leftAfter := meanOf(out.Region(0, 0, 16, 16))
	rightAfter := meanOf(out.Region(0, 48, 16, 16))
	if math.Abs(rightAfter-leftAfter) >= math.Abs(rightBefore-leftBefore) {
		t.Fatalf("dodge/burn did not flatten illumination: before=%.1f after=%.1f",
			rightBefore-leftBefore, rightAfter-leftAfter)
	}
	// amount 0 is the identity.
	if !sameData(intensity.DodgeAndBurn(src, 0, 5), src) {
		t.Fatalf("amount 0 should be identity")
	}
}

// --- unsharp mask --------------------------------------------------------

func TestUnsharpMaskOvershootsEdge(t *testing.T) {
	// Step edge: left 100, right 150.
	src := cv.NewMat(8, 16, 1)
	for y := 0; y < 8; y++ {
		for x := 0; x < 16; x++ {
			v := uint8(100)
			if x >= 8 {
				v = 150
			}
			src.Set(y, x, 0, v)
		}
	}
	out := intensity.UnsharpMask(src, 2, 1.5, 0)
	var lo, hi uint8 = 255, 0
	for _, b := range out.Data {
		if b < lo {
			lo = b
		}
		if b > hi {
			hi = b
		}
	}
	if hi <= 150 || lo >= 100 {
		t.Fatalf("unsharp should overshoot beyond [100,150], got [%d,%d]", lo, hi)
	}
	// Flat image with threshold above any difference is the identity.
	flat := solid(8, 8, 120)
	if !sameData(intensity.UnsharpMask(flat, 2, 1.5, 0), flat) {
		t.Fatalf("unsharp altered a flat image")
	}
}

// --- CLAHE colour --------------------------------------------------------

func TestCLAHEColorGrayMatchesRoot(t *testing.T) {
	gray := cv.NewMat(16, 16, 1)
	for i := range gray.Data {
		gray.Data[i] = uint8(60 + (i*7)%40)
	}
	if !sameData(intensity.CLAHEColor(gray, 2, 4), cv.CLAHE(gray, 2, 4)) {
		t.Fatalf("CLAHEColor on grayscale should match cv.CLAHE")
	}
}

func TestCLAHEColorPreservesGrayHue(t *testing.T) {
	// An achromatic colour image (R==G==B) must stay achromatic.
	m := cv.NewMat(16, 16, 3)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			v := uint8(50 + ((x + y) % 8 * 6))
			m.SetPixel(y, x, []uint8{v, v, v})
		}
	}
	out := intensity.CLAHEColor(m, 3, 4)
	sameShape(t, m, out)
	for p := 0; p < out.Total(); p++ {
		b := p * 3
		if out.Data[b] != out.Data[b+1] || out.Data[b+1] != out.Data[b+2] {
			t.Fatalf("CLAHEColor shifted hue at pixel %d: %v", p, out.Data[b:b+3])
		}
	}
}

// --- log adaptive tonemap ------------------------------------------------

func TestLogAdaptiveTonemapBrightensMonotone(t *testing.T) {
	ramp := fullRamp()
	out := intensity.LogAdaptiveTonemap(ramp, 0.85)
	if meanOf(out) <= meanOf(ramp) {
		t.Fatalf("tonemap should lift shadows: %.1f -> %.1f", meanOf(ramp), meanOf(out))
	}
	for i := 1; i < 256; i++ {
		if out.Data[i] < out.Data[i-1] {
			t.Fatalf("tonemap not monotonic at %d", i)
		}
	}
	if out.Data[255] != 255 {
		t.Fatalf("tonemap should map the max to 255, got %d", out.Data[255])
	}
	if !sameData(out, intensity.LogAdaptiveTonemap(ramp, 0.85)) {
		t.Fatalf("tonemap not deterministic")
	}
}

// --- refined BIMEF -------------------------------------------------------

func TestBIMEFRefinedBrightensPreservesStructure(t *testing.T) {
	src := darkTextured(24, 24)
	out := intensity.BIMEFRefined(src)
	sameShape(t, src, out)
	if meanOf(out) <= meanOf(src) {
		t.Fatalf("BIMEFRefined should brighten: %.1f -> %.1f", meanOf(src), meanOf(out))
	}
	if c := lumaCorr(src, out); c < 0.5 {
		t.Fatalf("BIMEFRefined should preserve structure, correlation=%.3f", c)
	}
	if !sameData(out, intensity.BIMEFRefined(src)) {
		t.Fatalf("BIMEFRefined not deterministic")
	}
	// Three-channel path.
	c3 := darkColor(20, 20)
	o3 := intensity.BIMEFRefined(c3)
	sameShape(t, c3, o3)
	if meanOf(o3) <= meanOf(c3) {
		t.Fatalf("BIMEFRefined should brighten a dark RGB image")
	}
}

func TestBIMEFWithParamsValidates(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on KMin < 1")
		}
	}()
	p := intensity.DefaultBIMEFParams()
	p.KMin = 0.5
	intensity.BIMEFWithParams(darkTextured(4, 4), p)
}

func TestBIMEFRefinedDiffersFromSimple(t *testing.T) {
	src := darkTextured(20, 20)
	if sameData(intensity.BIMEFRefined(src), intensity.BIMEF(src)) {
		t.Fatalf("refined and simple BIMEF unexpectedly identical")
	}
}
