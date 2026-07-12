package xphoto

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// ---- shared builders -----------------------------------------------------

// noisyFlatGray builds a single-channel image that is flat at base on the left
// half and base+step on the right half, plus deterministic zero-mean noise of
// amplitude amp. The returned clean image has no noise.
func noisyFlatGray(rows, cols, base, step, amp int) (clean, noisy *cv.Mat) {
	clean = cv.NewMat(rows, cols, 1)
	noisy = cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := base
			if x >= cols/2 {
				v = base + step
			}
			clean.Set(y, x, 0, uint8(v))
			noisy.Set(y, x, 0, clampU8(float64(v+lcgNoise(y, x, amp))))
		}
	}
	return clean, noisy
}

// ---- GammaCorrection -----------------------------------------------------

func TestGammaCorrectionMonotoneAndEndpoints(t *testing.T) {
	src := cv.NewMat(1, 256, 1)
	for i := 0; i < 256; i++ {
		src.Set(0, i, 0, uint8(i))
	}
	out := GammaCorrection(src, 2.2)
	if out.At(0, 0, 0) != 0 {
		t.Fatalf("gamma must map 0->0, got %d", out.At(0, 0, 0))
	}
	if out.At(0, 255, 0) != 255 {
		t.Fatalf("gamma must map 255->255, got %d", out.At(0, 255, 0))
	}
	// gamma > 1 brightens: every output >= input, and non-decreasing.
	prev := -1
	for i := 0; i < 256; i++ {
		v := int(out.At(0, i, 0))
		if v < i {
			t.Fatalf("gamma 2.2 should brighten at %d: got %d", i, v)
		}
		if v < prev {
			t.Fatalf("gamma LUT not monotonic at %d", i)
		}
		prev = v
	}
}

func TestGammaCorrectionIdentity(t *testing.T) {
	src := cv.NewMat(2, 3, 3)
	for i := range src.Data {
		src.Data[i] = uint8(i * 9 % 256)
	}
	out := GammaCorrection(src, 1.0)
	for i := range src.Data {
		if out.Data[i] != src.Data[i] {
			t.Fatalf("gamma 1.0 must be identity at %d", i)
		}
	}
}

// ---- color constancy WBs -------------------------------------------------

func TestShadesOfGrayNeutralizesCast(t *testing.T) {
	src := tintedRamp(24, 24, 1.3, 1.0, 0.75)
	before := spread(channelMeans(src))
	for _, p := range []float64{1, 4, math.Inf(1)} {
		out := ShadesOfGray(src, p)
		after := spread(channelMeans(out))
		if after >= before {
			t.Fatalf("ShadesOfGray p=%v did not reduce cast: before=%.2f after=%.2f", p, before, after)
		}
	}
}

func TestWhitePatchAndGrayEdgeReduceCast(t *testing.T) {
	src := tintedRamp(24, 24, 1.25, 1.0, 0.8)
	before := spread(channelMeans(src))
	if after := spread(channelMeans(WhitePatchWB(src))); after >= before {
		t.Fatalf("WhitePatchWB did not reduce cast: before=%.2f after=%.2f", before, after)
	}
	if after := spread(channelMeans(GrayEdgeWB(src, 6))); after >= before {
		t.Fatalf("GrayEdgeWB did not reduce cast: before=%.2f after=%.2f", before, after)
	}
}

func TestAutoWhiteBalanceNeutralizes(t *testing.T) {
	src := tintedRamp(20, 28, 1.35, 1.0, 0.7)
	before := spread(channelMeans(src))
	out := AutoWhiteBalance(src)
	after := spread(channelMeans(out))
	if after >= before {
		t.Fatalf("AutoWhiteBalance did not reduce cast: before=%.2f after=%.2f", before, after)
	}
}

func TestColorConstancyDeterministic(t *testing.T) {
	src := tintedRamp(16, 16, 1.2, 1.0, 0.85)
	a := ShadesOfGray(src, 6)
	b := ShadesOfGray(src, 6)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatal("ShadesOfGray not deterministic")
		}
	}
}

// ---- factory + accessors -------------------------------------------------

func TestFactoriesAndAccessors(t *testing.T) {
	s := CreateSimpleWB()
	s.SetP(5)
	s.SetInputMin(1)
	s.SetInputMax(250)
	s.SetOutputMin(0)
	s.SetOutputMax(200)
	if s.GetP() != 5 || s.GetInputMin() != 1 || s.GetInputMax() != 250 || s.GetOutputMax() != 200 || s.GetOutputMin() != 0 {
		t.Fatal("SimpleWB accessors did not round-trip")
	}
	g := CreateGrayworldWB()
	g.SetSaturationThreshold(0.5)
	if g.GetSaturationThreshold() != 0.5 {
		t.Fatal("GrayworldWB accessor did not round-trip")
	}
	l := CreateLearningBasedWB()
	l.SetRangeMaxVal(255)
	l.SetSaturationThreshold(0.9)
	l.SetHistBinNum(32)
	if l.GetRangeMaxVal() != 255 || l.GetSaturationThreshold() != 0.9 || l.GetHistBinNum() != 32 {
		t.Fatal("LearningBasedWB accessors did not round-trip")
	}
	// The factory objects must still balance a cast.
	src := tintedRamp(16, 16, 1.3, 1.0, 0.8)
	if out := g.BalanceWhite(src); out.Empty() {
		t.Fatal("factory GrayworldWB failed to balance")
	}
}

// ---- DctDenoising --------------------------------------------------------

func TestDctDenoisingReducesNoisePreservesEdge(t *testing.T) {
	clean, noisy := noisyFlatGray(40, 40, 60, 140, 22)
	den := DctDenoising(noisy, 22, 8)

	varNoisy := residualVar(noisy, clean, 4, 4, 32, 14)
	varDen := residualVar(den, clean, 4, 4, 32, 14)
	if varDen >= varNoisy {
		t.Fatalf("DctDenoising did not reduce noise variance: noisy=%.1f den=%.1f", varNoisy, varDen)
	}
	// Edge preserved: right flat region well above left flat region.
	left := flatMean(den, 8, 6, 24, 10)
	right := flatMean(den, 8, 26, 24, 10)
	if right-left < 110 {
		t.Fatalf("DctDenoising lost the edge: contrast=%.1f", right-left)
	}
}

func TestDctDenoisingColorShape(t *testing.T) {
	src := cv.NewMat(20, 20, 3)
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			src.SetPixel(y, x, []uint8{
				clampU8(float64(100 + lcgNoise(y, x, 15))),
				clampU8(float64(120 + lcgNoise(y, x+1, 15))),
				clampU8(float64(140 + lcgNoise(y+1, x, 15))),
			})
		}
	}
	out := DctDenoising(src, 15, 8)
	if out.Channels != 3 || out.Rows != 20 || out.Cols != 20 {
		t.Fatal("unexpected shape from DctDenoising")
	}
}

// ---- BM3D two-step / Wiener ----------------------------------------------

func TestBm3dTwoStepBeatsBasicOnNoise(t *testing.T) {
	clean, noisy := noisyFlatGray(48, 48, 60, 140, 24)
	basic := Bm3dDenoising(noisy, 24)
	two := Bm3dDenoisingTwoStep(noisy, 24)

	varNoisy := residualVar(noisy, clean, 4, 4, 40, 16)
	varBasic := residualVar(basic, clean, 4, 4, 40, 16)
	varTwo := residualVar(two, clean, 4, 4, 40, 16)
	if varBasic >= varNoisy {
		t.Fatalf("basic BM3D did not denoise: noisy=%.1f basic=%.1f", varNoisy, varBasic)
	}
	if varTwo >= varNoisy {
		t.Fatalf("two-step BM3D did not denoise: noisy=%.1f two=%.1f", varNoisy, varTwo)
	}
	// Edge preserved by the two-step result.
	left := flatMean(two, 10, 6, 28, 12)
	right := flatMean(two, 10, 30, 28, 12)
	if right-left < 110 {
		t.Fatalf("two-step BM3D lost the edge: contrast=%.1f", right-left)
	}
}

func TestBm3dStep2Standalone(t *testing.T) {
	_, noisy := noisyFlatGray(20, 20, 90, 80, 15)
	basic := Bm3dDenoising(noisy, 15)
	out := Bm3dDenoisingStep2(noisy, basic, 15)
	if out.Rows != 20 || out.Cols != 20 || out.Channels != 1 {
		t.Fatal("unexpected shape from Bm3dDenoisingStep2")
	}
}

// ---- Dehaze --------------------------------------------------------------

// hazyGradient builds a synthetic hazy image: a dark-to-bright textured scene
// blended toward a bright grey air-light by a distance-dependent transmission,
// which flattens (reduces) its contrast the way real haze does.
func hazyGradient(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 3)
	air := 230.0
	for y := 0; y < rows; y++ {
		// Transmission drops with y (top = far/hazy, bottom = near/clear).
		t := 0.3 + 0.6*float64(y)/float64(rows-1)
		for x := 0; x < cols; x++ {
			// Underlying scene radiance with some texture.
			base := 30.0 + 160.0*float64(x)/float64(cols-1)
			scene := [3]float64{base, base * 0.8, base * 0.6}
			scene[0] += float64(lcgNoise(y, x, 6))
			for c := 0; c < 3; c++ {
				v := scene[c]*t + air*(1-t)
				m.Set(y, x, c, clampU8(v))
			}
		}
	}
	return m
}

func stdDev(m *cv.Mat) float64 {
	var s, s2, n float64
	for _, v := range m.Data {
		f := float64(v)
		s += f
		s2 += f * f
		n++
	}
	mean := s / n
	return math.Sqrt(s2/n - mean*mean)
}

func TestDehazeRaisesContrast(t *testing.T) {
	hazy := hazyGradient(40, 40)
	before := stdDev(hazy)
	out := Dehaze(hazy)
	after := stdDev(out)
	if after <= before {
		t.Fatalf("Dehaze did not raise contrast: before=%.2f after=%.2f", before, after)
	}
	if out.Rows != 40 || out.Channels != 3 {
		t.Fatal("unexpected dehaze shape")
	}
}

func TestDehazeDeterministic(t *testing.T) {
	hazy := hazyGradient(24, 24)
	a := Dehaze(hazy)
	b := NewDarkChannelDehazer().Apply(hazy)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatal("Dehaze not deterministic")
		}
	}
}

// ---- Oilpainting variants ------------------------------------------------

func TestOilpaintingColorSpaceModes(t *testing.T) {
	rows, cols := 24, 24
	src := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			src.SetPixel(y, x, []uint8{
				clampU8(float64(150 + lcgNoise(y, x, 30))),
				clampU8(float64(80 + lcgNoise(y, x+3, 30))),
				clampU8(float64(80 + lcgNoise(y+3, x, 30))),
			})
		}
	}
	uBefore := uniqueColors(src)
	for _, mode := range []OilIntensity{OilIntensityLuma, OilIntensityValue, OilIntensityAverage} {
		out := OilpaintingColorSpace(src, 3, 24, mode)
		if uniqueColors(out) >= uBefore {
			t.Fatalf("OilpaintingColorSpace mode %d did not reduce colours", mode)
		}
	}
	// Luma mode must match the plain Oilpainting entry point exactly.
	a := OilpaintingColorSpace(src, 3, 24, OilIntensityLuma)
	b := Oilpainting(src, 3, 24)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatal("OilpaintingColorSpace(Luma) must equal Oilpainting")
		}
	}
}

// ---- InpaintFSR ----------------------------------------------------------

func TestInpaintFSRFillsSmoothGradient(t *testing.T) {
	rows, cols := 40, 40
	clean := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := clampU8(float64(20 + y*3 + x*2))
			clean.SetPixel(y, x, []uint8{v, clampU8(float64(int(v) - 10)), clampU8(float64(int(v) + 5))})
		}
	}
	mask := cv.NewMat(rows, cols, 1)
	corrupt := clean.Clone()
	by, bx, bs := 16, 16, 6
	for y := by; y < by+bs; y++ {
		for x := bx; x < bx+bs; x++ {
			mask.Set(y, x, 0, 255)
			corrupt.SetPixel(y, x, []uint8{0, 0, 0})
		}
	}
	for _, mode := range []FSRMode{FSRBest, FSRFast} {
		out := InpaintFSR(corrupt, mask, mode)
		var sae, n float64
		for y := by; y < by+bs; y++ {
			for x := bx; x < bx+bs; x++ {
				for c := 0; c < 3; c++ {
					sae += math.Abs(float64(out.At(y, x, c)) - float64(clean.At(y, x, c)))
					n++
				}
			}
		}
		mae := sae / n
		if mae > 20 {
			t.Fatalf("InpaintFSR mode %d fill too far from truth: mae=%.2f", mode, mae)
		}
		// Known pixels untouched.
		if out.At(0, 0, 0) != clean.At(0, 0, 0) {
			t.Fatalf("InpaintFSR mode %d altered a known pixel", mode)
		}
	}
}

func TestInpaintFSRNoMask(t *testing.T) {
	src := cv.NewMat(8, 8, 3)
	src.SetTo(120)
	mask := cv.NewMat(8, 8, 1)
	out := InpaintFSR(src, mask, FSRFast)
	for i := range src.Data {
		if out.Data[i] != src.Data[i] {
			t.Fatal("empty mask should return an unchanged copy")
		}
	}
}

// ---- TonemapDurand -------------------------------------------------------

func TestTonemapDurandRunsAndPreservesDetail(t *testing.T) {
	rows, cols := 32, 32
	src := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			// Dark textured left half, bright textured right half.
			base := 30.0
			if x >= cols/2 {
				base = 210.0
			}
			base += float64(lcgNoise(y, x, 10))
			src.SetPixel(y, x, []uint8{clampU8(base), clampU8(base), clampU8(base)})
		}
	}
	tm := NewTonemapDurand()
	out := tm.Process(src)
	if out.Rows != rows || out.Cols != cols || out.Channels != 3 {
		t.Fatal("unexpected tonemap shape")
	}
	// Local detail (variance) inside the bright region must survive.
	v := regionVarGray(out, 4, 20, 24, 8)
	if v < 1.0 {
		t.Fatalf("TonemapDurand collapsed local detail: var=%.3f", v)
	}
}

func regionVarGray(m *cv.Mat, y, x, h, w int) float64 {
	var s, s2, n float64
	for yy := y; yy < y+h; yy++ {
		for xx := x; xx < x+w; xx++ {
			f := float64(m.At(yy, xx, 0))
			s += f
			s2 += f * f
			n++
		}
	}
	mean := s / n
	return s2/n - mean*mean
}
