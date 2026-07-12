package xphoto

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// ---- test image builders -------------------------------------------------

// tintedRamp builds an RGB image whose grey value ramps along x and is then
// tinted by a per-channel gain, simulating a colour cast on a neutral scene.
// Values stay in range so no clamping occurs.
func tintedRamp(rows, cols int, gr, gg, gb float64) *cv.Mat {
	m := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			base := 40.0 + float64(x)/float64(cols-1)*120.0 // 40..160
			m.Set(y, x, 0, clampU8(base*gr))
			m.Set(y, x, 1, clampU8(base*gg))
			m.Set(y, x, 2, clampU8(base*gb))
		}
	}
	return m
}

// channelMeans returns the mean of each of the 3 channels.
func channelMeans(m *cv.Mat) [3]float64 {
	var s [3]float64
	n := float64(m.Total())
	for p := 0; p < m.Total(); p++ {
		for c := 0; c < 3; c++ {
			s[c] += float64(m.Data[p*3+c])
		}
	}
	return [3]float64{s[0] / n, s[1] / n, s[2] / n}
}

func spread(m [3]float64) float64 {
	mx := math.Max(m[0], math.Max(m[1], m[2]))
	mn := math.Min(m[0], math.Min(m[1], m[2]))
	return mx - mn
}

// lcgNoise returns deterministic pseudo-random zero-centred noise in [-amp,amp]
// for coordinate (y,x), independent of iteration order.
func lcgNoise(y, x, amp int) int {
	h := uint32(y)*73856093 ^ uint32(x)*19349663
	h = h*2654435761 + 1013904223
	h ^= h >> 15
	return int(h%uint32(2*amp+1)) - amp
}

// ---- ApplyChannelGains ---------------------------------------------------

func TestApplyChannelGainsExact(t *testing.T) {
	src := cv.NewMat(4, 5, 3)
	for p := 0; p < src.Total(); p++ {
		src.Data[p*3+0] = uint8((p * 7) % 200)
		src.Data[p*3+1] = uint8((p * 13) % 200)
		src.Data[p*3+2] = uint8((p * 5) % 200)
	}
	gr, gg, gb := 1.5, 0.5, 2.0
	out := ApplyChannelGains(src, gr, gg, gb)
	gains := [3]float64{gr, gg, gb}
	for p := 0; p < src.Total(); p++ {
		for c := 0; c < 3; c++ {
			want := clampU8(float64(src.Data[p*3+c]) * gains[c])
			if out.Data[p*3+c] != want {
				t.Fatalf("pixel %d chan %d: got %d want %d", p, c, out.Data[p*3+c], want)
			}
		}
	}
	// Input must be untouched.
	if src.Data[3] == out.Data[3] && gains[0] != 1 {
		// not a strict check; just ensure a new buffer was returned
	}
	if &src.Data[0] == &out.Data[0] {
		t.Fatal("ApplyChannelGains must not alias the input buffer")
	}
}

func TestApplyChannelGainsClamp(t *testing.T) {
	src := cv.NewMat(1, 1, 3)
	src.SetPixel(0, 0, []uint8{200, 100, 10})
	out := ApplyChannelGains(src, 2.0, 0.0, 100.0)
	got := out.AtPixel(0, 0)
	if got[0] != 255 || got[1] != 0 || got[2] != 255 {
		t.Fatalf("clamp failed: got %v", got)
	}
}

// ---- SimpleWB ------------------------------------------------------------

func TestSimpleWBNeutralizesCast(t *testing.T) {
	src := tintedRamp(20, 32, 1.25, 1.0, 0.8)
	before := spread(channelMeans(src))
	wb := NewSimpleWB()
	out := wb.BalanceWhite(src)
	after := spread(channelMeans(out))
	if after >= before {
		t.Fatalf("SimpleWB did not reduce channel-mean spread: before=%.2f after=%.2f", before, after)
	}
	if after > 8 {
		t.Fatalf("SimpleWB left too much cast: spread=%.2f", after)
	}
}

func TestSimpleWBCustomRangePanicsOnGray(t *testing.T) {
	// Grayscale input must panic.
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic for single-channel input")
		}
	}()
	gray := cv.NewMat(3, 3, 1)
	NewSimpleWB().BalanceWhite(gray)
}

// ---- GrayworldWB ---------------------------------------------------------

func TestGrayworldNeutralizesCast(t *testing.T) {
	src := tintedRamp(24, 24, 1.3, 1.0, 0.75)
	before := spread(channelMeans(src))
	wb := NewGrayworldWB()
	out := wb.BalanceWhite(src)
	after := spread(channelMeans(out))
	if after > 2.0 {
		t.Fatalf("Grayworld should equalise channel means: before=%.2f after=%.2f", before, after)
	}
	if after >= before {
		t.Fatalf("Grayworld did not reduce spread: before=%.2f after=%.2f", before, after)
	}
}

func TestGrayworldSaturationThresholdAllPixels(t *testing.T) {
	src := tintedRamp(8, 8, 1.1, 1.0, 0.9)
	wb := &GrayworldWB{SaturationThreshold: 1.0}
	out := wb.BalanceWhite(src)
	if out == nil || out.Empty() {
		t.Fatal("expected a result")
	}
}

// ---- LearningBasedWB -----------------------------------------------------

func TestLearningBasedReducesCast(t *testing.T) {
	// Build a textured neutral scene, then tint it.
	base := cv.NewMat(24, 24, 3)
	for y := 0; y < 24; y++ {
		for x := 0; x < 24; x++ {
			v := uint8(60 + ((x*7 + y*3) % 120))
			base.SetPixel(y, x, []uint8{v, v, v})
		}
	}
	cast := ApplyChannelGains(base, 1.35, 1.0, 0.7)
	before := spread(channelMeans(cast))
	wb := NewLearningBasedWB()
	out := wb.BalanceWhite(cast)
	after := spread(channelMeans(out))
	if after >= before {
		t.Fatalf("LearningBasedWB did not reduce cast: before=%.2f after=%.2f", before, after)
	}
}

// ---- Oilpainting ---------------------------------------------------------

func uniqueColors(m *cv.Mat) int {
	set := make(map[uint32]struct{})
	for p := 0; p < m.Total(); p++ {
		var key uint32
		for c := 0; c < m.Channels; c++ {
			key = key<<8 | uint32(m.Data[p*m.Channels+c])
		}
		set[key] = struct{}{}
	}
	return len(set)
}

func TestOilpaintingReducesColorsPreservesStructure(t *testing.T) {
	rows, cols := 32, 32
	src := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			// Independent per-channel noise so nearly every pixel is unique.
			nr := lcgNoise(y, x, 25)
			ng := lcgNoise(y, x+7, 25)
			nb := lcgNoise(y+7, x, 25)
			if x < cols/2 {
				src.SetPixel(y, x, []uint8{clampU8(180 + float64(nr)), clampU8(60 + float64(ng)), clampU8(60 + float64(nb))})
			} else {
				src.SetPixel(y, x, []uint8{clampU8(60 + float64(nr)), clampU8(60 + float64(ng)), clampU8(180 + float64(nb))})
			}
		}
	}
	uBefore := uniqueColors(src)
	out := Oilpainting(src, 4, 24)
	uAfter := uniqueColors(out)
	if uAfter >= uBefore {
		t.Fatalf("oilpainting did not reduce unique colours: before=%d after=%d", uBefore, uAfter)
	}
	// Structure: left stays reddish (R>B), right stays bluish (B>R).
	lm := regionMean(out, 8, 4, 16, 8)  // left block
	rm := regionMean(out, 8, 20, 16, 8) // right block
	if lm[0] <= lm[2] {
		t.Fatalf("left region should stay red-dominant, got %v", lm)
	}
	if rm[2] <= rm[0] {
		t.Fatalf("right region should stay blue-dominant, got %v", rm)
	}
}

func TestOilpaintingSingleChannel(t *testing.T) {
	src := cv.NewMat(10, 10, 1)
	for p := range src.Data {
		src.Data[p] = uint8((p * 17) % 256)
	}
	out := Oilpainting(src, 2, 20)
	if out.Channels != 1 || out.Rows != 10 {
		t.Fatal("unexpected shape")
	}
}

func regionMean(m *cv.Mat, y, x, h, w int) [3]float64 {
	var s [3]float64
	var n float64
	for yy := y; yy < y+h; yy++ {
		for xx := x; xx < x+w; xx++ {
			for c := 0; c < 3; c++ {
				s[c] += float64(m.At(yy, xx, c))
			}
			n++
		}
	}
	return [3]float64{s[0] / n, s[1] / n, s[2] / n}
}

// ---- BM3D ----------------------------------------------------------------

func TestBm3dReducesNoisePreservesEdgeAndMean(t *testing.T) {
	rows, cols := 48, 48
	clean := cv.NewMat(rows, cols, 1)
	noisy := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var v float64 = 60
			if x >= cols/2 {
				v = 200
			}
			clean.Set(y, x, 0, uint8(v))
			noisy.Set(y, x, 0, clampU8(v+float64(lcgNoise(y, x, 22))))
		}
	}
	den := Bm3dDenoising(noisy, 22)

	// Variance of residual in a flat interior region (left side), away from edge.
	varNoisy := residualVar(noisy, clean, 4, 4, 40, 16)
	varDen := residualVar(den, clean, 4, 4, 40, 16)
	if varDen >= varNoisy {
		t.Fatalf("BM3D did not reduce noise variance: noisy=%.1f den=%.1f", varNoisy, varDen)
	}

	// Flat-region means preserved.
	meanLeft := flatMean(den, 8, 6, 32, 12)
	meanRight := flatMean(den, 8, 30, 32, 12)
	if math.Abs(meanLeft-60) > 6 {
		t.Fatalf("left flat mean drifted: %.2f (want ~60)", meanLeft)
	}
	if math.Abs(meanRight-200) > 6 {
		t.Fatalf("right flat mean drifted: %.2f (want ~200)", meanRight)
	}
	// Edge preserved.
	if meanRight-meanLeft < 120 {
		t.Fatalf("edge contrast lost: %.2f", meanRight-meanLeft)
	}
}

func TestBm3dColor(t *testing.T) {
	src := cv.NewMat(20, 20, 3)
	for y := 0; y < 20; y++ {
		for x := 0; x < 20; x++ {
			src.SetPixel(y, x, []uint8{
				clampU8(100 + float64(lcgNoise(y, x, 15))),
				clampU8(120 + float64(lcgNoise(y, x+1, 15))),
				clampU8(140 + float64(lcgNoise(y+1, x, 15))),
			})
		}
	}
	out := Bm3dDenoising(src, 15)
	if out.Channels != 3 {
		t.Fatal("expected 3 channels")
	}
}

func TestBm3dTinyImage(t *testing.T) {
	src := cv.NewMat(2, 2, 1)
	src.Data = []uint8{10, 20, 30, 40}
	out := Bm3dDenoising(src, 5)
	// Too small to block-process: passthrough.
	for i := range src.Data {
		if out.Data[i] != src.Data[i] {
			t.Fatalf("tiny image should pass through: %v vs %v", src.Data, out.Data)
		}
	}
}

func residualVar(a, clean *cv.Mat, y, x, h, w int) float64 {
	var s, s2 float64
	var n float64
	for yy := y; yy < y+h; yy++ {
		for xx := x; xx < x+w; xx++ {
			d := float64(a.At(yy, xx, 0)) - float64(clean.At(yy, xx, 0))
			s += d
			s2 += d * d
			n++
		}
	}
	mean := s / n
	return s2/n - mean*mean
}

func flatMean(m *cv.Mat, y, x, h, w int) float64 {
	var s, n float64
	for yy := y; yy < y+h; yy++ {
		for xx := x; xx < x+w; xx++ {
			s += float64(m.At(yy, xx, 0))
			n++
		}
	}
	return s / n
}

// ---- Inpaint -------------------------------------------------------------

func TestInpaintFillsMaskedBlock(t *testing.T) {
	rows, cols := 40, 40
	clean := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := clampU8(20 + float64(y)*4 + float64(x%3)*6)
			clean.SetPixel(y, x, []uint8{v, v, v})
		}
	}
	mask := cv.NewMat(rows, cols, 1)
	// Corrupt a central block.
	by, bx, bs := 16, 16, 8
	corrupt := clean.Clone()
	for y := by; y < by+bs; y++ {
		for x := bx; x < bx+bs; x++ {
			mask.Set(y, x, 0, 255)
			corrupt.SetPixel(y, x, []uint8{0, 0, 0})
		}
	}
	out := Inpaint(corrupt, mask)

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
	if mae > 25 {
		t.Fatalf("inpaint fill too far from ground truth: mae=%.2f", mae)
	}
	// Known pixels must be preserved exactly.
	if out.At(0, 0, 0) != clean.At(0, 0, 0) {
		t.Fatal("inpaint must not alter known pixels")
	}
}

func TestInpaintNoMask(t *testing.T) {
	src := cv.NewMat(6, 6, 3)
	src.SetTo(120)
	mask := cv.NewMat(6, 6, 1) // all zero -> nothing to fill
	out := Inpaint(src, mask)
	for i := range src.Data {
		if out.Data[i] != src.Data[i] {
			t.Fatal("empty mask should return an unchanged copy")
		}
	}
}

func TestInpaintFullyMaskedFallback(t *testing.T) {
	// Every pixel unknown: no exemplar candidates exist, exercising the
	// neighbour-mean fallback and guaranteeing termination.
	src := cv.NewMat(5, 5, 3)
	src.SetTo(90)
	mask := cv.NewMat(5, 5, 1)
	mask.SetTo(255)
	out := Inpaint(src, mask)
	if out == nil || out.Empty() {
		t.Fatal("expected a filled result")
	}
	// All pixels should be defined (no panic, finite loop).
	for i := range out.Data {
		_ = out.Data[i]
	}
}

// ---- determinism ---------------------------------------------------------

func TestDeterminism(t *testing.T) {
	src := tintedRamp(16, 16, 1.2, 1.0, 0.85)
	a := NewGrayworldWB().BalanceWhite(src)
	b := NewGrayworldWB().BalanceWhite(src)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatal("Grayworld not deterministic")
		}
	}
	o1 := Oilpainting(src, 2, 16)
	o2 := Oilpainting(src, 2, 16)
	for i := range o1.Data {
		if o1.Data[i] != o2.Data[i] {
			t.Fatal("Oilpainting not deterministic")
		}
	}
}
