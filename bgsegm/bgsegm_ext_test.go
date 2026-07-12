package bgsegm

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- BackgroundSubtractorMOG ---------------------------------------------------

func TestMOGForegroundBlob(t *testing.T) {
	foregroundBlobCase(t, NewBackgroundSubtractorMOG(10, 5, false), 30)
}

func TestMOGBackgroundImage(t *testing.T) {
	backgroundConvergesCase(t, NewBackgroundSubtractorMOG(10, 5, false), 30, 1)
}

func TestMOGNilBeforeApply(t *testing.T) {
	if NewBackgroundSubtractorMOG(10, 5, false).GetBackgroundImage() != nil {
		t.Error("GetBackgroundImage should be nil before first Apply")
	}
}

func TestMOGShadowDetection(t *testing.T) {
	sub := NewBackgroundSubtractorMOG(10, 5, true)
	bright := uint8(200)
	warmUp(sub, rows, cols, bright, 30)
	frame := withBlob(solidFrame(rows, cols, bright), blobY, blobX, blobH, blobW, uint8(120))
	mask := sub.Apply(frame)
	if countMask(mask, ShadowValue) == 0 {
		t.Error("expected some shadow-classified pixels")
	}
	if countMask(mask, ForegroundValue) != 0 {
		t.Error("darkened region should be shadow, not foreground")
	}
}

// --- BackgroundSubtractorCNT ---------------------------------------------------

func TestCNTForegroundBlob(t *testing.T) {
	foregroundBlobCase(t, NewBackgroundSubtractorCNT(5, true, 900), 8)
}

func TestCNTBackgroundImage(t *testing.T) {
	backgroundConvergesCase(t, NewBackgroundSubtractorCNT(5, true, 900), 8, 0)
}

func TestCNTNilBeforeApply(t *testing.T) {
	if NewBackgroundSubtractorCNT(5, true, 900).GetBackgroundImage() != nil {
		t.Error("GetBackgroundImage should be nil before first Apply")
	}
}

// TestCNTAbsorption verifies a pixel that jumps to a new constant value is
// absorbed into the background within MinPixelStability frames.
func TestCNTAbsorption(t *testing.T) {
	min := 5
	sub := NewBackgroundSubtractorCNT(min, true, 900)
	warmUp(sub, rows, cols, bgVal, 10)

	const py, px = 6, 8
	newVal := uint8(200)
	var last *cv.Mat
	for i := 0; i < min+2; i++ {
		last = sub.Apply(withBlob(solidFrame(rows, cols, bgVal), py, px, 1, 1, newVal))
	}
	if got := last.At(py, px, 0); got != BackgroundValue {
		t.Errorf("changed pixel not absorbed after %d frames: mask = %d", min+2, got)
	}
}

// --- BackgroundSubtractorLSBP --------------------------------------------------

func TestLSBPForegroundBlob(t *testing.T) {
	foregroundBlobCase(t, NewBackgroundSubtractorLSBP(8, 30, false), 10)
}

func TestLSBPBackgroundImage(t *testing.T) {
	backgroundConvergesCase(t, NewBackgroundSubtractorLSBP(8, 30, false), 10, 0)
}

func TestLSBPNilBeforeApply(t *testing.T) {
	if NewBackgroundSubtractorLSBP(8, 30, false).GetBackgroundImage() != nil {
		t.Error("GetBackgroundImage should be nil before first Apply")
	}
}

// TestLSBPDeterministic confirms two identically seeded LSBP subtractors produce
// byte-identical masks over the same sequence.
func TestLSBPDeterministic(t *testing.T) {
	a := NewBackgroundSubtractorLSBP(8, 30, false)
	b := NewBackgroundSubtractorLSBP(8, 30, false)
	for i := 0; i < 12; i++ {
		f := withBlob(solidFrame(rows, cols, bgVal), (i%3)+2, (i%4)+3, blobH, blobW, fgVal)
		ma := a.Apply(f)
		mb := b.Apply(f)
		for p := range ma.Data {
			if ma.Data[p] != mb.Data[p] {
				t.Fatalf("frame %d pixel %d differs: %d vs %d", i, p, ma.Data[p], mb.Data[p])
			}
		}
	}
}

// TestLSBPDescriptorFlatIsUniform checks that a flat patch yields a zero local
// SVD feature and an all-ones descriptor, while structure changes the feature.
func TestLSBPDescriptorFlatIsUniform(t *testing.T) {
	flat := make([]float64, rows*cols)
	for i := range flat {
		flat[i] = 30
	}
	g := localSVDFeature(flat, rows, cols)
	for i, v := range g {
		if v > 1e-6 {
			t.Fatalf("flat SVD feature[%d] = %v, want 0", i, v)
		}
	}
	desc := lsbpDescriptors(g, rows, cols, 1.0)
	for i, d := range desc {
		if d != 0xFF {
			t.Fatalf("flat descriptor[%d] = %08b, want all ones", i, d)
		}
	}
	// A structured patch produces a non-zero feature somewhere.
	flat[5*cols+5] = 200
	g2 := localSVDFeature(flat, rows, cols)
	nonzero := false
	for _, v := range g2 {
		if v > 1e-6 {
			nonzero = true
			break
		}
	}
	if !nonzero {
		t.Error("structured patch should produce a non-zero SVD feature")
	}
}

// --- BackgroundSubtractorGSOC --------------------------------------------------

func TestGSOCForegroundBlob(t *testing.T) {
	foregroundBlobCase(t, NewBackgroundSubtractorGSOC(20, 30, false), 10)
}

func TestGSOCBackgroundImage(t *testing.T) {
	backgroundConvergesCase(t, NewBackgroundSubtractorGSOC(20, 30, false), 10, 0)
}

func TestGSOCNilBeforeApply(t *testing.T) {
	if NewBackgroundSubtractorGSOC(20, 30, false).GetBackgroundImage() != nil {
		t.Error("GetBackgroundImage should be nil before first Apply")
	}
}

// TestGSOCStaticAbsorption verifies that a stationary foreground object is
// absorbed into the background after StaticFrames frames.
func TestGSOCStaticAbsorption(t *testing.T) {
	sub := NewBackgroundSubtractorGSOC(20, 30, false)
	sub.StaticFrames = 5
	warmUp(sub, rows, cols, bgVal, 10)

	const py, px = 6, 8
	newVal := uint8(200)
	var last *cv.Mat
	for i := 0; i < sub.StaticFrames; i++ {
		last = sub.Apply(withBlob(solidFrame(rows, cols, bgVal), py, px, 1, 1, newVal))
	}
	if got := last.At(py, px, 0); got != BackgroundValue {
		t.Errorf("static object not absorbed after %d frames: mask = %d", sub.StaticFrames, got)
	}
}

// --- Morphological post-processing ---------------------------------------------

func TestCloseMaskFillsHole(t *testing.T) {
	mask := cv.NewMat(6, 6, 1)
	for y := 1; y < 5; y++ {
		for x := 1; x < 5; x++ {
			mask.Set(y, x, 0, ForegroundValue)
		}
	}
	mask.Set(3, 3, 0, 0) // a single-pixel hole
	out := CloseMask(mask, 3)
	if out.At(3, 3, 0) != ForegroundValue {
		t.Error("closing should fill an interior single-pixel hole")
	}
}

func TestCloseMaskNoOp(t *testing.T) {
	mask := cv.NewMat(4, 4, 1)
	mask.Set(1, 1, 0, ForegroundValue)
	if out := CloseMask(mask, 1); out != mask {
		t.Error("ksize <= 1 should return the mask unchanged")
	}
}

func TestRefineMask(t *testing.T) {
	mask := cv.NewMat(10, 10, 1)
	// A solid blob plus an isolated speck.
	for y := 2; y < 8; y++ {
		for x := 2; x < 8; x++ {
			mask.Set(y, x, 0, ForegroundValue)
		}
	}
	mask.Set(0, 9, 0, ForegroundValue) // speck to be opened away
	out := RefineMask(mask, 3, 3)
	if out.At(0, 9, 0) != 0 {
		t.Error("isolated speck should be removed by the opening stage")
	}
	if out.At(4, 4, 0) != ForegroundValue {
		t.Error("interior of a solid blob should survive refinement")
	}
}

// --- SyntheticSequenceGenerator ------------------------------------------------

func TestSyntheticSequenceGeneratorDeterministic(t *testing.T) {
	bg := solidFrame(20, 24, 40)
	obj := solidFrame(4, 4, 220)
	ga := NewSyntheticSequenceGenerator(bg, obj, 3, 20, 1, 1, 5, 7)
	gb := NewSyntheticSequenceGenerator(bg, obj, 3, 20, 1, 1, 5, 7)
	for i := 0; i < 6; i++ {
		fa, ma := ga.Next()
		fb, mb := gb.Next()
		for p := range fa.Data {
			if fa.Data[p] != fb.Data[p] {
				t.Fatalf("frame %d pixel %d differs between identically seeded generators", i, p)
			}
		}
		for p := range ma.Data {
			if ma.Data[p] != mb.Data[p] {
				t.Fatalf("gt mask %d pixel %d differs", i, p)
			}
		}
		if got := countMask(ma, ForegroundValue); got != obj.Total() {
			t.Fatalf("gt foreground area = %d, want %d", got, obj.Total())
		}
	}
}

// TestSyntheticSequenceGeneratorDetected feeds the synthetic sequence to a
// subtractor and checks that, after warm-up, every ground-truth object pixel is
// flagged foreground.
func TestSyntheticSequenceGeneratorDetected(t *testing.T) {
	bg := solidFrame(20, 24, 40)
	obj := solidFrame(3, 3, 220)
	gen := NewSyntheticSequenceGenerator(bg, obj, 3, 20, 1, 1, 0, 3)
	sub := NewRunningAverage(10, 60)

	var frame, gt *cv.Mat
	for i := 0; i < 20; i++ {
		frame, gt = gen.Next()
		sub.Apply(frame)
	}
	mask := sub.Apply(frame) // classify the final frame after warm-up
	for p := range gt.Data {
		if gt.Data[p] == ForegroundValue && mask.Data[p] != ForegroundValue {
			t.Fatalf("object pixel %d not detected: mask = %d", p, mask.Data[p])
		}
	}
}

// --- Getter/setter surface -----------------------------------------------------

func TestExistingSettersRoundTrip(t *testing.T) {
	m := NewBackgroundSubtractorMOG2(500, 16, false)
	m.SetHistory(123)
	m.SetVarThreshold(9)
	m.SetDetectShadows(true)
	m.SetShadowThreshold(0.4)
	m.SetNMixtures(4)
	m.SetBackgroundRatio(0.8)
	if m.GetHistory() != 123 || m.GetVarThreshold() != 9 || !m.GetDetectShadows() ||
		m.GetShadowThreshold() != 0.4 || m.GetNMixtures() != 4 || m.GetBackgroundRatio() != 0.8 {
		t.Error("MOG2 setters did not round-trip")
	}

	k := NewBackgroundSubtractorKNN(500, 400, false)
	k.SetHistory(77)
	k.SetDist2Threshold(250)
	k.SetKNNSamples(3)
	k.SetNSamples(9)
	if k.GetHistory() != 77 || k.GetDist2Threshold() != 250 || k.GetKNNSamples() != 3 || k.GetNSamples() != 9 {
		t.Error("KNN setters did not round-trip")
	}

	g := NewBackgroundSubtractorGMG(20, 0.8)
	g.SetNumInitFrames(30)
	g.SetDecisionThreshold(0.9)
	g.SetNumBins(32)
	g.SetLearningRate(0.05)
	if g.GetNumInitFrames() != 30 || g.GetDecisionThreshold() != 0.9 || g.GetNumBins() != 32 || g.GetLearningRate() != 0.05 {
		t.Error("GMG setters did not round-trip")
	}

	r := NewRunningAverage(10, 40)
	r.SetAlpha(0.2)
	r.SetThreshold(15)
	if r.GetAlpha() != 0.2 || r.GetThreshold() != 15 {
		t.Error("RunningAverage setters did not round-trip")
	}
}

// TestShadowParamsSetters checks that SetShadowValue changes the emitted shadow
// sample on a new subtractor.
func TestShadowParamsSetters(t *testing.T) {
	sub := NewBackgroundSubtractorMOG(10, 5, true)
	sub.SetShadowValue(100)
	sub.SetShadowThreshold(0.5)
	if sub.GetShadowValue() != 100 || sub.GetShadowThreshold() != 0.5 || !sub.GetDetectShadows() {
		t.Fatal("shadow setters did not round-trip")
	}
	bright := uint8(200)
	warmUp(sub, rows, cols, bright, 30)
	frame := withBlob(solidFrame(rows, cols, bright), blobY, blobX, blobH, blobW, uint8(120))
	mask := sub.Apply(frame)
	if countMask(mask, 100) == 0 {
		t.Error("expected pixels marked with the custom shadow value 100")
	}
	if countMask(mask, ShadowValue) != 0 {
		t.Error("no pixels should carry the default shadow value after override")
	}
}
