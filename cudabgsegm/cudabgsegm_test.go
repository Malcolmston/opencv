package cudabgsegm_test

import (
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/bgsegm"
	"github.com/malcolmston/opencv/cudabgsegm"
)

// solid returns a rows×cols single-channel Mat filled with val.
func solid(rows, cols int, val uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	m.SetTo(val)
	return m
}

// noisyBackground returns a static background of base intensity with a small
// deterministic amount of per-pixel noise drawn from rng, so the model has some
// variance to learn instead of a perfectly flat frame.
func noisyBackground(rows, cols int, base uint8, rng *rand.Rand) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for i := range m.Data {
		v := int(base) + rng.Intn(5) - 2
		if v < 0 {
			v = 0
		}
		if v > 255 {
			v = 255
		}
		m.Data[i] = uint8(v)
	}
	return m
}

// addBlob paints a filled rectangle [y0,y1)×[x0,x1) of value val onto a copy of
// frame and returns the copy.
func addBlob(frame *cv.Mat, y0, y1, x0, x1 int, val uint8) *cv.Mat {
	out := frame.Clone()
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			out.Set(y, x, 0, val)
		}
	}
	return out
}

// countValue counts mask samples equal to v.
func countValue(mask *cv.Mat, v uint8) int {
	n := 0
	for _, s := range mask.Data {
		if s == v {
			n++
		}
	}
	return n
}

// blobFlagged reports whether every pixel in the rectangle [y0,y1)×[x0,x1) is
// flagged as foreground in mask.
func blobFlagged(mask *cv.Mat, y0, y1, x0, x1 int) bool {
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			if mask.At(y, x, 0) != bgsegm.ForegroundValue {
				return false
			}
		}
	}
	return true
}

const (
	rows, cols = 16, 16
	bgVal      = uint8(40)
	fgVal      = uint8(230)
	by0, by1   = 5, 9
	bx0, bx1   = 6, 10
)

// warm feeds n static (mildly noisy) background frames through a subtractor via
// its GpuMat Apply, using a fixed seed for determinism.
func warm(apply func(*cudabgsegm.GpuMat), n int, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	for i := 0; i < n; i++ {
		frame := cudabgsegm.GpuMatFromMat(noisyBackground(rows, cols, bgVal, rng))
		apply(frame)
	}
}

func TestMOGDetectsMovingBlob(t *testing.T) {
	sub := cudabgsegm.CreateBackgroundSubtractorMOG(20, 5, 0.7, 15)
	stream := cudabgsegm.NewStream()

	warm(func(g *cudabgsegm.GpuMat) { sub.Apply(g, -1, stream) }, 40, 1)

	rng := rand.New(rand.NewSource(99))
	frame := addBlob(noisyBackground(rows, cols, bgVal, rng), by0, by1, bx0, bx1, fgVal)
	mask := sub.Apply(cudabgsegm.GpuMatFromMat(frame), -1, stream).Download(stream)

	if !blobFlagged(mask, by0, by1, bx0, bx1) {
		t.Fatalf("MOG failed to flag the moving blob as foreground")
	}
	// The static background should stay overwhelmingly background.
	bgCount := countValue(mask, bgsegm.BackgroundValue)
	if bgCount < rows*cols-2*((by1-by0)*(bx1-bx0)) {
		t.Fatalf("MOG flagged too much background as foreground: bg=%d", bgCount)
	}
}

func TestMOG2DetectsMovingBlob(t *testing.T) {
	sub := cudabgsegm.CreateBackgroundSubtractorMOG2(20, 16, false)
	stream := cudabgsegm.NewStream()

	warm(func(g *cudabgsegm.GpuMat) { sub.Apply(g, -1, stream) }, 40, 2)

	rng := rand.New(rand.NewSource(7))
	frame := addBlob(noisyBackground(rows, cols, bgVal, rng), by0, by1, bx0, bx1, fgVal)
	mask := sub.Apply(cudabgsegm.GpuMatFromMat(frame), -1, stream).Download(stream)

	if !blobFlagged(mask, by0, by1, bx0, bx1) {
		t.Fatalf("MOG2 failed to flag the moving blob as foreground")
	}
	if fg := countValue(mask, bgsegm.ForegroundValue); fg < (by1-by0)*(bx1-bx0) {
		t.Fatalf("MOG2 foreground count too low: %d", fg)
	}
}

func TestMOG2ShadowDetection(t *testing.T) {
	sub := cudabgsegm.CreateBackgroundSubtractorMOG2(20, 16, true)
	if !sub.GetDetectShadows() {
		t.Fatalf("expected detectShadows true")
	}
	stream := cudabgsegm.NewStream()

	// Warm on a flat bright background so a darkened region reads as a shadow.
	for i := 0; i < 40; i++ {
		sub.Apply(cudabgsegm.GpuMatFromMat(solid(rows, cols, 200)), -1, stream)
	}
	// A region at ~60% brightness: darker than background but above the 0.5
	// shadow floor -> should be classified as shadow, not foreground.
	frame := addBlob(solid(rows, cols, 200), by0, by1, bx0, bx1, 120)
	mask := sub.Apply(cudabgsegm.GpuMatFromMat(frame), -1, stream).Download(stream)

	if shadows := countValue(mask, bgsegm.ShadowValue); shadows == 0 {
		t.Fatalf("expected some shadow-classified pixels, got none")
	}
}

func TestGMGDetectsMovingBlob(t *testing.T) {
	sub := cudabgsegm.CreateBackgroundSubtractorGMG(15, 0.8)
	stream := cudabgsegm.NewStream()

	// During the init period the mask must be all background.
	initMask := sub.Apply(cudabgsegm.GpuMatFromMat(solid(rows, cols, bgVal)), -1, stream).Download(stream)
	if fg := countValue(initMask, bgsegm.ForegroundValue); fg != 0 {
		t.Fatalf("GMG produced foreground during init period: %d", fg)
	}

	for i := 0; i < 30; i++ {
		sub.Apply(cudabgsegm.GpuMatFromMat(solid(rows, cols, bgVal)), -1, stream)
	}
	frame := addBlob(solid(rows, cols, bgVal), by0, by1, bx0, bx1, fgVal)
	mask := sub.Apply(cudabgsegm.GpuMatFromMat(frame), -1, stream).Download(stream)
	if !blobFlagged(mask, by0, by1, bx0, bx1) {
		t.Fatalf("GMG failed to flag the moving blob as foreground")
	}
}

func TestBackgroundImage(t *testing.T) {
	stream := cudabgsegm.NewStream()

	mog := cudabgsegm.CreateBackgroundSubtractorMOG(10, 5, 0, 0)
	if !mog.GetBackgroundImage(stream).Empty() {
		t.Fatalf("MOG background image should be empty before Apply")
	}
	for i := 0; i < 20; i++ {
		mog.Apply(cudabgsegm.GpuMatFromMat(solid(8, 8, 100)), -1, stream)
	}
	bg := mog.GetBackgroundImage(stream)
	if bg.Empty() {
		t.Fatalf("MOG background image empty after Apply")
	}
	if got := bg.Mat.At(0, 0, 0); got < 90 || got > 110 {
		t.Fatalf("MOG background value %d not near 100", got)
	}

	mog2 := cudabgsegm.CreateBackgroundSubtractorMOG2(10, 16, false)
	if !mog2.GetBackgroundImage(stream).Empty() {
		t.Fatalf("MOG2 background image should be empty before Apply")
	}
	for i := 0; i < 20; i++ {
		mog2.Apply(cudabgsegm.GpuMatFromMat(solid(8, 8, 100)), -1, stream)
	}
	if mog2.GetBackgroundImage(stream).Empty() {
		t.Fatalf("MOG2 background image empty after Apply")
	}

	gmg := cudabgsegm.CreateBackgroundSubtractorGMG(5, 0.8)
	if !gmg.GetBackgroundImage(stream).Empty() {
		t.Fatalf("GMG background image should be empty before Apply")
	}
	gmg.Apply(cudabgsegm.GpuMatFromMat(solid(8, 8, 100)), -1, stream)
	if gmg.GetBackgroundImage(stream).Empty() {
		t.Fatalf("GMG background image empty after Apply")
	}
}

func TestMOGAccessors(t *testing.T) {
	sub := cudabgsegm.CreateBackgroundSubtractorMOG(0, 0, 0, 0)
	if sub.GetHistory() != 200 {
		t.Fatalf("default history = %d, want 200", sub.GetHistory())
	}
	if sub.GetNMixtures() != 5 {
		t.Fatalf("default nmixtures = %d, want 5", sub.GetNMixtures())
	}
	sub.SetHistory(123)
	sub.SetNMixtures(7)
	sub.SetBackgroundRatio(0.6)
	sub.SetNoiseSigma(9)
	if sub.GetHistory() != 123 || sub.GetNMixtures() != 7 {
		t.Fatalf("history/nmixtures setters failed: %d %d", sub.GetHistory(), sub.GetNMixtures())
	}
	if sub.GetBackgroundRatio() != 0.6 || sub.GetNoiseSigma() != 9 {
		t.Fatalf("ratio/sigma setters failed: %v %v", sub.GetBackgroundRatio(), sub.GetNoiseSigma())
	}
}

func TestMOG2Accessors(t *testing.T) {
	sub := cudabgsegm.CreateBackgroundSubtractorMOG2(0, 0, false)
	if sub.GetHistory() != 500 {
		t.Fatalf("default history = %d, want 500", sub.GetHistory())
	}
	if sub.GetVarThreshold() != 16 {
		t.Fatalf("default varThreshold = %v, want 16", sub.GetVarThreshold())
	}
	if sub.GetDetectShadows() {
		t.Fatalf("default detectShadows should be false")
	}
	sub.SetHistory(250)
	sub.SetVarThreshold(25)
	sub.SetDetectShadows(true)
	if sub.GetHistory() != 250 || sub.GetVarThreshold() != 25 || !sub.GetDetectShadows() {
		t.Fatalf("MOG2 setters failed: %d %v %v", sub.GetHistory(), sub.GetVarThreshold(), sub.GetDetectShadows())
	}
}

func TestGMGAccessors(t *testing.T) {
	sub := cudabgsegm.CreateBackgroundSubtractorGMG(0, 0)
	if sub.GetNumFrames() != 20 {
		t.Fatalf("default numFrames = %d, want 20", sub.GetNumFrames())
	}
	if sub.GetDecisionThreshold() != 0.8 {
		t.Fatalf("default decisionThreshold = %v, want 0.8", sub.GetDecisionThreshold())
	}
	sub.SetNumFrames(11)
	sub.SetDecisionThreshold(0.6)
	if sub.GetNumFrames() != 11 || sub.GetDecisionThreshold() != 0.6 {
		t.Fatalf("GMG setters failed: %d %v", sub.GetNumFrames(), sub.GetDecisionThreshold())
	}
}

func TestGpuMat(t *testing.T) {
	stream := cudabgsegm.NewStream()

	empty := cudabgsegm.NewGpuMat()
	if !empty.Empty() {
		t.Fatalf("NewGpuMat should be empty")
	}
	if r, c := empty.Size(); r != 0 || c != 0 {
		t.Fatalf("empty size = %d,%d want 0,0", r, c)
	}
	if empty.Channels() != 0 {
		t.Fatalf("empty channels = %d want 0", empty.Channels())
	}
	if empty.Download(stream) != nil {
		t.Fatalf("empty download should be nil")
	}
	if !empty.Clone().Empty() {
		t.Fatalf("clone of empty should be empty")
	}

	// Upload replaces contents with a deep copy.
	src := solid(4, 5, 77)
	g := cudabgsegm.NewGpuMat()
	g.Upload(src, stream)
	if r, c := g.Size(); r != 4 || c != 5 {
		t.Fatalf("uploaded size = %d,%d want 4,5", r, c)
	}
	if g.Channels() != 1 {
		t.Fatalf("uploaded channels = %d want 1", g.Channels())
	}
	// Mutating the source must not affect the uploaded copy.
	src.Set(0, 0, 0, 0)
	if got := g.Download(stream).At(0, 0, 0); got != 77 {
		t.Fatalf("upload did not deep-copy: got %d want 77", got)
	}

	// Clone is independent.
	clone := g.Clone()
	g.Mat.Set(1, 1, 0, 5)
	if clone.Mat.At(1, 1, 0) != 77 {
		t.Fatalf("clone shared storage with original")
	}

	// Upload of nil empties the GpuMat.
	g.Upload(nil, stream)
	if !g.Empty() {
		t.Fatalf("upload(nil) should empty the GpuMat")
	}

	// FromMat wraps without copying.
	m := solid(2, 2, 9)
	wrapped := cudabgsegm.GpuMatFromMat(m)
	wrapped.Mat.Set(0, 0, 0, 1)
	if m.At(0, 0, 0) != 1 {
		t.Fatalf("GpuMatFromMat should share storage")
	}
}

func TestStreamNoOp(t *testing.T) {
	s := cudabgsegm.NewStream()
	s.WaitForCompletion() // must not block or panic
	if !s.QueryIfComplete() {
		t.Fatalf("no-op stream should always report complete")
	}
	// A nil stream must be accepted everywhere.
	sub := cudabgsegm.CreateBackgroundSubtractorMOG2(5, 16, false)
	mask := sub.Apply(cudabgsegm.GpuMatFromMat(solid(4, 4, 30)), -1, nil)
	if mask.Empty() {
		t.Fatalf("Apply with nil stream returned empty mask")
	}
}

func TestApplyPanicsOnEmptyFrame(t *testing.T) {
	sub := cudabgsegm.CreateBackgroundSubtractorMOG2(5, 16, false)
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on nil frame")
		}
	}()
	sub.Apply(nil, -1, nil)
}

func TestApplyPanicsOnEmptyGpuMat(t *testing.T) {
	sub := cudabgsegm.CreateBackgroundSubtractorMOG(5, 5, 0, 0)
	defer func() {
		if recover() == nil {
			t.Fatalf("expected panic on empty GpuMat frame")
		}
	}()
	sub.Apply(cudabgsegm.NewGpuMat(), -1, nil)
}
