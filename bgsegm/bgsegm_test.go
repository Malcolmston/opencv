package bgsegm

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// solidFrame returns a rows×cols single-channel frame filled with val.
func solidFrame(rows, cols int, val uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	m.SetTo(val)
	return m
}

// withBlob returns a copy of base with a height×width rectangle at (y0,x0) set
// to val.
func withBlob(base *cv.Mat, y0, x0, height, width int, val uint8) *cv.Mat {
	m := base.Clone()
	for y := y0; y < y0+height; y++ {
		for x := x0; x < x0+width; x++ {
			m.Set(y, x, 0, val)
		}
	}
	return m
}

// countMask counts the pixels in mask equal to want.
func countMask(mask *cv.Mat, want uint8) int {
	n := 0
	for _, v := range mask.Data {
		if v == want {
			n++
		}
	}
	return n
}

// warmUp feeds n identical background frames to a subtractor.
func warmUp(sub BackgroundSubtractor, rows, cols int, bgVal uint8, n int) {
	for i := 0; i < n; i++ {
		sub.Apply(solidFrame(rows, cols, bgVal))
	}
}

const (
	rows, cols = 12, 16
	bgVal      = uint8(30)
	fgVal      = uint8(240)
	blobY      = 4
	blobX      = 5
	blobH      = 3
	blobW      = 4
	blobArea   = blobH * blobW
)

// foregroundBlobCase checks that after warm-up a subtractor flags exactly the
// bright blob and leaves the static background alone.
func foregroundBlobCase(t *testing.T, sub BackgroundSubtractor, warm int) {
	t.Helper()
	warmUp(sub, rows, cols, bgVal, warm)

	frame := withBlob(solidFrame(rows, cols, bgVal), blobY, blobX, blobH, blobW, fgVal)
	mask := sub.Apply(frame)

	fg := countMask(mask, ForegroundValue)
	if fg != blobArea {
		t.Errorf("foreground area = %d, want %d", fg, blobArea)
	}
	// The blob region itself must be fully foreground.
	for y := blobY; y < blobY+blobH; y++ {
		for x := blobX; x < blobX+blobW; x++ {
			if mask.At(y, x, 0) != ForegroundValue {
				t.Fatalf("blob pixel (%d,%d) not foreground: %d", y, x, mask.At(y, x, 0))
			}
		}
	}
	// The background must be (almost) all zero.
	if bgOnly := countMask(mask, BackgroundValue); bgOnly < rows*cols-blobArea {
		t.Errorf("background pixels = %d, want >= %d", bgOnly, rows*cols-blobArea)
	}
}

// backgroundConvergesCase checks GetBackgroundImage matches the static
// background after warm-up.
func backgroundConvergesCase(t *testing.T, sub BackgroundSubtractor, warm int, tol int) {
	t.Helper()
	warmUp(sub, rows, cols, bgVal, warm)
	bg := sub.GetBackgroundImage()
	if bg == nil {
		t.Fatal("GetBackgroundImage returned nil after warm-up")
	}
	for i, v := range bg.Data {
		if d := int(v) - int(bgVal); d < -tol || d > tol {
			t.Fatalf("background[%d] = %d, want %d±%d", i, v, bgVal, tol)
		}
	}
}

func TestMOG2ForegroundBlob(t *testing.T) {
	foregroundBlobCase(t, NewBackgroundSubtractorMOG2(10, 16, false), 30)
}

func TestMOG2BackgroundImage(t *testing.T) {
	backgroundConvergesCase(t, NewBackgroundSubtractorMOG2(10, 16, false), 30, 1)
}

func TestMOG2NilBeforeApply(t *testing.T) {
	sub := NewBackgroundSubtractorMOG2(10, 16, false)
	if sub.GetBackgroundImage() != nil {
		t.Error("GetBackgroundImage should be nil before first Apply")
	}
}

// TestMOG2Absorption verifies that a pixel that jumps to a new constant value is
// absorbed into the background within History frames.
func TestMOG2Absorption(t *testing.T) {
	history := 20
	sub := NewBackgroundSubtractorMOG2(history, 16, false)
	warmUp(sub, rows, cols, bgVal, 2*history)

	const py, px = 6, 8
	newVal := uint8(200)
	var last *cv.Mat
	for i := 0; i < history; i++ {
		last = sub.Apply(withBlob(solidFrame(rows, cols, bgVal), py, px, 1, 1, newVal))
	}
	if got := last.At(py, px, 0); got != BackgroundValue {
		t.Errorf("changed pixel not absorbed after %d frames: mask = %d", history, got)
	}
	// The background estimate at that pixel should have moved toward the new value.
	bg := sub.GetBackgroundImage()
	if got := bg.At(py, px, 0); int(got) < 150 {
		t.Errorf("background at changed pixel = %d, want >= 150", got)
	}
}

func TestMOG2ShadowDetection(t *testing.T) {
	sub := NewBackgroundSubtractorMOG2(10, 16, true)
	bright := uint8(200)
	warmUp(sub, rows, cols, bright, 30)
	// A pixel darkened to ~60% of the background is a shadow, not foreground.
	frame := withBlob(solidFrame(rows, cols, bright), blobY, blobX, blobH, blobW, uint8(120))
	mask := sub.Apply(frame)
	if s := countMask(mask, ShadowValue); s == 0 {
		t.Error("expected some shadow-classified pixels")
	}
	if countMask(mask, ForegroundValue) != 0 {
		t.Error("darkened region should be shadow, not foreground")
	}
}

func TestKNNForegroundBlob(t *testing.T) {
	foregroundBlobCase(t, NewBackgroundSubtractorKNN(20, 400, false), 15)
}

func TestKNNBackgroundImage(t *testing.T) {
	backgroundConvergesCase(t, NewBackgroundSubtractorKNN(20, 400, false), 15, 0)
}

func TestKNNAbsorption(t *testing.T) {
	history := 20
	sub := NewBackgroundSubtractorKNN(history, 400, false)
	warmUp(sub, rows, cols, bgVal, 10)

	const py, px = 6, 8
	newVal := uint8(200)
	var last *cv.Mat
	for i := 0; i < history; i++ {
		last = sub.Apply(withBlob(solidFrame(rows, cols, bgVal), py, px, 1, 1, newVal))
	}
	if got := last.At(py, px, 0); got != BackgroundValue {
		t.Errorf("changed pixel not absorbed after %d frames: mask = %d", history, got)
	}
}

func TestRunningAverageForegroundBlob(t *testing.T) {
	foregroundBlobCase(t, NewRunningAverage(10, 40), 10)
}

func TestRunningAverageBackgroundImage(t *testing.T) {
	backgroundConvergesCase(t, NewRunningAverage(10, 40), 10, 0)
}

func TestRunningAverageNilBeforeApply(t *testing.T) {
	if NewRunningAverage(10, 40).GetBackgroundImage() != nil {
		t.Error("GetBackgroundImage should be nil before first Apply")
	}
}

// TestRunningAverageAbsorption verifies a pixel settling at a new value is
// absorbed into the exponential average.
func TestRunningAverageAbsorption(t *testing.T) {
	history := 10
	sub := NewRunningAverage(history, 40)
	warmUp(sub, rows, cols, bgVal, 5)

	const py, px = 6, 8
	newVal := uint8(200)
	var last *cv.Mat
	// Enough frames for the exponential average to close the |200-30| gap below
	// the threshold.
	for i := 0; i < 6*history; i++ {
		last = sub.Apply(withBlob(solidFrame(rows, cols, bgVal), py, px, 1, 1, newVal))
	}
	if got := last.At(py, px, 0); got != BackgroundValue {
		t.Errorf("changed pixel not absorbed: mask = %d", got)
	}
}

func TestGMGForegroundBlob(t *testing.T) {
	// GMG only classifies after NumInitFrames; warm past it.
	foregroundBlobCase(t, NewBackgroundSubtractorGMG(20, 0.8), 25)
}

func TestGMGBackgroundImage(t *testing.T) {
	// Bin centres are quantised, so allow a bin-width tolerance.
	backgroundConvergesCase(t, NewBackgroundSubtractorGMG(20, 0.8), 25, 8)
}

func TestGMGLearningPeriodIsBackground(t *testing.T) {
	sub := NewBackgroundSubtractorGMG(20, 0.8)
	// Even a wildly different frame during the learning period is background.
	mask := sub.Apply(solidFrame(rows, cols, 200))
	if countMask(mask, BackgroundValue) != rows*cols {
		t.Error("first frame during init period should be all background")
	}
}

func TestCleanupMaskRemovesSpecks(t *testing.T) {
	mask := cv.NewMat(rows, cols, 1)
	// A large blob that survives opening.
	for y := 2; y < 8; y++ {
		for x := 2; x < 8; x++ {
			mask.Set(y, x, 0, ForegroundValue)
		}
	}
	// An isolated single-pixel speck that opening should erase.
	mask.Set(10, 14, 0, ForegroundValue)

	out := CleanupMask(mask, 3)
	if out.At(10, 14, 0) != 0 {
		t.Error("isolated speck should be removed by opening")
	}
	if out.At(5, 5, 0) != ForegroundValue {
		t.Error("interior of large blob should survive opening")
	}
}

func TestCleanupMaskNoOp(t *testing.T) {
	mask := cv.NewMat(4, 4, 1)
	mask.Set(1, 1, 0, ForegroundValue)
	if out := CleanupMask(mask, 1); out != mask {
		t.Error("ksize <= 1 should return the mask unchanged")
	}
}

func TestOpenKernelIntegration(t *testing.T) {
	sub := NewBackgroundSubtractorMOG2(10, 16, false)
	sub.OpenKernel = 3
	warmUp(sub, rows, cols, bgVal, 30)
	// A single-pixel bright speck should be opened away.
	frame := withBlob(solidFrame(rows, cols, bgVal), 6, 8, 1, 1, fgVal)
	mask := sub.Apply(frame)
	if countMask(mask, ForegroundValue) != 0 {
		t.Error("single-pixel foreground should be removed by OpenKernel cleanup")
	}
}

func TestRGBFrameSupported(t *testing.T) {
	sub := NewRunningAverage(10, 40)
	// Warm up with a solid RGB background.
	for i := 0; i < 10; i++ {
		f := cv.NewMat(rows, cols, 3)
		f.SetTo(bgVal)
		sub.Apply(f)
	}
	frame := cv.NewMat(rows, cols, 3)
	frame.SetTo(bgVal)
	for y := blobY; y < blobY+blobH; y++ {
		for x := blobX; x < blobX+blobW; x++ {
			frame.SetPixel(y, x, []uint8{fgVal, fgVal, fgVal})
		}
	}
	mask := sub.Apply(frame)
	if mask.Channels != 1 {
		t.Fatalf("mask channels = %d, want 1", mask.Channels)
	}
	if got := countMask(mask, ForegroundValue); got != blobArea {
		t.Errorf("RGB foreground area = %d, want %d", got, blobArea)
	}
}

func TestFrameSizeMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic on mismatched frame size")
		}
	}()
	sub := NewBackgroundSubtractorMOG2(10, 16, false)
	sub.Apply(solidFrame(8, 8, bgVal))
	sub.Apply(solidFrame(9, 9, bgVal))
}
