package ximgproc_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/ximgproc"
)

// solidColor returns a filled 3-channel image.
func solidColor(rows, cols int, r, g, b uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 3)
	for p := 0; p < rows*cols; p++ {
		m.Data[p*3+0] = r
		m.Data[p*3+1] = g
		m.Data[p*3+2] = b
	}
	return m
}

func TestGuidedFilterColorGuideAndSrc(t *testing.T) {
	// 3-channel src and 3-channel guide exercise the colour code paths.
	src := solidColor(16, 16, 100, 150, 200)
	src.Data[(8*16+8)*3+0] = 250 // one outlier
	guide := src.Clone()

	out := ximgproc.GuidedFilter(src, guide, 3, 500)
	if out.Channels != 3 || out.Rows != 16 || out.Cols != 16 {
		t.Fatalf("wrong output shape %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
	// A perfectly flat channel stays flat.
	if out.Data[(2*16+2)*3+1] != 150 {
		t.Errorf("flat green channel changed: got %d", out.Data[(2*16+2)*3+1])
	}
}

func TestGuidedFilterPanics(t *testing.T) {
	src := cv.NewMat(8, 8, 1)
	mustPanic(t, "negative radius", func() { ximgproc.GuidedFilter(src, src, 0, 1) })
	other := cv.NewMat(8, 9, 1)
	mustPanic(t, "size mismatch", func() { ximgproc.GuidedFilter(src, other, 2, 1) })
}

func TestThinningPanicsOnColor(t *testing.T) {
	mustPanic(t, "color thinning", func() { ximgproc.Thinning(cv.NewMat(4, 4, 3)) })
}

func TestNiBlackPanics(t *testing.T) {
	img := cv.NewMat(8, 8, 1)
	mustPanic(t, "color", func() { ximgproc.NiBlackThreshold(cv.NewMat(8, 8, 3), 0.2, 5, 0) })
	mustPanic(t, "even block", func() { ximgproc.NiBlackThreshold(img, 0.2, 4, 0) })
	mustPanic(t, "bad variant", func() { ximgproc.NiBlackThreshold(img, 0.2, 5, 99) })
}

func TestAnisotropicDiffusionPanics(t *testing.T) {
	img := cv.NewMat(6, 6, 1)
	mustPanic(t, "alpha", func() { ximgproc.AnisotropicDiffusion(img, 0, 10, 3) })
	mustPanic(t, "k", func() { ximgproc.AnisotropicDiffusion(img, 0.2, 0, 3) })
}

func TestAnisotropicDiffusionColor(t *testing.T) {
	img := solidColor(8, 8, 30, 60, 90)
	out := ximgproc.AnisotropicDiffusion(img, 0.2, 15, 3)
	if out.Channels != 3 {
		t.Fatal("expected 3 channels")
	}
}

func TestSuperpixelSLICPanicsAndGrayscale(t *testing.T) {
	mustPanic(t, "small region", func() { ximgproc.SuperpixelSLIC(cv.NewMat(8, 8, 3), 1, 10) })

	// Grayscale input path.
	gray := cv.NewMat(24, 24, 1)
	for y := 0; y < 24; y++ {
		for x := 0; x < 24; x++ {
			gray.Data[y*24+x] = uint8((x / 8) * 80)
		}
	}
	labels, n := ximgproc.SuperpixelSLIC(gray, 8, 15)
	if n <= 0 || labels.Channels != 1 {
		t.Fatalf("grayscale SLIC failed: n=%d", n)
	}
}

func TestPeiLinPanicsOnEmpty(t *testing.T) {
	mustPanic(t, "zero intensity", func() { ximgproc.PeiLinNormalization(cv.NewMat(6, 6, 1)) })
}

func TestFastLineDetectorEmpty(t *testing.T) {
	// A blank image has no edges and thus no segments.
	if segs := ximgproc.FastLineDetector(cv.NewMat(16, 16, 1)); segs != nil {
		t.Errorf("expected no segments on blank image, got %d", len(segs))
	}
}

func mustPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic, got none", name)
		}
	}()
	fn()
}
