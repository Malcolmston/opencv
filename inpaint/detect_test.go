package inpaint

import (
	"image/color"
	"testing"
)

func TestDetectScratchesVerticalLine(t *testing.T) {
	// Uniform 100 background with a 1-pixel bright vertical line at column 4.
	img := uniformMat(9, 9, 1, 100)
	for y := 0; y < 9; y++ {
		img.Set(y, 4, 0, 200)
	}
	opts := ScratchOptions{Threshold: 30, MaxWidth: 1, Vertical: true, DetectBright: true}
	mask := DetectScratches(img, opts)
	for y := 1; y < 8; y++ {
		if !mask.At(y, 4) {
			t.Fatalf("scratch pixel (%d,4) not detected", y)
		}
		if mask.At(y, 1) {
			t.Fatalf("background pixel (%d,1) wrongly detected", y)
		}
	}
}

func TestDetectTextRuns(t *testing.T) {
	img := uniformMat(12, 12, 1, 100)
	// a small high-contrast block ("text")
	for y := 4; y < 7; y++ {
		for x := 4; x < 7; x++ {
			img.Set(y, x, 0, 250)
		}
	}
	mask := DetectText(img, DefaultTextOptions())
	if mask.Count() == 0 {
		t.Fatalf("DetectText found nothing near a high-contrast block")
	}
}

func TestDetectBlotchesBrightSpeck(t *testing.T) {
	img := uniformMat(11, 11, 1, 80)
	img.Set(5, 5, 0, 220) // single bright speck
	mask := DetectBlotches(img, BlotchOptions{Threshold: 40, MaxSize: 2, DetectBright: true})
	if !mask.At(5, 5) {
		t.Fatalf("bright speck not detected")
	}
	if mask.At(0, 0) {
		t.Fatalf("flat background wrongly flagged")
	}
}

func TestMaskFromColor(t *testing.T) {
	img := uniformMat(5, 5, 3, 0)
	for y := 1; y < 3; y++ {
		for x := 1; x < 3; x++ {
			img.Set(y, x, 0, 255) // red block
		}
	}
	mask := MaskFromColor(img, color.RGBA{R: 255, G: 0, B: 0, A: 255}, 10)
	if mask.Count() != 4 {
		t.Fatalf("MaskFromColor count = %d, want 4", mask.Count())
	}
	if !mask.At(1, 1) {
		t.Fatalf("red pixel not selected")
	}
}

func TestMaskFromColorRange(t *testing.T) {
	img := uniformMat(5, 5, 3, 0)
	for y := 1; y < 3; y++ {
		for x := 1; x < 3; x++ {
			img.Set(y, x, 0, 230)
			img.Set(y, x, 1, 20)
			img.Set(y, x, 2, 20)
		}
	}
	lo := color.RGBA{R: 200, G: 0, B: 0, A: 255}
	hi := color.RGBA{R: 255, G: 50, B: 50, A: 255}
	mask := MaskFromColorRange(img, lo, hi)
	if mask.Count() != 4 {
		t.Fatalf("MaskFromColorRange count = %d, want 4", mask.Count())
	}
}
