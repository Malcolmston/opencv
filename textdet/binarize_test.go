package textdet

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestOtsuThresholdBimodal(t *testing.T) {
	// Half the pixels at 50, half at 200: Otsu separates the two clusters and,
	// with the first-maximum tie rule, returns the lower cluster value 50.
	m := newGray(4, 4, 50)
	paintRect(m, 0, 0, 4, 2, 50)
	paintRect(m, 0, 2, 4, 2, 200)
	got, err := OtsuThreshold(m)
	if err != nil {
		t.Fatal(err)
	}
	if got != 50 {
		t.Fatalf("OtsuThreshold = %d, want 50", got)
	}
}

func TestBinarizePolarity(t *testing.T) {
	m := newGray(2, 3, 0)
	m.Data = []uint8{10, 100, 200, 10, 100, 200}
	dark, err := Binarize(m, 120, DarkText)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint8{255, 255, 0, 255, 255, 0}
	for i := range want {
		if dark.Data[i] != want[i] {
			t.Fatalf("DarkText Binarize = %v, want %v", dark.Data, want)
		}
	}
	bright, _ := Binarize(m, 120, BrightText)
	wantB := []uint8{0, 0, 255, 0, 0, 255}
	for i := range wantB {
		if bright.Data[i] != wantB[i] {
			t.Fatalf("BrightText Binarize = %v, want %v", bright.Data, wantB)
		}
	}
}

func TestForegroundRatio(t *testing.T) {
	m := newGray(1, 4, 0)
	m.Data = []uint8{10, 20, 200, 250}
	r, err := ForegroundRatio(m, 100, DarkText)
	if err != nil {
		t.Fatal(err)
	}
	if r != 0.5 {
		t.Fatalf("ForegroundRatio = %v, want 0.5", r)
	}
}

func TestIntegralImageKnownValues(t *testing.T) {
	// 3x3 image with samples 1..9 row-major.
	m := cv.NewMat(3, 3, 1)
	m.Data = []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9}
	ii, err := NewIntegralImage(m)
	if err != nil {
		t.Fatal(err)
	}
	// Whole-image sum = 45, mean = 5.
	full := cv.Rect{X: 0, Y: 0, Width: 3, Height: 3}
	if s := ii.Sum(full); s != 45 {
		t.Fatalf("Sum full = %v, want 45", s)
	}
	if mn := ii.Mean(full); mn != 5 {
		t.Fatalf("Mean full = %v, want 5", mn)
	}
	// Top-left 2x2 block {1,2,4,5} sums to 12, mean 3.
	tl := cv.Rect{X: 0, Y: 0, Width: 2, Height: 2}
	if s := ii.Sum(tl); s != 12 {
		t.Fatalf("Sum 2x2 = %v, want 12", s)
	}
	// Population std of 1..9: variance = 60/9 = 6.6667, std = 2.5820.
	mean, std := ii.MeanStdDev(full)
	if mean != 5 {
		t.Fatalf("MeanStdDev mean = %v, want 5", mean)
	}
	if math.Abs(std-math.Sqrt(60.0/9.0)) > 1e-9 {
		t.Fatalf("MeanStdDev std = %v, want %v", std, math.Sqrt(60.0/9.0))
	}
}

func TestIntegralImageClamp(t *testing.T) {
	m := cv.NewMat(2, 2, 1)
	m.Data = []uint8{10, 20, 30, 40}
	ii, _ := NewIntegralImage(m)
	// A rectangle partly outside the image is clamped to the 2x2 region.
	got := ii.Sum(cv.Rect{X: -5, Y: -5, Width: 20, Height: 20})
	if got != 100 {
		t.Fatalf("clamped Sum = %v, want 100", got)
	}
}

func TestSauvolaSeparatesInkFromPaper(t *testing.T) {
	// Dark glyph (value 30) on a light page (value 220): Sauvola marks the
	// glyph as ink and leaves the paper as background.
	m := newGray(15, 15, 220)
	paintRect(m, 5, 5, 5, 5, 30)
	mask, err := Sauvola(m, 4, 0.2, 128, DarkText)
	if err != nil {
		t.Fatal(err)
	}
	// Centre of the glyph must be ink.
	if mask.Data[7*15+7] != 255 {
		t.Fatalf("Sauvola glyph centre not ink")
	}
	// A far corner of the page must be background.
	if mask.Data[0] != 0 {
		t.Fatalf("Sauvola paper corner marked ink")
	}
}

func TestNiblackAndWolfAndBernsenRun(t *testing.T) {
	m := newGray(20, 20, 210)
	paintRect(m, 6, 6, 8, 8, 40)
	for _, tc := range []struct {
		name string
		fn   func() (*cv.Mat, error)
	}{
		{"niblack", func() (*cv.Mat, error) { return Niblack(m, 5, -0.2, DarkText) }},
		{"wolf", func() (*cv.Mat, error) { return Wolf(m, 5, 0.5, DarkText) }},
		{"bernsen", func() (*cv.Mat, error) { return Bernsen(m, 5, 15, DarkText) }},
		{"adaptivemean", func() (*cv.Mat, error) { return AdaptiveMean(m, 5, 10, DarkText) }},
	} {
		mask, err := tc.fn()
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if mask.Data[10*20+10] != 255 {
			t.Fatalf("%s: glyph centre not ink", tc.name)
		}
	}
}

func TestBinarizeErrorsAndEmpty(t *testing.T) {
	var empty cv.Mat
	if _, err := OtsuThreshold(&empty); err != ErrEmpty {
		t.Fatalf("empty OtsuThreshold err = %v, want ErrEmpty", err)
	}
	if _, err := AdaptiveMean(newGray(4, 4, 0), 0, 0, DarkText); err != ErrInvalidArgument {
		t.Fatalf("radius 0 err = %v, want ErrInvalidArgument", err)
	}
}
