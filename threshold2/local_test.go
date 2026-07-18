package threshold2

import "testing"

func TestAdaptiveMeanUniform(t *testing.T) {
	img := grayImage(9, 9, func(_, _ int) uint8 { return 100 })
	// With c = 0 the local mean equals the pixel, so nothing is foreground.
	eq, err := AdaptiveMean(img, 3, 0, ObjectBright)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range eq.Data {
		if v != 0 {
			t.Fatalf("AdaptiveMean c=0 at %d = %d, want 0", i, v)
		}
	}
	// With c = 10 the threshold drops below every pixel, so all foreground.
	all, err := AdaptiveMean(img, 3, 10, ObjectBright)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range all.Data {
		if v != 255 {
			t.Fatalf("AdaptiveMean c=10 at %d = %d, want 255", i, v)
		}
	}
	if _, err := AdaptiveMean(img, 4, 0, ObjectBright); err == nil {
		t.Fatal("expected error for even window")
	}
}

func TestAdaptiveGaussianUniform(t *testing.T) {
	img := grayImage(9, 9, func(_, _ int) uint8 { return 120 })
	all, err := AdaptiveGaussian(img, 5, 5, ObjectBright)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range all.Data {
		if v != 255 {
			t.Fatalf("AdaptiveGaussian at %d = %d, want 255", i, v)
		}
	}
}

func TestAdaptiveMedian(t *testing.T) {
	img := grayImage(9, 9, func(_, _ int) uint8 { return 80 })
	all, err := AdaptiveMedian(img, 3, 5, ObjectBright)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range all.Data {
		if v != 255 {
			t.Fatalf("AdaptiveMedian at %d = %d, want 255", i, v)
		}
	}
}

func TestSauvolaUniform(t *testing.T) {
	// On a flat field std is 0, so T = mean*(1-k) = 100*0.5 = 50 < 100:
	// every pixel is above threshold and foreground.
	img := grayImage(9, 9, func(_, _ int) uint8 { return 100 })
	dst, err := Sauvola(img, 5, 0.5, 128, ObjectBright)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range dst.Data {
		if v != 255 {
			t.Fatalf("Sauvola at %d = %d, want 255", i, v)
		}
	}
}

func TestNiblackUniform(t *testing.T) {
	// Flat field: std 0 so T = mean, pixel == mean is not strictly greater,
	// so nothing is foreground.
	img := grayImage(9, 9, func(_, _ int) uint8 { return 100 })
	dst, err := Niblack(img, 5, -0.2, ObjectBright)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range dst.Data {
		if v != 0 {
			t.Fatalf("Niblack at %d = %d, want 0", i, v)
		}
	}
}

func TestBernsenUniform(t *testing.T) {
	bright := grayImage(9, 9, func(_, _ int) uint8 { return 200 })
	dst, err := Bernsen(bright, 3, 15, ObjectBright)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range dst.Data {
		if v != 255 {
			t.Fatalf("Bernsen bright at %d = %d, want 255", i, v)
		}
	}
	dark := grayImage(9, 9, func(_, _ int) uint8 { return 40 })
	dst2, _ := Bernsen(dark, 3, 15, ObjectBright)
	for i, v := range dst2.Data {
		if v != 0 {
			t.Fatalf("Bernsen dark at %d = %d, want 0", i, v)
		}
	}
}

func TestWolfNICKPhansalkarRun(t *testing.T) {
	img := twoBands(16, 16, 60, 190)
	if _, err := Wolf(img, 5, 0.5, ObjectBright); err != nil {
		t.Fatal(err)
	}
	if _, err := NICK(img, 5, -0.1, ObjectBright); err != nil {
		t.Fatal(err)
	}
	if _, err := Phansalkar(img, 5, 0.25, 0.5, ObjectBright); err != nil {
		t.Fatal(err)
	}
}

// BenchmarkSauvola exercises the heaviest local routine on a 128x128 image.
func BenchmarkSauvola(b *testing.B) {
	img := grayImage(128, 128, func(y, x int) uint8 {
		return uint8((x*2 + y) % 256)
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := Sauvola(img, 15, 0.5, 128, ObjectDark); err != nil {
			b.Fatal(err)
		}
	}
}
