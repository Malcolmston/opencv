package objdetect

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestComputeGradient(t *testing.T) {
	h := NewHOGDescriptor()
	// Horizontal ramp: gradient points along +x, orientation ~0 in the interior.
	w, ht := 20, 16
	img := cv.NewMat(ht, w, 1)
	for y := 0; y < ht; y++ {
		for x := 0; x < w; x++ {
			img.Set(y, x, 0, uint8(x*10%256))
		}
	}
	mag, ang, gw, gh := h.ComputeGradient(img)
	if gw != w || gh != ht {
		t.Fatalf("gradient dims = %dx%d, want %dx%d", gw, gh, w, ht)
	}
	if len(mag) != w*ht || len(ang) != w*ht {
		t.Fatalf("gradient slice lengths = %d/%d, want %d", len(mag), len(ang), w*ht)
	}
	// Interior pixel: gradient magnitude positive, angle near 0 (or 180).
	i := 8*w + 10
	if mag[i] <= 0 {
		t.Fatalf("interior magnitude = %v, want > 0", mag[i])
	}
	if ang[i] > 5 && ang[i] < 175 {
		t.Fatalf("interior angle = %v, want near 0/180 for horizontal gradient", ang[i])
	}
}

func TestDetectMultiScaleWeights(t *testing.T) {
	h := &HOGDescriptor{
		WinSize:     Size{16, 16},
		BlockSize:   Size{8, 8},
		BlockStride: Size{8, 8},
		CellSize:    Size{8, 8},
		NBins:       9,
	}
	img := cv.NewMat(32, 32, 1)
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			img.Set(y, x, 0, uint8(x*4%256))
		}
	}
	descLen := h.DescriptorSize()
	biased := make([]float64, descLen+1)
	biased[descLen] = 100
	hits, scores := h.DetectMultiScaleWeights(img, biased, 1.0, 1.5)
	if len(hits) == 0 {
		t.Fatal("expected hits with large positive bias")
	}
	if len(hits) != len(scores) {
		t.Fatalf("hits/scores length mismatch: %d vs %d", len(hits), len(scores))
	}
	for _, s := range scores {
		if math.Abs(s-100) > 1e-6 {
			t.Fatalf("score = %v, want 100 (bias only, zero weights)", s)
		}
	}
}

// drawSilhouette paints a dark head-and-torso figure on a light background.
func drawSilhouette(img *cv.Mat) {
	w, ht := img.Cols, img.Rows
	img.SetTo(210)
	cx := float64(w) / 2
	headR := float64(w) * 0.16
	headCy := float64(ht) * 0.14
	bodyTop := float64(ht) * 0.24
	bodyHalf := float64(w) * 0.26
	for y := 0; y < ht; y++ {
		fy := float64(y)
		for x := 0; x < w; x++ {
			fx := float64(x)
			inHead := math.Hypot(fx-cx, fy-headCy) <= headR
			inBody := fy >= bodyTop && math.Abs(fx-cx) <= bodyHalf
			if inHead || inBody {
				img.Set(y, x, 0, 60)
			}
		}
	}
}

func TestDefaultPeopleDetector(t *testing.T) {
	h := NewHOGDescriptor()
	w := h.DefaultPeopleDetector()
	if len(w) != h.DescriptorSize()+1 {
		t.Fatalf("detector length = %d, want %d", len(w), h.DescriptorSize()+1)
	}

	score := func(img *cv.Mat) float64 {
		desc := h.Compute(img)
		s := w[len(w)-1] // bias
		for i, v := range desc {
			s += w[i] * v
		}
		return s
	}

	// A flat window has an all-zero descriptor, so it scores exactly the (negative)
	// bias.
	flat := cv.NewMat(h.WinSize.Height, h.WinSize.Width, 1)
	flat.SetTo(128)
	if fs := score(flat); fs >= 0 {
		t.Fatalf("flat window score = %v, want negative", fs)
	}

	// A person-like silhouette scores positively.
	sil := cv.NewMat(h.WinSize.Height, h.WinSize.Width, 1)
	drawSilhouette(sil)
	if ss := score(sil); ss <= 0 {
		t.Fatalf("silhouette score = %v, want positive", ss)
	}
}
