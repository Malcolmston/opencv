package objdetect

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// TestDescriptorSizeMatchesGeometry checks the descriptor length equals the
// cell/block math, both for the canonical geometry and for what Compute
// actually returns.
func TestDescriptorSizeMatchesGeometry(t *testing.T) {
	h := NewHOGDescriptor()
	// Canonical Dalal-Triggs / OpenCV person detector length.
	if got := h.DescriptorSize(); got != 3780 {
		t.Fatalf("DescriptorSize() = %d, want 3780", got)
	}

	// Independently recompute from the geometry.
	cpbX := h.BlockSize.Width / h.CellSize.Width
	cpbY := h.BlockSize.Height / h.CellSize.Height
	blocksX := (h.WinSize.Width-h.BlockSize.Width)/h.BlockStride.Width + 1
	blocksY := (h.WinSize.Height-h.BlockSize.Height)/h.BlockStride.Height + 1
	want := blocksX * blocksY * cpbX * cpbY * h.NBins
	if want != 3780 {
		t.Fatalf("geometry recompute = %d, want 3780", want)
	}

	// Compute on an exactly-window-sized image must produce that many values.
	img := cv.NewMat(h.WinSize.Height, h.WinSize.Width, 1)
	desc := h.Compute(img)
	if len(desc) != want {
		t.Fatalf("len(Compute) = %d, want %d", len(desc), want)
	}
}

// TestDescriptorSizeSmallGeometry verifies the formula for a non-default
// geometry too.
func TestDescriptorSizeSmallGeometry(t *testing.T) {
	h := &HOGDescriptor{
		WinSize:     Size{16, 16},
		BlockSize:   Size{8, 8},
		BlockStride: Size{4, 4},
		CellSize:    Size{4, 4},
		NBins:       6,
	}
	// blocksX = (16-8)/4+1 = 3, blocksY = 3, cells/block = 2x2, bins = 6.
	want := 3 * 3 * 2 * 2 * 6
	if got := h.DescriptorSize(); got != want {
		t.Fatalf("DescriptorSize() = %d, want %d", got, want)
	}
	img := cv.NewMat(16, 16, 1)
	if got := len(h.Compute(img)); got != want {
		t.Fatalf("len(Compute) = %d, want %d", got, want)
	}
}

// TestHOGOrientationBin checks that a horizontal intensity ramp (a purely
// vertical-edge / horizontal-gradient field) concentrates all descriptor
// energy in orientation bin 0.
func TestHOGOrientationBin(t *testing.T) {
	h := NewHOGDescriptor()
	w, ht := h.WinSize.Width, h.WinSize.Height

	// Horizontal ramp: intensity increases with x, so gradient points along
	// +x (angle 0) everywhere; gy is 0.
	img := cv.NewMat(ht, w, 1)
	for y := 0; y < ht; y++ {
		for x := 0; x < w; x++ {
			img.Set(y, x, 0, uint8(x%256))
		}
	}
	desc := h.Compute(img)

	// Sum descriptor energy per orientation bin across the whole vector.
	perBin := make([]float64, h.NBins)
	for i, v := range desc {
		perBin[i%h.NBins] += math.Abs(v)
	}
	argmax := 0
	for k := 1; k < h.NBins; k++ {
		if perBin[k] > perBin[argmax] {
			argmax = k
		}
	}
	if argmax != 0 {
		t.Fatalf("dominant orientation bin = %d, want 0 (perBin=%v)", argmax, perBin)
	}
	// Bin 0 should dominate by a wide margin.
	var others float64
	for k := 1; k < h.NBins; k++ {
		others += perBin[k]
	}
	if perBin[0] <= others {
		t.Fatalf("bin0 energy %.3f not dominant over rest %.3f", perBin[0], others)
	}
}

// TestHOGVerticalGradientBin checks a vertical intensity ramp lands in the
// 90-degree bin (bin 4 for NBins=9).
func TestHOGVerticalGradientBin(t *testing.T) {
	h := NewHOGDescriptor()
	w, ht := h.WinSize.Width, h.WinSize.Height
	img := cv.NewMat(ht, w, 1)
	for y := 0; y < ht; y++ {
		for x := 0; x < w; x++ {
			img.Set(y, x, 0, uint8(y%256))
		}
	}
	desc := h.Compute(img)
	perBin := make([]float64, h.NBins)
	for i, v := range desc {
		perBin[i%h.NBins] += math.Abs(v)
	}
	argmax := 0
	for k := 1; k < h.NBins; k++ {
		if perBin[k] > perBin[argmax] {
			argmax = k
		}
	}
	// 90 degrees / (180/9=20) = bin 4.
	if argmax != 4 {
		t.Fatalf("dominant orientation bin = %d, want 4 (perBin=%v)", argmax, perBin)
	}
}

// TestHOGDetectMultiScale exercises the sliding-window SVM path and verifies
// windows and scaling behave.
func TestHOGDetectMultiScale(t *testing.T) {
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

	// All-zero weights => every window scores 0. With hitThreshold above 0,
	// nothing fires.
	zero := make([]float64, descLen)
	if hits := h.DetectMultiScale(img, zero, 0.5, 1.5); len(hits) != 0 {
		t.Fatalf("expected no hits with zero weights and positive threshold, got %d", len(hits))
	}

	// Bias-only weights (length descLen+1) with a large positive bias => every
	// window fires. Verify at least the base-level windows are returned and
	// coordinates fall inside the image.
	biased := make([]float64, descLen+1)
	biased[descLen] = 100
	hits := h.DetectMultiScale(img, biased, 1.0, 1.5)
	if len(hits) == 0 {
		t.Fatal("expected hits with large positive bias")
	}
	for _, r := range hits {
		if r.X < 0 || r.Y < 0 || r.X+r.Width > 32 || r.Y+r.Height > 32 {
			t.Fatalf("hit %+v out of image bounds", r)
		}
	}
}

// TestHOGInvalidGeometryPanics verifies validation.
func TestHOGInvalidGeometryPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on non-divisible block geometry")
		}
	}()
	h := &HOGDescriptor{
		WinSize:     Size{16, 16},
		BlockSize:   Size{8, 8},
		BlockStride: Size{8, 8},
		CellSize:    Size{3, 3}, // 8 % 3 != 0
		NBins:       9,
	}
	h.DescriptorSize()
}
