package texture_test

import (
	"testing"

	"github.com/malcolmston/opencv/texture"
)

func TestLBPImageKnown(t *testing.T) {
	// Only the east neighbour exceeds the centre -> bit set for k=0 only.
	img := makeGray([][]uint8{
		{50, 50, 50},
		{50, 100, 200},
		{50, 50, 50},
	})
	lbp := texture.LBPImage(img)
	if got := lbp.Data[1*3+1]; got != 128 {
		t.Fatalf("LBPImage interior code = %d, want 128", got)
	}
}

func TestLBPFlatImage(t *testing.T) {
	// On a flat image every neighbour is >= centre, so all bits set.
	img := fill(5, 5, 120)
	codes := texture.LBP(img, 1, 8)
	if codes[2][2] != 255 {
		t.Fatalf("flat LBP interior = %d, want 255", codes[2][2])
	}
}

func TestUniformDetection(t *testing.T) {
	if !texture.IsUniform(0b10000000, 8) {
		t.Error("single-bit pattern should be uniform")
	}
	if !texture.IsUniform(0b00011110, 8) {
		t.Error("contiguous run should be uniform")
	}
	if texture.IsUniform(0b10101010, 8) {
		t.Error("alternating pattern should be non-uniform")
	}
	if !texture.IsUniform(0, 8) || !texture.IsUniform(0xFF, 8) {
		t.Error("all-zeros and all-ones are uniform")
	}
}

func TestUniformBinCount(t *testing.T) {
	if got := texture.UniformBinCount(8); got != 59 {
		t.Fatalf("UniformBinCount(8) = %d, want 59", got)
	}
}

func TestRotationInvariance(t *testing.T) {
	// All single-bit rotations collapse to 1.
	for shift := 0; shift < 8; shift++ {
		code := uint32(1) << uint(shift)
		if got := texture.RotateMinimum(code, 8); got != 1 {
			t.Errorf("RotateMinimum(%b) = %d, want 1", code, got)
		}
	}
	// riu2 labelling by popcount for uniform patterns.
	if got := texture.MapUniformRotationInvariant(0b10000000, 8); got != 1 {
		t.Errorf("riu2 single bit = %d, want 1", got)
	}
	if got := texture.MapUniformRotationInvariant(0xFF, 8); got != 8 {
		t.Errorf("riu2 all ones = %d, want 8", got)
	}
	if got := texture.MapUniformRotationInvariant(0b10101010, 8); got != 9 {
		t.Errorf("riu2 non-uniform = %d, want 9 (points+1)", got)
	}
}

func TestLBPHistogramSumsToOne(t *testing.T) {
	img := makeGray([][]uint8{
		{10, 20, 30, 40},
		{40, 30, 20, 10},
		{10, 90, 20, 60},
		{70, 20, 80, 10},
	})
	h := texture.LBPUniformHistogram(img, 1, 8)
	if len(h) != texture.UniformBinCount(8) {
		t.Fatalf("histogram length = %d, want %d", len(h), texture.UniformBinCount(8))
	}
	var sum float64
	for _, v := range h {
		sum += v
	}
	if !approx(sum, 1.0, 1e-9) {
		t.Errorf("uniform histogram sums to %v, want 1", sum)
	}

	hr := texture.LBPRotationInvariantUniformHistogram(img, 1, 8)
	if len(hr) != 10 {
		t.Fatalf("riu2 histogram length = %d, want 10", len(hr))
	}
	sum = 0
	for _, v := range hr {
		sum += v
	}
	if !approx(sum, 1.0, 1e-9) {
		t.Errorf("riu2 histogram sums to %v, want 1", sum)
	}
}

func TestMapUniformRange(t *testing.T) {
	// Every possible 8-bit code maps within range and uniform codes are stable.
	bins := texture.UniformBinCount(8)
	for c := 0; c < 256; c++ {
		m := texture.MapUniform(uint32(c), 8)
		if m < 0 || m >= bins {
			t.Fatalf("MapUniform(%d) = %d out of [0,%d)", c, m, bins)
		}
	}
}
