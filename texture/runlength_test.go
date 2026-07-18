package texture_test

import (
	"testing"

	"github.com/malcolmston/opencv/texture"
)

// A 2x2 image with two horizontal runs of length 2 (one per level) gives a
// fully-known run-length matrix.
func twoRuns() *texture.RunLengthMatrix {
	img := makeGray([][]uint8{{0, 0}, {200, 200}})
	return texture.NewRunLengthMatrix(img, 2, texture.Horizontal)
}

func TestRunLengthCounts(t *testing.T) {
	rl := twoRuns()
	if rl.TotalRuns() != 2 {
		t.Fatalf("TotalRuns = %v, want 2", rl.TotalRuns())
	}
	if rl.At(0, 2) != 1 || rl.At(1, 2) != 1 {
		t.Fatalf("expected one length-2 run per level, got %v,%v", rl.At(0, 2), rl.At(1, 2))
	}
	if rl.At(0, 1) != 0 {
		t.Fatalf("no length-1 runs expected, got %v", rl.At(0, 1))
	}
}

func TestRunLengthFeaturesKnown(t *testing.T) {
	rl := twoRuns()
	cases := []struct {
		name string
		got  float64
		want float64
	}{
		{"ShortRunEmphasis", rl.ShortRunEmphasis(), 0.25},
		{"LongRunEmphasis", rl.LongRunEmphasis(), 4.0},
		{"GrayLevelNonUniformity", rl.GrayLevelNonUniformity(), 1.0},
		{"RunLengthNonUniformity", rl.RunLengthNonUniformity(), 2.0},
		{"RunPercentage", rl.RunPercentage(), 0.5},
	}
	for _, c := range cases {
		if !approx(c.got, c.want, 1e-9) {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

func TestRunLengthGrayLevelEmphasis(t *testing.T) {
	rl := twoRuns()
	// LGRE weights low levels (1/(i+1)^2); level 0 -> 1, level 1 -> 1/4.
	// Two runs of length 2: (1/2)*(1/1 + 1/4) = 0.625.
	if got := rl.LowGrayLevelRunEmphasis(); !approx(got, 0.625, 1e-9) {
		t.Errorf("LGRE = %v, want 0.625", got)
	}
	// HGRE: (1/2)*(1 + 4) = 2.5.
	if got := rl.HighGrayLevelRunEmphasis(); !approx(got, 2.5, 1e-9) {
		t.Errorf("HGRE = %v, want 2.5", got)
	}
}

func TestRunLengthAllShort(t *testing.T) {
	// A checkerboard has only length-1 runs, so RunPercentage is 1 and short
	// and long emphasis both equal 1.
	img := makeGray([][]uint8{{0, 200, 0}, {200, 0, 200}, {0, 200, 0}})
	rl := texture.NewRunLengthMatrix(img, 2, texture.Horizontal)
	if !approx(rl.RunPercentage(), 1.0, 1e-9) {
		t.Errorf("checkerboard RunPercentage = %v, want 1", rl.RunPercentage())
	}
	if !approx(rl.ShortRunEmphasis(), 1.0, 1e-9) {
		t.Errorf("checkerboard SRE = %v, want 1", rl.ShortRunEmphasis())
	}
	if !approx(rl.LongRunEmphasis(), 1.0, 1e-9) {
		t.Errorf("checkerboard LRE = %v, want 1", rl.LongRunEmphasis())
	}
}

func TestRunLengthFeaturesBundle(t *testing.T) {
	f := twoRuns().Features()
	if !approx(f.ShortRunEmphasis, 0.25, 1e-9) || !approx(f.LongRunEmphasis, 4.0, 1e-9) {
		t.Errorf("bundle mismatch: %+v", f)
	}
}

func TestRunLengthVerticalLongRun(t *testing.T) {
	// A column of identical pixels is one long vertical run.
	img := makeGray([][]uint8{{0, 200}, {0, 200}, {0, 200}})
	rl := texture.NewRunLengthMatrix(img, 2, texture.Vertical)
	if rl.At(0, 3) != 1 || rl.At(1, 3) != 1 {
		t.Fatalf("expected length-3 vertical runs, got %v,%v", rl.At(0, 3), rl.At(1, 3))
	}
}
