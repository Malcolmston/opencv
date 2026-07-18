package texture_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/texture"
)

func TestBoxCountingFilledPlane(t *testing.T) {
	// A fully-foreground square fills the plane: dimension exactly 2.
	full := fill(16, 16, 255)
	if got := texture.BoxCountingDimension(full, 1); !approx(got, 2.0, 1e-9) {
		t.Fatalf("box dimension of filled plane = %v, want 2", got)
	}
}

func TestBoxCountingLine(t *testing.T) {
	// A single foreground row is line-like: dimension exactly 1.
	line := cv.NewMat(16, 16, 1)
	for x := 0; x < 16; x++ {
		line.Data[x] = 255
	}
	if got := texture.BoxCountingDimension(line, 1); !approx(got, 1.0, 1e-9) {
		t.Fatalf("box dimension of a line = %v, want 1", got)
	}
}

func TestBoxCountingEmpty(t *testing.T) {
	if got := texture.BoxCountingDimension(fill(8, 8, 0), 128); got != 0 {
		t.Errorf("empty foreground dimension = %v, want 0", got)
	}
}

func TestDifferentialBoxCountingRange(t *testing.T) {
	// A flat intensity surface is planar: DBC dimension near 2.
	d := texture.DifferentialBoxCounting(fill(32, 32, 100))
	if d < 1.5 || d > 2.5 {
		t.Errorf("DBC of flat image = %v, want roughly 2", d)
	}
	// FractalDimension is an alias.
	if texture.FractalDimension(fill(32, 32, 100)) != d {
		t.Errorf("FractalDimension should equal DifferentialBoxCounting")
	}
}

func TestLacunarityUniform(t *testing.T) {
	// A fully-foreground image has constant box mass, so lacunarity is 1.
	if got := texture.Lacunarity(fill(16, 16, 255), 1, 4); !approx(got, 1.0, 1e-9) {
		t.Fatalf("lacunarity of uniform foreground = %v, want 1", got)
	}
}

func TestLacunarityClustered(t *testing.T) {
	// A clustered (half-filled) image is gappier than a uniform one:
	// lacunarity > 1.
	img := cv.NewMat(16, 16, 1)
	for y := 0; y < 16; y++ {
		for x := 0; x < 8; x++ {
			img.Data[y*16+x] = 255
		}
	}
	if got := texture.Lacunarity(img, 1, 4); !(got > 1.0) {
		t.Errorf("lacunarity of clustered image = %v, want > 1", got)
	}
}

func TestLacunarityPanics(t *testing.T) {
	assertPanic(t, "boxSize too large", func() {
		texture.Lacunarity(fill(4, 4, 255), 1, 5)
	})
}
