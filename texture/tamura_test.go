package texture_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/texture"
)

func TestTamuraContrastKnown(t *testing.T) {
	// A two-value equal-population image has kurtosis 1, so Tamura contrast
	// equals the standard deviation (here 50).
	img := makeGray([][]uint8{{0, 0}, {100, 100}})
	if got := texture.TamuraContrast(img); !approx(got, 50, 1e-9) {
		t.Fatalf("TamuraContrast = %v, want 50", got)
	}
}

func TestTamuraContrastFlat(t *testing.T) {
	if got := texture.TamuraContrast(fill(4, 4, 77)); got != 0 {
		t.Errorf("flat contrast = %v, want 0", got)
	}
}

func TestTamuraCoarsenessFlat(t *testing.T) {
	// No scale produces cross-window difference, so the smallest size (1) wins.
	if got := texture.TamuraCoarseness(fill(8, 8, 100), 4); !approx(got, 1, 1e-9) {
		t.Errorf("flat coarseness = %v, want 1", got)
	}
}

func TestTamuraCoarsenessScale(t *testing.T) {
	// A coarse two-block image is coarser than fine random noise.
	blocks := cv.NewMat(16, 16, 1)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			if x >= 8 {
				blocks.Data[y*16+x] = 220
			}
		}
	}
	fine := cv.NewMat(16, 16, 1)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			if (x+y)%2 == 0 {
				fine.Data[y*16+x] = 220
			}
		}
	}
	cb := texture.TamuraCoarseness(blocks, 4)
	cf := texture.TamuraCoarseness(fine, 4)
	if !(cb > cf) {
		t.Errorf("block coarseness %.3f should exceed fine coarseness %.3f", cb, cf)
	}
}

func TestTamuraDirectionality(t *testing.T) {
	// Flat image: no gradients -> 0.
	if got := texture.TamuraDirectionality(fill(8, 8, 100), 16, 12); got != 0 {
		t.Errorf("flat directionality = %v, want 0", got)
	}
	// A single vertical edge is strongly directional -> near 1.
	edge := cv.NewMat(16, 16, 1)
	for y := 0; y < 16; y++ {
		for x := 8; x < 16; x++ {
			edge.Data[y*16+x] = 200
		}
	}
	d := texture.TamuraDirectionality(edge, 16, 10)
	if !(d > 0.9) {
		t.Errorf("vertical-edge directionality = %v, want > 0.9", d)
	}
	if d < 0 || d > 1 {
		t.Errorf("directionality %v out of [0,1]", d)
	}
}

func TestTamuraFeaturesBundle(t *testing.T) {
	img := makeGray([][]uint8{{0, 0}, {100, 100}})
	f := texture.TamuraFeatures(img)
	if !approx(f.Contrast, 50, 1e-9) {
		t.Errorf("bundle contrast = %v, want 50", f.Contrast)
	}
	if !approx(f.Roughness, f.Coarseness+f.Contrast, 1e-12) {
		t.Errorf("roughness != coarseness + contrast")
	}
}
