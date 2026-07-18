package inpaint

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestColorChangeIdentityOnUniform(t *testing.T) {
	// A flat image has zero gradients; scaling them and reintegrating against the
	// unchanged surroundings leaves the region equal to the surroundings.
	img := uniformMat(9, 9, 3, 120)
	mask := centerHoleMask(9, 9, 3, 3, 3, 3)
	out := ColorChange(img, mask, 2.0, 0.5, 1.0)
	for y := 3; y < 6; y++ {
		for x := 3; x < 6; x++ {
			for c := 0; c < 3; c++ {
				if out.At(y, x, c) != 120 {
					t.Fatalf("ColorChange flat (%d,%d,%d) = %d, want 120", y, x, c, out.At(y, x, c))
				}
			}
		}
	}
}

func TestTextureFlatteningRuns(t *testing.T) {
	img := uniformMat(9, 9, 3, 100)
	// a bright cross of texture inside
	for x := 0; x < 9; x++ {
		img.Set(4, x, 0, 180)
	}
	mask := centerHoleMask(9, 9, 2, 2, 5, 5)
	out := TextureFlattening(img, mask, 20, 100)
	if out.Rows != 9 || out.Cols != 9 || out.Channels != 3 {
		t.Fatalf("unexpected output shape")
	}
}

func TestIlluminationChangeRuns(t *testing.T) {
	img := rampMat(9, 9, 6, 0, 20)
	rgb := toThree(img)
	mask := centerHoleMask(9, 9, 2, 2, 5, 5)
	out := IlluminationChange(rgb, mask, 0.2, 0.4)
	if out.Channels != 3 {
		t.Fatalf("expected 3 channels")
	}
}

// toThree replicates a single-channel Mat into three channels.
func toThree(m *cv.Mat) *cv.Mat {
	out := cv.NewMat(m.Rows, m.Cols, 3)
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			v := m.At(y, x, 0)
			out.Set(y, x, 0, v)
			out.Set(y, x, 1, v)
			out.Set(y, x, 2, v)
		}
	}
	return out
}
