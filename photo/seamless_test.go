package photo

import (
	"image"
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestSeamlessCloneMatchesBackgroundAtSeam(t *testing.T) {
	// Uniform destination background.
	bg := []uint8{100, 150, 200}
	dst := cv.NewMat(24, 24, 3)
	for p := 0; p < dst.Total(); p++ {
		copy(dst.Data[p*3:p*3+3], bg)
	}

	// A flat source patch of a different colour with a filled square mask.
	src := cv.NewMat(12, 12, 3)
	for p := 0; p < src.Total(); p++ {
		copy(src.Data[p*3:p*3+3], []uint8{50, 50, 50})
	}
	mask := cv.NewMat(12, 12, 1)
	for y := 3; y < 9; y++ {
		for x := 3; x < 9; x++ {
			mask.Set(y, x, 0, 255)
		}
	}

	out := SeamlessClone(src, dst, mask, image.Point{X: 12, Y: 12}, NormalClone)
	if out.Rows != 24 || out.Cols != 24 || out.Channels != 3 {
		t.Fatalf("unexpected shape %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}

	// The source patch is flat, so Poisson blending with a uniform destination
	// boundary must reproduce the background across the whole cloned region.
	for y := 9; y <= 15; y++ {
		for x := 9; x <= 15; x++ {
			for c := 0; c < 3; c++ {
				if d := math.Abs(float64(out.At(y, x, c)) - float64(bg[c])); d > 2 {
					t.Errorf("clone pixel (%d,%d,%d)=%d far from bg %d (d=%.0f)",
						y, x, c, out.At(y, x, c), bg[c], d)
				}
			}
		}
	}

	// Destination outside the clone is untouched.
	if out.At(0, 0, 0) != bg[0] {
		t.Errorf("destination corner changed to %d", out.At(0, 0, 0))
	}
	// Original destination not mutated.
	if dst.At(12, 12, 0) != bg[0] {
		t.Errorf("input destination was mutated")
	}
}

func TestSeamlessCloneEmptyMask(t *testing.T) {
	dst := cv.NewMat(8, 8, 3)
	src := cv.NewMat(8, 8, 3)
	mask := cv.NewMat(8, 8, 1) // all zero
	out := SeamlessClone(src, dst, mask, image.Point{X: 4, Y: 4}, NormalClone)
	for i := range out.Data {
		if out.Data[i] != dst.Data[i] {
			t.Fatal("empty mask should leave destination unchanged")
		}
	}
}
