package photo

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestInpaintFillsUniformRegion(t *testing.T) {
	for _, tc := range []struct {
		name   string
		method InpaintMethod
	}{
		{"Telea", InpaintTelea},
		{"NS", InpaintNS},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const bg = 120
			img := cv.NewMat(20, 20, 3)
			for i := range img.Data {
				img.Data[i] = bg
			}
			mask := cv.NewMat(20, 20, 1)
			// Corrupt a 4x4 hole and mask it.
			for y := 8; y < 12; y++ {
				for x := 8; x < 12; x++ {
					img.SetPixel(y, x, []uint8{0, 255, 0})
					mask.Set(y, x, 0, 255)
				}
			}

			out := Inpaint(img, mask, 3, tc.method)
			if out.Rows != 20 || out.Cols != 20 || out.Channels != 3 {
				t.Fatalf("unexpected shape %dx%dx%d", out.Rows, out.Cols, out.Channels)
			}
			// Filled pixels must match the surrounding background within tolerance.
			for y := 8; y < 12; y++ {
				for x := 8; x < 12; x++ {
					for c := 0; c < 3; c++ {
						if d := math.Abs(float64(out.At(y, x, c)) - bg); d > 3 {
							t.Errorf("%s: pixel (%d,%d,%d)=%d far from bg %d (d=%.0f)",
								tc.name, y, x, c, out.At(y, x, c), bg, d)
						}
					}
				}
			}
			// Unmasked pixels must be untouched.
			if out.At(0, 0, 1) != bg {
				t.Errorf("%s: unmasked pixel changed to %d", tc.name, out.At(0, 0, 1))
			}
			// The original image must be untouched.
			if img.At(9, 9, 1) != 255 {
				t.Errorf("%s: input image was mutated", tc.name)
			}
		})
	}
}
