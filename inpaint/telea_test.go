package inpaint

import "testing"

func TestTeleaUniformExact(t *testing.T) {
	img := uniformMat(9, 9, 3, 128)
	mask := centerHoleMask(9, 9, 3, 3, 3, 3)
	out := InpaintTelea(img, mask, 3)
	for y := 3; y < 6; y++ {
		for x := 3; x < 6; x++ {
			for c := 0; c < 3; c++ {
				if out.At(y, x, c) != 128 {
					t.Fatalf("Telea uniform fill (%d,%d,%d) = %d, want 128", y, x, c, out.At(y, x, c))
				}
			}
		}
	}
}

func TestTeleaDoesNotModifyKnown(t *testing.T) {
	img := rampMat(11, 11, 6, 0, 10)
	mask := centerHoleMask(11, 11, 4, 4, 3, 3)
	out := InpaintTelea(img, mask, 3)
	for y := 0; y < 11; y++ {
		for x := 0; x < 11; x++ {
			if mask.At(y, x) {
				continue
			}
			if out.At(y, x, 0) != img.At(y, x, 0) {
				t.Fatalf("known pixel (%d,%d) changed %d->%d", y, x, img.At(y, x, 0), out.At(y, x, 0))
			}
		}
	}
}

func TestTeleaRampApprox(t *testing.T) {
	// A horizontal ramp: gradient-corrected filling should recover it closely.
	img := rampMat(11, 11, 6, 0, 20)
	mask := centerHoleMask(11, 11, 4, 4, 3, 3)
	out := InpaintTelea(img, mask, 4)
	for y := 4; y < 7; y++ {
		for x := 4; x < 7; x++ {
			want := img.At(y, x, 0)
			if d := absU8(out.At(y, x, 0), want); d > 8 {
				t.Fatalf("Telea ramp (%d,%d) = %d, want ~%d (d=%d)", y, x, out.At(y, x, 0), want, d)
			}
		}
	}
}
