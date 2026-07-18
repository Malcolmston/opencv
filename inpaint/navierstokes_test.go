package inpaint

import "testing"

func TestDiffusionUniformExact(t *testing.T) {
	img := uniformMat(9, 9, 1, 60)
	mask := centerHoleMask(9, 9, 3, 3, 3, 3)
	out := InpaintDiffusion(img, mask)
	for y := 3; y < 6; y++ {
		for x := 3; x < 6; x++ {
			if out.At(y, x, 0) != 60 {
				t.Fatalf("diffusion uniform (%d,%d) = %d, want 60", y, x, out.At(y, x, 0))
			}
		}
	}
}

func TestDiffusionRampReproduced(t *testing.T) {
	// Harmonic fill of a hole in a linear ramp reproduces the ramp (±rounding).
	img := rampMat(11, 11, 5, 3, 20)
	mask := centerHoleMask(11, 11, 4, 4, 3, 3)
	out := InpaintDiffusion(img, mask)
	for y := 4; y < 7; y++ {
		for x := 4; x < 7; x++ {
			if d := absU8(out.At(y, x, 0), img.At(y, x, 0)); d > 1 {
				t.Fatalf("diffusion ramp (%d,%d) = %d, want %d (d=%d)", y, x, out.At(y, x, 0), img.At(y, x, 0), d)
			}
		}
	}
}

func TestNavierStokesUniformExact(t *testing.T) {
	img := uniformMat(9, 9, 1, 90)
	mask := centerHoleMask(9, 9, 3, 3, 3, 3)
	out := InpaintNavierStokes(img, mask, 50)
	for y := 3; y < 6; y++ {
		for x := 3; x < 6; x++ {
			if out.At(y, x, 0) != 90 {
				t.Fatalf("NS uniform (%d,%d) = %d, want 90", y, x, out.At(y, x, 0))
			}
		}
	}
}

func TestNavierStokesRampApprox(t *testing.T) {
	img := rampMat(11, 11, 5, 3, 20)
	mask := centerHoleMask(11, 11, 4, 4, 3, 3)
	out := InpaintNavierStokes(img, mask, 100)
	for y := 4; y < 7; y++ {
		for x := 4; x < 7; x++ {
			if d := absU8(out.At(y, x, 0), img.At(y, x, 0)); d > 3 {
				t.Fatalf("NS ramp (%d,%d) = %d, want %d (d=%d)", y, x, out.At(y, x, 0), img.At(y, x, 0), d)
			}
		}
	}
}
