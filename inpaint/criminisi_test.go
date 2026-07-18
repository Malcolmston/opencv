package inpaint

import "testing"

func TestCriminisiUniformExact(t *testing.T) {
	img := uniformMat(16, 16, 3, 100)
	mask := centerHoleMask(16, 16, 6, 6, 4, 4)
	out := InpaintCriminisi(img, mask, DefaultCriminisiOptions())
	for y := 6; y < 10; y++ {
		for x := 6; x < 10; x++ {
			for c := 0; c < 3; c++ {
				if out.At(y, x, c) != 100 {
					t.Fatalf("Criminisi uniform (%d,%d,%d) = %d, want 100", y, x, c, out.At(y, x, c))
				}
			}
		}
	}
}

func TestCriminisiSingleRegionExact(t *testing.T) {
	// Two flat regions; a hole entirely inside the left region must fill with the
	// left region's value (best-matching patches are all from the left region).
	img := uniformMat(20, 20, 1, 40)
	for y := 0; y < 20; y++ {
		for x := 12; x < 20; x++ {
			img.Set(y, x, 0, 200)
		}
	}
	mask := centerHoleMask(20, 20, 8, 3, 3, 3) // inside the left (value 40) region
	out := InpaintCriminisi(img, mask, CriminisiOptions{PatchRadius: 3})
	for y := 8; y < 11; y++ {
		for x := 3; x < 6; x++ {
			if out.At(y, x, 0) != 40 {
				t.Fatalf("Criminisi region fill (%d,%d) = %d, want 40", y, x, out.At(y, x, 0))
			}
		}
	}
}

func BenchmarkCriminisi(b *testing.B) {
	img := uniformMat(24, 24, 3, 100)
	// add some texture so the exemplar search does real work
	for y := 0; y < 24; y++ {
		for x := 0; x < 24; x++ {
			if (x/2+y/2)%2 == 0 {
				img.Set(y, x, 0, 130)
			}
		}
	}
	mask := centerHoleMask(24, 24, 9, 9, 6, 6)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = InpaintCriminisi(img, mask, CriminisiOptions{PatchRadius: 4, SearchRadius: 8})
	}
}
