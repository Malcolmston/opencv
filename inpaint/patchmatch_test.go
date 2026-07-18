package inpaint

import "testing"

func TestPatchMatchNNFUniformZeroDistance(t *testing.T) {
	// On a uniform image every patch matches every other exactly (SSD 0).
	img := uniformMat(10, 10, 3, 120)
	nnf := PatchMatchNNF(img, img, 2, 4)
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			if nnf.Distance(y, x) != 0 {
				t.Fatalf("NNF distance (%d,%d) = %v, want 0", y, x, nnf.Distance(y, x))
			}
		}
	}
}

func TestPatchMatchNNFReconstructUniform(t *testing.T) {
	img := uniformMat(10, 10, 1, 55)
	nnf := PatchMatchNNF(img, img, 2, 4)
	rec := nnf.Reconstruct(img)
	for i, v := range rec.Data {
		if v != 55 {
			t.Fatalf("reconstruct sample %d = %d, want 55", i, v)
		}
	}
}

func TestPatchMatchDeterministic(t *testing.T) {
	img := rampMat(12, 12, 4, 2, 20)
	a := PatchMatchNNF(img, img, 2, 5)
	b := PatchMatchNNF(img, img, 2, 5)
	for i := range a.dist {
		if a.dist[i] != b.dist[i] || a.offX[i] != b.offX[i] || a.offY[i] != b.offY[i] {
			t.Fatalf("PatchMatch not deterministic at index %d", i)
		}
	}
}

func TestInpaintPatchMatchUniformExact(t *testing.T) {
	img := uniformMat(16, 16, 3, 100)
	mask := centerHoleMask(16, 16, 6, 6, 4, 4)
	out := InpaintPatchMatch(img, mask, DefaultPatchMatchOptions())
	for y := 6; y < 10; y++ {
		for x := 6; x < 10; x++ {
			for c := 0; c < 3; c++ {
				if out.At(y, x, c) != 100 {
					t.Fatalf("PatchMatch uniform (%d,%d,%d) = %d, want 100", y, x, c, out.At(y, x, c))
				}
			}
		}
	}
}
