package inpaint

import (
	"image"
	"testing"
)

func TestSolvePoissonHarmonicUniform(t *testing.T) {
	// Zero guidance + uniform boundary => the uniform value everywhere.
	boundary := uniformMat(7, 7, 1, 50)
	region := centerHoleMask(7, 7, 2, 2, 3, 3)
	guidance := NewFloatImage(7, 7, 1) // all zero
	out := SolvePoisson(guidance, boundary, region, 0)
	for y := 2; y < 5; y++ {
		for x := 2; x < 5; x++ {
			if out.At(y, x, 0) != 50 {
				t.Fatalf("SolvePoisson uniform (%d,%d) = %d, want 50", y, x, out.At(y, x, 0))
			}
		}
	}
}

func TestSeamlessCloneUniformTakesDestination(t *testing.T) {
	// Cloning a flat source (zero gradient) with NormalClone retargets the
	// region to the destination's value at the seam: it becomes the dst value.
	src := uniformMat(9, 9, 3, 50)
	dst := uniformMat(20, 20, 3, 200)
	mask := centerHoleMask(9, 9, 2, 2, 5, 5)
	out := SeamlessClone(src, dst, mask, image.Pt(10, 10), NormalClone)
	if out.At(10, 10, 0) != 200 {
		t.Fatalf("seamless clone centre = %d, want 200", out.At(10, 10, 0))
	}
	// A destination pixel far from the region is unchanged.
	if out.At(0, 0, 0) != 200 {
		t.Fatalf("far destination pixel changed to %d", out.At(0, 0, 0))
	}
}

func TestPoissonBlendZeroGuidance(t *testing.T) {
	dst := uniformMat(8, 8, 1, 90)
	region := centerHoleMask(8, 8, 2, 2, 4, 4)
	gx := NewFloatImage(8, 8, 1)
	gy := NewFloatImage(8, 8, 1)
	out := PoissonBlend(dst, gx, gy, region, 0)
	for y := 2; y < 6; y++ {
		for x := 2; x < 6; x++ {
			if out.At(y, x, 0) != 90 {
				t.Fatalf("PoissonBlend (%d,%d) = %d, want 90", y, x, out.At(y, x, 0))
			}
		}
	}
}
