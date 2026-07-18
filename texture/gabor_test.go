package texture_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/texture"
)

func TestGaborKernelShape(t *testing.T) {
	p := texture.GaborParams{Sigma: 3, Theta: 0, Lambda: 5, Gamma: 1, Psi: 0}
	k := texture.GaborKernel(9, p)
	if len(k) != 81 {
		t.Fatalf("kernel length = %d, want 81", len(k))
	}
	// At the centre the envelope is 1 and the cosine (psi=0) is 1.
	if !approx(k[9*4+4], 1.0, 1e-12) {
		t.Errorf("kernel centre = %v, want 1", k[9*4+4])
	}
}

func TestGaborKernelPanics(t *testing.T) {
	assertPanic(t, "even size", func() {
		texture.GaborKernel(8, texture.GaborParams{Sigma: 3, Lambda: 5, Gamma: 1})
	})
	assertPanic(t, "sigma<=0", func() {
		texture.GaborKernel(9, texture.GaborParams{Sigma: 0, Lambda: 5, Gamma: 1})
	})
}

func TestGaborOrientationSelectivity(t *testing.T) {
	// Vertical stripes vary along x, so a horizontally-tuned (theta=0) Gabor
	// responds far more strongly than a vertically-tuned one.
	stripe := cv.NewMat(16, 16, 1)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			if x%4 < 2 {
				stripe.Data[y*16+x] = 200
			}
		}
	}
	e0 := texture.GaborEnergy(stripe, 9, texture.GaborParams{Sigma: 3, Theta: 0, Lambda: 4, Gamma: 1, Psi: 0})
	e90 := texture.GaborEnergy(stripe, 9, texture.GaborParams{Sigma: 3, Theta: 1.5707963, Lambda: 4, Gamma: 1, Psi: 0})
	if !(e0 > e90) {
		t.Errorf("expected e0 (%.3f) > e90 (%.3f)", e0, e90)
	}
}

func TestGaborFeaturesLength(t *testing.T) {
	img := fill(12, 12, 100)
	feats := texture.GaborFeatures(img, 9, texture.GaborParams{Sigma: 3, Gamma: 1}, 4, []float64{4, 8})
	if len(feats) != 2*4*2 {
		t.Fatalf("feature length = %d, want %d", len(feats), 2*4*2)
	}
}

func TestGaborFilterBankSize(t *testing.T) {
	bank := texture.GaborFilterBank(9, texture.GaborParams{Sigma: 3, Gamma: 1}, 6, []float64{4, 8, 12})
	if len(bank) != 18 {
		t.Fatalf("bank size = %d, want 18", len(bank))
	}
}

func TestGaborMagnitudeShape(t *testing.T) {
	img := fill(8, 10, 100)
	mag := texture.GaborMagnitude(img, 7, texture.GaborParams{Sigma: 2, Lambda: 4, Gamma: 1})
	if len(mag) != 8 || len(mag[0]) != 10 {
		t.Fatalf("magnitude grid = %dx%d, want 8x10", len(mag), len(mag[0]))
	}
}

func TestGaborFilterMatShape(t *testing.T) {
	img := fill(8, 8, 100)
	out := texture.GaborFilter(img, 7, texture.GaborParams{Sigma: 2, Lambda: 4, Gamma: 1})
	if out.Rows != 8 || out.Cols != 8 || out.Channels != 1 {
		t.Fatalf("filter output shape = %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
}
