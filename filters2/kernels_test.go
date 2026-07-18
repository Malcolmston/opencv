package filters2

import (
	"math"
	"testing"
)

func sum1D(k []float64) float64 {
	var s float64
	for _, v := range k {
		s += v
	}
	return s
}

func sum2D(k [][]float64) float64 {
	var s float64
	for _, r := range k {
		for _, v := range r {
			s += v
		}
	}
	return s
}

func TestGaussianKernel1DNormalised(t *testing.T) {
	k := GaussianKernel1D(3, 1.0)
	if len(k) != 7 {
		t.Fatalf("length = %d, want 7", len(k))
	}
	if s := sum1D(k); math.Abs(s-1) > 1e-12 {
		t.Errorf("sum = %g, want 1", s)
	}
	// Symmetric and peaked at the centre.
	if k[3] <= k[2] || k[2] <= k[1] {
		t.Errorf("kernel is not centre-peaked: %v", k)
	}
	if math.Abs(k[0]-k[6]) > 1e-15 || math.Abs(k[2]-k[4]) > 1e-15 {
		t.Errorf("kernel not symmetric: %v", k)
	}
}

func TestGaussianKernel2DNormalised(t *testing.T) {
	k := GaussianKernel2D(5, 1.2)
	if s := sum2D(k); math.Abs(s-1) > 1e-12 {
		t.Errorf("sum = %g, want 1", s)
	}
	if k[2][2] != maxOf2D(k) {
		t.Errorf("peak is not at the centre")
	}
}

func maxOf2D(k [][]float64) float64 {
	m := k[0][0]
	for _, r := range k {
		for _, v := range r {
			if v > m {
				m = v
			}
		}
	}
	return m
}

func TestLoGKernelZeroSum(t *testing.T) {
	k := LaplacianOfGaussianKernel(9, 1.4)
	if s := sum2D(k); math.Abs(s) > 1e-9 {
		t.Errorf("LoG sum = %g, want ~0", s)
	}
	// Centre must be negative (mexican-hat trough).
	if k[4][4] >= 0 {
		t.Errorf("LoG centre = %g, want negative", k[4][4])
	}
}

func TestDoGKernelZeroSum(t *testing.T) {
	k := DifferenceOfGaussiansKernel(9, 1.0, 2.0)
	if s := sum2D(k); math.Abs(s) > 1e-12 {
		t.Errorf("DoG sum = %g, want ~0", s)
	}
	// Centre positive for sigma1 < sigma2.
	if k[4][4] <= 0 {
		t.Errorf("DoG centre = %g, want positive", k[4][4])
	}
}

func TestGaussianDerivativeKernelSums(t *testing.T) {
	// A derivative kernel must have (approximately) zero sum.
	dx := GaussianDerivativeKernel(11, 1.5, 1, 0)
	if s := sum2D(dx); math.Abs(s) > 1e-9 {
		t.Errorf("dGx sum = %g, want ~0", s)
	}
	// The x-derivative is antisymmetric across the vertical axis.
	r := 11 / 2
	if math.Abs(dx[r][r-1]+dx[r][r+1]) > 1e-12 {
		t.Errorf("dGx not antisymmetric in x")
	}
	// Second derivative in x is symmetric in x and has a negative centre.
	dxx := GaussianDerivativeKernel(11, 1.5, 2, 0)
	if dxx[r][r] >= 0 {
		t.Errorf("dGxx centre = %g, want negative", dxx[r][r])
	}
	if math.Abs(dxx[r][r-1]-dxx[r][r+1]) > 1e-12 {
		t.Errorf("dGxx not symmetric in x")
	}
}

func TestGaborKernelOrthogonalPhase(t *testing.T) {
	p := GaborParams{Sigma: 3, Theta: 0, Lambda: 6, Gamma: 1, Psi: 0}
	even := GaborKernel(11, p)
	r := 11 / 2
	// With psi=0 the even kernel peaks at the centre and is symmetric in x.
	if math.Abs(even[r][r-1]-even[r][r+1]) > 1e-12 {
		t.Errorf("even Gabor not symmetric in x")
	}
	if even[r][r] <= 0 {
		t.Errorf("even Gabor centre = %g, want positive", even[r][r])
	}
}
