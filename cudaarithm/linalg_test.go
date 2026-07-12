package cudaarithm

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func mat1(rows, cols int, vals ...uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	copy(m.Data, vals)
	return m
}

func TestGemm(t *testing.T) {
	// A = [[1,2],[0,1]], B = [[1,1],[0,1]] -> A*B = [[1,3],[0,1]].
	a := mat1(2, 2, 1, 2, 0, 1)
	b := mat1(2, 2, 1, 1, 0, 1)
	got := Gemm(NewGpuMat(a), NewGpuMat(b), 1, nil, 0).Download()
	want := []uint8{1, 3, 0, 1}
	for i, w := range want {
		if got.Data[i] != w {
			t.Errorf("Gemm[%d] = %d, want %d", i, got.Data[i], w)
		}
	}

	// With alpha=1, C added: dst = A*B + 2*C, C = [[1,1],[1,1]] -> [[3,5],[2,3]].
	c := mat1(2, 2, 1, 1, 1, 1)
	got2 := Gemm(NewGpuMat(a), NewGpuMat(b), 1, NewGpuMat(c), 2).Download()
	want2 := []uint8{3, 5, 2, 3}
	for i, w := range want2 {
		if got2.Data[i] != w {
			t.Errorf("Gemm+C[%d] = %d, want %d", i, got2.Data[i], w)
		}
	}
}

func TestGemmDimMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on inner-dimension mismatch")
		}
	}()
	Gemm(NewGpuMat(mat1(2, 3, 1, 1, 1, 1, 1, 1)), NewGpuMat(mat1(2, 2, 1, 1, 1, 1)), 1, nil, 0)
}

func TestDFTRoundTrip(t *testing.T) {
	src := cv.NewMat(3, 4, 1)
	for i := range src.Data {
		src.Data[i] = uint8((i*13 + 7) % 200)
	}
	spec := DFT(NewGpuMat(src))
	back := IDFT(spec).Download()
	if !sameData(back, src) {
		t.Error("IDFT(DFT(x)) should reproduce x")
	}
}

func TestDFTConstantImageDCTerm(t *testing.T) {
	// A constant image has all energy in the DC (0,0) term equal to the sum.
	src := constMat(2, 2, 10)
	spec := DFT(NewGpuMat(src))
	re, im := spec.At(0, 0)
	if math.Abs(re-40) > 1e-9 || math.Abs(im) > 1e-9 {
		t.Errorf("DC term = (%v,%v), want (40,0)", re, im)
	}
	// All other terms should be ~0.
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			if x == 0 && y == 0 {
				continue
			}
			r, ii := spec.At(y, x)
			if math.Abs(r) > 1e-9 || math.Abs(ii) > 1e-9 {
				t.Errorf("term (%d,%d) = (%v,%v), want ~0", y, x, r, ii)
			}
		}
	}
}

func TestMulSpectrumsConvolutionTheorem(t *testing.T) {
	// Circular convolution of a with a shift delta is a circular shift.
	a := mat1(1, 4, 1, 2, 3, 4)
	shift := mat1(1, 4, 0, 1, 0, 0) // delta at index 1
	sa := DFT(NewGpuMat(a))
	sb := DFT(NewGpuMat(shift))
	prod := MulSpectrums(sa, sb, false)
	conv := IDFTComplex(prod)
	// Expected circular convolution: result[n] = a[(n-1) mod 4].
	want := []float64{4, 1, 2, 3}
	for i, w := range want {
		if math.Abs(conv.Re[i]-w) > 1e-6 {
			t.Errorf("conv[%d] = %v, want %v", i, conv.Re[i], w)
		}
	}
}

func TestMulSpectrumsConjCorrelation(t *testing.T) {
	a := mat1(1, 4, 1, 2, 3, 4)
	sa := DFT(NewGpuMat(a))
	// Correlating a signal with itself (conjB) peaks at lag 0 with the energy.
	corr := IDFTComplex(MulSpectrums(sa, sa, true))
	energy := 1.0 + 4 + 9 + 16
	if math.Abs(corr.Re[0]-energy) > 1e-6 {
		t.Errorf("autocorrelation lag0 = %v, want %v", corr.Re[0], energy)
	}
}
