package saliency2

import (
	"math"
	"math/cmplx"
	"testing"
)

// TestFFT1DKnownAnswer checks the radix-2 FFT against hand-computed spectra.
func TestFFT1DKnownAnswer(t *testing.T) {
	// DC input: [1,1,1,1] transforms to [4,0,0,0].
	a := []complex128{1, 1, 1, 1}
	saliency2FFT1D(a, false)
	want := []complex128{4, 0, 0, 0}
	for i := range a {
		if cmplx.Abs(a[i]-want[i]) > 1e-9 {
			t.Fatalf("FFT of DC: bin %d = %v, want %v", i, a[i], want[i])
		}
	}

	// Impulse at index 0: [1,0,0,0] transforms to all ones.
	b := []complex128{1, 0, 0, 0}
	saliency2FFT1D(b, false)
	for i := range b {
		if cmplx.Abs(b[i]-1) > 1e-9 {
			t.Fatalf("FFT of impulse: bin %d = %v, want 1", i, b[i])
		}
	}
}

// TestFFTRoundTrip verifies forward then inverse FFT recovers the input.
func TestFFTRoundTrip(t *testing.T) {
	const h, w = 8, 8
	field := make([]complex128, h*w)
	orig := make([]complex128, h*w)
	for i := range field {
		v := complex(float64((i*7+3)%17), float64((i*5+1)%13))
		field[i] = v
		orig[i] = v
	}
	saliency2FFT2D(field, h, w, false)
	saliency2FFT2D(field, h, w, true)
	for i := range field {
		if cmplx.Abs(field[i]-orig[i]) > 1e-9 {
			t.Fatalf("round trip index %d = %v, want %v", i, field[i], orig[i])
		}
	}
}

// TestBoxMeanUniform checks that the integral-image box mean of a constant grid
// equals that constant everywhere, regardless of radius or clamping.
func TestBoxMeanUniform(t *testing.T) {
	m := NewSaliencyMap(6, 7)
	for i := range m.Data {
		m.Data[i] = 5
	}
	ii := saliency2Integral(m)
	for _, r := range []int{0, 1, 2, 5} {
		for y := 0; y < m.Rows; y++ {
			for x := 0; x < m.Cols; x++ {
				got := saliency2BoxMean(ii, m.Rows, m.Cols, y, x, r)
				if math.Abs(got-5) > 1e-9 {
					t.Fatalf("box mean r=%d at (%d,%d) = %v, want 5", r, x, y, got)
				}
			}
		}
	}
}

// TestBoxMeanKnown checks a hand-computed 3x3 window mean.
func TestBoxMeanKnown(t *testing.T) {
	m := NewSaliencyMap(3, 3)
	for i := range m.Data {
		m.Data[i] = float64(i) // 0..8
	}
	ii := saliency2Integral(m)
	// radius-1 window centred at (1,1) covers all nine values 0..8, mean 4.
	got := saliency2BoxMean(ii, 3, 3, 1, 1, 1)
	if math.Abs(got-4) > 1e-9 {
		t.Fatalf("centre box mean = %v, want 4", got)
	}
}

// TestResizeConstant checks that resampling a constant map preserves the value.
func TestResizeConstant(t *testing.T) {
	m := NewSaliencyMap(4, 4)
	for i := range m.Data {
		m.Data[i] = 3.5
	}
	up := saliency2ResizeMap(m, 9, 11)
	for _, v := range up.Data {
		if math.Abs(v-3.5) > 1e-9 {
			t.Fatalf("resized constant = %v, want 3.5", v)
		}
	}
}

// TestGaussianBlurPreservesMean checks that separable Gaussian blur of a
// constant field is exact (kernel normalised to unit sum).
func TestGaussianBlurPreservesConstant(t *testing.T) {
	m := NewSaliencyMap(5, 5)
	for i := range m.Data {
		m.Data[i] = 10
	}
	b := saliency2GaussianBlurMap(m, 5, 1.5)
	for _, v := range b.Data {
		if math.Abs(v-10) > 1e-9 {
			t.Fatalf("blurred constant = %v, want 10", v)
		}
	}
}
