package freqdomain

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

const tol = 1e-9

func floatMatFrom(rows, cols int, vals ...float64) *cv.FloatMat {
	f := cv.NewFloatMat(rows, cols)
	copy(f.Data, vals)
	return f
}

func approx(a, b, eps float64) bool { return math.Abs(a-b) <= eps }

// deterministicImage builds a reproducible float image using a fixed formula.
func deterministicImage(rows, cols int) *cv.FloatMat {
	f := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			f.Data[y*cols+x] = float64((y*7+x*13)%17) + 0.5*float64((x*x+y)%5)
		}
	}
	return f
}

func TestFFT1DConstant(t *testing.T) {
	re := []float64{1, 1, 1, 1}
	im := []float64{0, 0, 0, 0}
	or, oi := FFT1D(re, im)
	if !approx(or[0], 4, tol) {
		t.Fatalf("DC = %v, want 4", or[0])
	}
	for k := 1; k < 4; k++ {
		if !approx(or[k], 0, tol) || !approx(oi[k], 0, tol) {
			t.Fatalf("bin %d = %v+%vi, want 0", k, or[k], oi[k])
		}
	}
}

func TestFFT1DImpulse(t *testing.T) {
	// FFT of a unit impulse is flat (all ones).
	re := []float64{1, 0, 0, 0, 0, 0, 0, 0}
	im := make([]float64, 8)
	or, oi := FFT1D(re, im)
	for k := 0; k < 8; k++ {
		if !approx(or[k], 1, tol) || !approx(oi[k], 0, tol) {
			t.Fatalf("bin %d = %v+%vi, want 1", k, or[k], oi[k])
		}
	}
}

func TestFFT1DRoundTrip(t *testing.T) {
	// Non-power-of-two length exercises the direct-DFT fallback.
	re := []float64{3, 1, 4, 1, 5, 9}
	im := []float64{0, 0, 0, 0, 0, 0}
	fr, fi := FFT1D(re, im)
	br, bi := IFFT1D(fr, fi)
	for i := range re {
		if !approx(br[i], re[i], 1e-9) || !approx(bi[i], 0, 1e-9) {
			t.Fatalf("round trip idx %d got %v+%vi want %v", i, br[i], bi[i], re[i])
		}
	}
}

func TestDFT2DKnown(t *testing.T) {
	f := floatMatFrom(2, 2, 1, 2, 3, 4)
	s := DFT2D(f)
	want := []float64{10, -2, -4, 0}
	for i, w := range want {
		if !approx(s.Re[i], w, tol) || !approx(s.Im[i], 0, tol) {
			t.Fatalf("bin %d = %v+%vi, want %v", i, s.Re[i], s.Im[i], w)
		}
	}
}

func TestFFT2DMatchesDFT2D(t *testing.T) {
	f := deterministicImage(8, 8)
	a := FFT2D(f)
	b := DFT2D(f)
	for i := range a.Re {
		if !approx(a.Re[i], b.Re[i], 1e-7) || !approx(a.Im[i], b.Im[i], 1e-7) {
			t.Fatalf("FFT/DFT mismatch at %d: %v+%vi vs %v+%vi", i, a.Re[i], a.Im[i], b.Re[i], b.Im[i])
		}
	}
}

func TestFFT2DRoundTrip(t *testing.T) {
	for _, sz := range [][2]int{{8, 8}, {6, 10}, {5, 5}} {
		f := deterministicImage(sz[0], sz[1])
		rec := IFFT2D(FFT2D(f))
		for i := range f.Data {
			if !approx(rec.Data[i], f.Data[i], 1e-7) {
				t.Fatalf("size %v round trip idx %d got %v want %v", sz, i, rec.Data[i], f.Data[i])
			}
		}
	}
}

func TestFFTShiftEven(t *testing.T) {
	// 4x4: fftshift swaps diagonal quadrants; DC (index 0) moves to (2,2).
	f := cv.NewFloatMat(4, 4)
	f.Data[0] = 1 // DC at corner
	s := FFTShift(f)
	if !approx(s.Data[2*4+2], 1, tol) {
		t.Fatalf("DC did not move to centre; got %v", s.Data[2*4+2])
	}
}

func TestFFTShiftInverseOdd(t *testing.T) {
	// For odd dimensions IFFTShift must exactly invert FFTShift.
	f := deterministicImage(5, 7)
	back := IFFTShift(FFTShift(f))
	for i := range f.Data {
		if !approx(back.Data[i], f.Data[i], tol) {
			t.Fatalf("ifftshift(fftshift) mismatch at %d: %v vs %v", i, back.Data[i], f.Data[i])
		}
	}
}

func TestGaussianLowPassShape(t *testing.T) {
	h := GaussianLowPass(32, 32, 5)
	center := h.Data[(32/2)*32+(32/2)]
	if !approx(center, 1, tol) {
		t.Fatalf("Gaussian LP centre = %v, want 1", center)
	}
	corner := h.Data[0]
	if corner >= center {
		t.Fatalf("Gaussian LP corner %v should be below centre %v", corner, center)
	}
	if corner < 0 || corner > 1 {
		t.Fatalf("Gaussian LP corner out of [0,1]: %v", corner)
	}
}

func TestGaussianHighPassComplement(t *testing.T) {
	lp := GaussianLowPass(16, 16, 4)
	hp := GaussianHighPass(16, 16, 4)
	for i := range lp.Data {
		if !approx(lp.Data[i]+hp.Data[i], 1, tol) {
			t.Fatalf("LP+HP != 1 at %d: %v", i, lp.Data[i]+hp.Data[i])
		}
	}
}

func TestButterworthHighPassCenterZero(t *testing.T) {
	h := ButterworthHighPass(16, 16, 4, 2)
	c := h.Data[(16/2)*16+(16/2)]
	if !approx(c, 0, tol) {
		t.Fatalf("Butterworth HP centre = %v, want 0", c)
	}
}

func TestIdealLowPassCount(t *testing.T) {
	h := IdealLowPass(16, 16, 3)
	var passed int
	for _, v := range h.Data {
		if v == 1 {
			passed++
		} else if v != 0 {
			t.Fatalf("ideal LP must be 0/1, got %v", v)
		}
	}
	if passed == 0 || passed == len(h.Data) {
		t.Fatalf("ideal LP passed count implausible: %d", passed)
	}
}

func TestApplyLowPassPreservesDC(t *testing.T) {
	// A low-pass keeps the DC term, so the mean brightness is preserved.
	f := deterministicImage(16, 16)
	var meanIn float64
	for _, v := range f.Data {
		meanIn += v
	}
	meanIn /= float64(len(f.Data))
	out := ApplyFilter(f, GaussianLowPass(16, 16, 4))
	var meanOut float64
	for _, v := range out.Data {
		meanOut += v
	}
	meanOut /= float64(len(out.Data))
	if !approx(meanIn, meanOut, 1e-6) {
		t.Fatalf("low-pass changed mean: in %v out %v", meanIn, meanOut)
	}
}

func TestApplyHighPassRemovesDC(t *testing.T) {
	f := deterministicImage(16, 16)
	out := ApplyFilter(f, GaussianHighPass(16, 16, 4))
	var mean float64
	for _, v := range out.Data {
		mean += v
	}
	mean /= float64(len(out.Data))
	if !approx(mean, 0, 1e-6) {
		t.Fatalf("high-pass mean should be ~0, got %v", mean)
	}
}

func TestConvolveDeltaIdentity(t *testing.T) {
	f := deterministicImage(8, 8)
	delta := cv.NewFloatMat(3, 3)
	delta.Data[1*3+1] = 1 // centre impulse
	out := ConvolveFFT(f, delta)
	for i := range f.Data {
		if !approx(out.Data[i], f.Data[i], 1e-7) {
			t.Fatalf("delta convolution changed pixel %d: %v vs %v", i, out.Data[i], f.Data[i])
		}
	}
}

func TestConvolveConstantPreserved(t *testing.T) {
	// A normalised kernel leaves a constant image unchanged.
	f := cv.NewFloatMat(8, 8)
	for i := range f.Data {
		f.Data[i] = 7
	}
	out := ConvolveFFT(f, GaussianPSF(5, 1.2))
	for i := range out.Data {
		if !approx(out.Data[i], 7, 1e-7) {
			t.Fatalf("constant not preserved at %d: %v", i, out.Data[i])
		}
	}
}

func TestGaussianPSFSumsToOne(t *testing.T) {
	psf := GaussianPSF(7, 1.5)
	var sum float64
	for _, v := range psf.Data {
		sum += v
	}
	if !approx(sum, 1, tol) {
		t.Fatalf("PSF sum = %v, want 1", sum)
	}
}

func TestWienerRecoversBlur(t *testing.T) {
	// Blur then deconvolve with nsr=0 recovers the original exactly, because
	// circular convolution and the pseudo-inverse are exact inverses.
	f := deterministicImage(16, 16)
	psf := GaussianPSF(5, 1.3)
	blurred := ConvolveFFT(f, psf)
	rec := WienerDeconvolution(blurred, psf, 0)
	for i := range f.Data {
		if !approx(rec.Data[i], f.Data[i], 1e-4) {
			t.Fatalf("Wiener recovery pixel %d: %v vs %v", i, rec.Data[i], f.Data[i])
		}
	}
}

func TestInverseFilterRecoversBlur(t *testing.T) {
	f := deterministicImage(16, 16)
	psf := GaussianPSF(5, 1.1)
	blurred := ConvolveFFT(f, psf)
	rec := InverseFilter(blurred, psf, 1e-6)
	for i := range f.Data {
		if !approx(rec.Data[i], f.Data[i], 1e-3) {
			t.Fatalf("inverse filter recovery pixel %d: %v vs %v", i, rec.Data[i], f.Data[i])
		}
	}
}

func TestPhaseCorrelationKnownShift(t *testing.T) {
	rows, cols := 16, 16
	a := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			a.Data[y*cols+x] = float64((x*3 + y*5) % 11)
		}
	}
	wantDX, wantDY := 3, -2
	b := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			sx := ((x-wantDX)%cols + cols) % cols
			sy := ((y-wantDY)%rows + rows) % rows
			b.Data[y*cols+x] = a.Data[sy*cols+sx]
		}
	}
	dx, dy, resp := PhaseCorrelation(a, b)
	if dx != wantDX || dy != wantDY {
		t.Fatalf("phase corr shift = (%d,%d), want (%d,%d)", dx, dy, wantDX, wantDY)
	}
	if resp <= 0 {
		t.Fatalf("response should be positive, got %v", resp)
	}
	rx, ry := RegisterTranslation(a, b)
	if rx != wantDX || ry != wantDY {
		t.Fatalf("RegisterTranslation = (%d,%d), want (%d,%d)", rx, ry, wantDX, wantDY)
	}
}

func TestNotchRejectSuppressesCenter(t *testing.T) {
	h := NotchReject(32, 32, []Notch{{U: 8, V: 8, D0: 3, Order: 2}})
	cy, cx := 32/2, 32/2
	atNotch := h.Data[(cy+8)*32+(cx+8)]
	if atNotch > 0.1 {
		t.Fatalf("notch centre not suppressed: %v", atNotch)
	}
	far := h.Data[cy*32+cx] // DC, far from notch
	if far < 0.9 {
		t.Fatalf("DC should pass through notch reject: %v", far)
	}
}

func TestMatFloatRoundTrip(t *testing.T) {
	m := cv.NewMat(4, 4, 1)
	for i := range m.Data {
		m.Data[i] = uint8(i * 10)
	}
	f := MatToFloat(m)
	back := FloatToMat(f)
	for i := range m.Data {
		if back.Data[i] != m.Data[i] {
			t.Fatalf("round trip idx %d: %d vs %d", i, back.Data[i], m.Data[i])
		}
	}
}

func TestSpectrumConjugateAndPower(t *testing.T) {
	s := NewSpectrum(2, 2)
	s.Set(0, 0, 3, 4)
	if p := s.PowerSpectrum().Data[0]; !approx(p, 25, tol) {
		t.Fatalf("power = %v, want 25", p)
	}
	if m := s.Magnitude().Data[0]; !approx(m, 5, tol) {
		t.Fatalf("magnitude = %v, want 5", m)
	}
	c := s.Conjugate()
	if _, im := c.At(0, 0); !approx(im, -4, tol) {
		t.Fatalf("conjugate imag = %v, want -4", im)
	}
}

func TestHomomorphicRuns(t *testing.T) {
	m := cv.NewMat(16, 16, 1)
	for i := range m.Data {
		m.Data[i] = uint8((i * 3) % 256)
	}
	out := HomomorphicFilter(m, 0.5, 2.0, 4, 1)
	if out.Rows != 16 || out.Cols != 16 || out.Channels != 1 {
		t.Fatalf("unexpected output shape %dx%dx%d", out.Rows, out.Cols, out.Channels)
	}
}

func TestPowerOfTwoHelpers(t *testing.T) {
	if !IsPowerOfTwo(16) || IsPowerOfTwo(12) || IsPowerOfTwo(0) {
		t.Fatal("IsPowerOfTwo wrong")
	}
	if NextPowerOfTwo(12) != 16 || NextPowerOfTwo(16) != 16 || NextPowerOfTwo(1) != 1 {
		t.Fatalf("NextPowerOfTwo wrong: %d %d %d", NextPowerOfTwo(12), NextPowerOfTwo(16), NextPowerOfTwo(1))
	}
}

func BenchmarkFFT2D256(b *testing.B) {
	f := deterministicImage(256, 256)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FFT2D(f)
	}
}
