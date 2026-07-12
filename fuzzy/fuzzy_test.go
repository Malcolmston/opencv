package fuzzy

import (
	"math"
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- helpers ---------------------------------------------------------------

func mean(m *cv.Mat) float64 {
	var s float64
	for _, v := range m.Data {
		s += float64(v)
	}
	return s / float64(len(m.Data))
}

func variance(m *cv.Mat) float64 {
	mu := mean(m)
	var s float64
	for _, v := range m.Data {
		d := float64(v) - mu
		s += d * d
	}
	return s / float64(len(m.Data))
}

// mae is the mean absolute per-sample error between a and b.
func mae(a, b *cv.Mat) float64 {
	if len(a.Data) != len(b.Data) {
		panic("size mismatch")
	}
	var s float64
	for i := range a.Data {
		s += math.Abs(float64(int(a.Data[i]) - int(b.Data[i])))
	}
	return s / float64(len(a.Data))
}

// maskedMAE averages absolute error only over pixels where mask is non-zero.
func maskedMAE(ref, got, mask *cv.Mat) float64 {
	var s float64
	var n int
	for p := 0; p < ref.Rows*ref.Cols; p++ {
		if mask.Data[p*mask.Channels] == 0 {
			continue
		}
		for c := 0; c < ref.Channels; c++ {
			s += math.Abs(float64(int(ref.Data[p*ref.Channels+c]) - int(got.Data[p*got.Channels+c])))
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return s / float64(n)
}

func ramp(rows, cols, ch int) *cv.Mat {
	m := cv.NewMat(rows, cols, ch)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := uint8(30 + (150*(x+y))/(rows+cols))
			for c := 0; c < ch; c++ {
				m.Data[(y*cols+x)*ch+c] = uint8(int(v) + c*10)
			}
		}
	}
	return m
}

func expectPanic(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("%s: expected panic, got none", name)
		}
	}()
	fn()
}

// --- kernel / basis --------------------------------------------------------

func TestCreateKernelShapeAndPeak(t *testing.T) {
	for _, fn := range []BasisFunction{LinearBasis, SinusBasis} {
		for _, r := range []int{1, 2, 3, 5} {
			k := CreateKernel(fn, r)
			n := 2*r + 1
			if k.Rows != n || k.Cols != n {
				t.Fatalf("%v r=%d: size %dx%d want %dx%d", fn, r, k.Rows, k.Cols, n, n)
			}
			if got := k.At(r, r); math.Abs(got-1) > 1e-12 {
				t.Errorf("%v r=%d: centre %.6f want 1", fn, r, got)
			}
			// Endpoints are zero.
			if got := k.At(0, 0); math.Abs(got) > 1e-12 {
				t.Errorf("%v r=%d: corner %.6f want 0", fn, r, got)
			}
			// Symmetry.
			for y := 0; y < n; y++ {
				for x := 0; x < n; x++ {
					if math.Abs(k.At(y, x)-k.At(n-1-y, n-1-x)) > 1e-12 {
						t.Fatalf("%v r=%d: not symmetric at (%d,%d)", fn, r, y, x)
					}
				}
			}
		}
	}
}

// TestPartitionOfUnity verifies that the 1-D basis satisfies the Ruspini
// condition: two neighbouring bases spaced radius apart sum to 1 everywhere in
// their shared interval.
func TestPartitionOfUnity(t *testing.T) {
	for _, fn := range []BasisFunction{LinearBasis, SinusBasis} {
		for _, r := range []int{2, 4, 7} {
			v := basisVector(fn, r)
			// v[i] is centred at radius; a neighbour centred at 2*radius overlaps
			// for offsets in [radius, 2*radius]. For offset i (0..radius) the two
			// contributions are v[radius+i] and v[i].
			for i := 0; i <= r; i++ {
				sum := v[r+i] + v[i]
				if math.Abs(sum-1) > 1e-9 {
					t.Errorf("%v r=%d offset=%d: partition sum %.6f want 1", fn, r, i, sum)
				}
			}
		}
	}
}

func TestCreateKernelPanics(t *testing.T) {
	expectPanic(t, "radius 0", func() { CreateKernel(LinearBasis, 0) })
	expectPanic(t, "bad function", func() { CreateKernel(BasisFunction(99), 2) })
	expectPanic(t, "kernelRadius even", func() { kernelRadius(cv.NewFloatMat(4, 4)) })
	expectPanic(t, "kernelRadius nonsquare", func() { kernelRadius(cv.NewFloatMat(3, 5)) })
}

func TestBasisFunctionString(t *testing.T) {
	cases := map[BasisFunction]string{LinearBasis: "LinearBasis", SinusBasis: "SinusBasis", BasisFunction(7): "BasisFunction(7)"}
	for f, want := range cases {
		if got := f.String(); got != want {
			t.Errorf("String(%d)=%q want %q", int(f), got, want)
		}
	}
}

// --- transform round trip / smoothing --------------------------------------

func TestFilterConstantIsUnchanged(t *testing.T) {
	for _, ch := range []int{1, 3} {
		img := cv.NewMat(12, 15, ch)
		img.SetTo(137)
		for _, fn := range []BasisFunction{LinearBasis, SinusBasis} {
			out := Filter(img, fn, 3)
			if out.Rows != img.Rows || out.Cols != img.Cols || out.Channels != ch {
				t.Fatalf("shape changed")
			}
			for i, v := range out.Data {
				if v != 137 {
					t.Fatalf("ch=%d %v: pixel %d = %d, want 137", ch, fn, i, v)
				}
			}
		}
	}
}

func TestRoundTripRecoversGradient(t *testing.T) {
	img := ramp(40, 50, 1)
	for _, fn := range []BasisFunction{LinearBasis, SinusBasis} {
		out := Filter(img, fn, 2)
		if e := mae(out, img); e > 0.5 {
			t.Errorf("%v: gradient round-trip MAE %.3f exceeds 0.5", fn, e)
		}
	}
}

func TestRoundTripColor(t *testing.T) {
	img := ramp(30, 30, 3)
	out := Filter(img, LinearBasis, 2)
	if out.Channels != 3 {
		t.Fatalf("channels=%d", out.Channels)
	}
	if e := mae(out, img); e > 0.6 {
		t.Errorf("colour gradient round-trip MAE %.3f too high", e)
	}
}

func TestFilterReducesNoisePreservesMean(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	rows, cols := 64, 64
	noisy := cv.NewMat(rows, cols, 1)
	for i := range noisy.Data {
		n := 128 + int(math.Round(rng.NormFloat64()*30))
		if n < 0 {
			n = 0
		} else if n > 255 {
			n = 255
		}
		noisy.Data[i] = uint8(n)
	}
	out := Filter(noisy, LinearBasis, 3)

	if dm := math.Abs(mean(out) - mean(noisy)); dm > 1.0 {
		t.Errorf("mean not preserved: shift %.3f", dm)
	}
	vn, vo := variance(noisy), variance(out)
	if vo >= vn*0.5 {
		t.Errorf("high-frequency variance not reduced: noisy=%.1f filtered=%.1f", vn, vo)
	}
}

// --- components / inverse low-level ----------------------------------------

func TestComponentsAndInverse(t *testing.T) {
	img := ramp(24, 28, 1)
	kernel := CreateKernel(SinusBasis, 3)
	c := FT02DComponents(img, kernel, nil)
	if c.Radius != 3 || c.Channels != 1 || c.Rows != 24 || c.Cols != 28 {
		t.Fatalf("unexpected components meta: %+v", *c)
	}
	if len(c.Data) != c.An*c.Bn*c.Channels {
		t.Fatalf("component length %d != %d", len(c.Data), c.An*c.Bn*c.Channels)
	}
	// At should agree with Data layout.
	if c.At(1, 1, 0) != c.Data[(1*c.An+1)*c.Channels] {
		t.Errorf("At disagrees with Data")
	}
	out := FT02DProcess(img, kernel, nil)
	if out.Rows != img.Rows || out.Cols != img.Cols {
		t.Fatalf("inverse size mismatch")
	}
}

func TestFT02DComponentsMaskExcludesPixels(t *testing.T) {
	// A field that is 200 everywhere except a masked-out block of 0s. With the
	// block excluded, the reconstruction near it should read ~200, not be pulled
	// toward 0.
	rows, cols := 20, 20
	img := cv.NewMat(rows, cols, 1)
	img.SetTo(200)
	mask := cv.NewMat(rows, cols, 1) // validity mask: non-zero == valid
	mask.SetTo(255)
	for y := 8; y < 12; y++ {
		for x := 8; x < 12; x++ {
			img.Data[y*cols+x] = 0
			mask.Data[y*cols+x] = 0
		}
	}
	out := FT02DProcess(img, CreateKernel(LinearBasis, 3), mask)
	if v := out.Data[10*cols+10]; v < 190 {
		t.Errorf("masked pixel reconstructed to %d, expected ~200", v)
	}
}

// --- inpainting ------------------------------------------------------------

func TestInpaintOneStepReconstructsSmallHole(t *testing.T) {
	ref := ramp(50, 50, 1)
	corrupt := ref.Clone()
	mask := cv.NewMat(50, 50, 1)
	for y := 22; y < 27; y++ {
		for x := 22; x < 27; x++ {
			mask.Data[y*50+x] = 255
			corrupt.Data[y*50+x] = 0
		}
	}
	out := Inpaint(corrupt, mask, 4, LinearBasis, OneStep)
	if e := maskedMAE(ref, out, mask); e > 3.0 {
		t.Errorf("one-step inpaint MAE %.3f exceeds 3.0", e)
	}
	// Known pixels are preserved exactly.
	if out.Data[0] != ref.Data[0] {
		t.Errorf("known pixel was modified: %d vs %d", out.Data[0], ref.Data[0])
	}
}

func TestInpaintColorHole(t *testing.T) {
	ref := ramp(40, 40, 3)
	corrupt := ref.Clone()
	mask := cv.NewMat(40, 40, 1)
	for y := 18; y < 22; y++ {
		for x := 18; x < 22; x++ {
			mask.Data[y*40+x] = 255
			for c := 0; c < 3; c++ {
				corrupt.Data[(y*40+x)*3+c] = 0
			}
		}
	}
	out := Inpaint(corrupt, mask, 3, SinusBasis, OneStep)
	if e := maskedMAE(ref, out, mask); e > 4.0 {
		t.Errorf("colour inpaint MAE %.3f too high", e)
	}
}

func TestInpaintIterativeFillsLargeHole(t *testing.T) {
	ref := ramp(60, 60, 1)
	corrupt := ref.Clone()
	mask := cv.NewMat(60, 60, 1)
	for y := 18; y < 34; y++ {
		for x := 18; x < 34; x++ {
			mask.Data[y*60+x] = 255
			corrupt.Data[y*60+x] = 0
		}
	}
	radius := 3

	// One-step cannot reach the centre of a hole wider than the kernel: some
	// pixels remain unfilled.
	one := Inpaint(corrupt, mask, radius, LinearBasis, OneStep)
	unfilled := 0
	for p := 0; p < 60*60; p++ {
		if mask.Data[p] != 0 && one.Data[p] == 0 {
			unfilled++
		}
	}
	if unfilled == 0 {
		t.Errorf("expected one-step to leave the hole centre unfilled")
	}

	// Iterative fills the whole hole and reconstructs it well.
	iter := Inpaint(corrupt, mask, radius, LinearBasis, Iterative)
	for p := 0; p < 60*60; p++ {
		if mask.Data[p] != 0 && iter.Data[p] == 0 && ref.Data[p] != 0 {
			t.Fatalf("iterative left pixel %d unfilled", p)
		}
	}
	if e := maskedMAE(ref, iter, mask); e > 6.0 {
		t.Errorf("iterative inpaint MAE %.3f exceeds 6.0", e)
	}
}

func TestInpaintAlgorithmString(t *testing.T) {
	cases := map[InpaintAlgorithm]string{OneStep: "OneStep", Iterative: "Iterative", InpaintAlgorithm(9): "InpaintAlgorithm(9)"}
	for a, want := range cases {
		if got := a.String(); got != want {
			t.Errorf("String(%d)=%q want %q", int(a), got, want)
		}
	}
}

func TestPanics(t *testing.T) {
	img := cv.NewMat(8, 8, 1)
	mask := cv.NewMat(8, 8, 1)
	small := cv.NewMat(4, 4, 1)
	expectPanic(t, "Filter empty", func() { Filter(&cv.Mat{}, LinearBasis, 2) })
	expectPanic(t, "Components empty", func() { FT02DComponents(&cv.Mat{}, CreateKernel(LinearBasis, 2), nil) })
	expectPanic(t, "Components mask size", func() { FT02DComponents(img, CreateKernel(LinearBasis, 2), small) })
	expectPanic(t, "Inpaint empty", func() { Inpaint(&cv.Mat{}, mask, 2, LinearBasis, OneStep) })
	expectPanic(t, "Inpaint mask size", func() { Inpaint(img, small, 2, LinearBasis, OneStep) })
	expectPanic(t, "Inpaint radius", func() { Inpaint(img, mask, 0, LinearBasis, OneStep) })
}
