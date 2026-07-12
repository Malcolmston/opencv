package fuzzy

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// gradient builds an exact linear ramp v(x,y) = a + bx*x + by*y clamped to bytes,
// per channel with a small channel offset. It is the natural stress test for the
// degree-1 transform, which should reproduce planes almost exactly.
func gradient(rows, cols, ch int, a, bx, by float64) *cv.Mat {
	m := cv.NewMat(rows, cols, ch)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			val := a + bx*float64(x) + by*float64(y)
			for c := 0; c < ch; c++ {
				v := val + float64(c*7)
				if v < 0 {
					v = 0
				} else if v > 255 {
					v = 255
				}
				m.Data[(y*cols+x)*ch+c] = uint8(v + 0.5)
			}
		}
	}
	return m
}

// --- degree-1 transform ----------------------------------------------------

func TestFT12DBeatsFT02DOnGradient(t *testing.T) {
	img := gradient(48, 56, 1, 20, 2.0, 1.3)
	kernel := CreateKernel(LinearBasis, 4)

	d0 := FT02DProcess(img, kernel, nil)
	d1 := FT12DProcess(img, kernel, nil)

	e0 := TransformError(img, d0)
	e1 := TransformError(img, d1)
	if e1.MAE >= e0.MAE {
		t.Errorf("degree-1 MAE %.4f should beat degree-0 MAE %.4f", e1.MAE, e0.MAE)
	}
	// Degree-1 reproduces a linear ramp to near machine precision after rounding.
	if e1.MAE > 0.05 {
		t.Errorf("degree-1 gradient MAE %.4f too high", e1.MAE)
	}
}

func TestFT12DColorGradient(t *testing.T) {
	img := gradient(40, 40, 3, 30, 1.5, 1.1)
	out := FT12DProcess(img, CreateKernel(SinusBasis, 3), nil)
	if out.Channels != 3 {
		t.Fatalf("channels=%d", out.Channels)
	}
	if e := TransformError(img, out); e.MAE > 0.6 {
		t.Errorf("colour degree-1 MAE %.4f too high", e.MAE)
	}
}

func TestFT12DConstantExact(t *testing.T) {
	img := cv.NewMat(20, 24, 1)
	img.SetTo(88)
	out := FT12DProcess(img, CreateKernel(LinearBasis, 3), nil)
	for i, v := range out.Data {
		if v != 88 {
			t.Fatalf("degree-1 constant not preserved at %d: %d", i, v)
		}
	}
}

func TestFT12DPolynomialAndReconstruct(t *testing.T) {
	img := gradient(30, 30, 2, 40, 1.0, -0.7)
	kernel := CreateKernel(LinearBasis, 3)
	comps, c00, c10, c01 := FT12DPolynomial(img, kernel, nil, 1)
	if c00.Rows != comps.Bn || c00.Cols != comps.An {
		t.Fatalf("coeff plane size %dx%d want %dx%d", c00.Rows, c00.Cols, comps.Bn, comps.An)
	}
	// An interior node's x-gradient should track the ramp slope (+1.0/px) and its
	// y-gradient the vertical slope (-0.7/px), regardless of the fuzzy weighting.
	i, o := comps.Bn/2, comps.An/2
	gx := c10.At(i, o)
	gy := c01.At(i, o)
	if math.Abs(gx-1.0) > 0.05 {
		t.Errorf("fitted x-gradient %.4f, want ~1.0", gx)
	}
	if math.Abs(gy-(-0.7)) > 0.05 {
		t.Errorf("fitted y-gradient %.4f, want ~-0.7", gy)
	}
	// Polynomial method agrees with the extracted planes.
	p00, p10, p01 := comps.Polynomial(i, o, 1)
	if p00 != c00.At(i, o) || p10 != gx || p01 != gy {
		t.Errorf("Polynomial disagrees with CoeffPlane")
	}
	// Reconstruct equals FT12DInverse.
	r1 := comps.Reconstruct()
	r2 := FT12DInverse(comps)
	if mae(r1, r2) != 0 {
		t.Errorf("Reconstruct differs from FT12DInverse")
	}
}

func TestFT12DSingularFallback(t *testing.T) {
	// A single valid pixel under each node cannot determine a gradient; the fit
	// must fall back to the degree-0 value without panicking or producing NaNs.
	img := cv.NewMat(10, 10, 1)
	img.SetTo(100)
	mask := cv.NewMat(10, 10, 1)
	mask.Set(5, 5, 0, 255) // exactly one valid pixel
	out := FT12DProcess(img, CreateKernel(LinearBasis, 2), mask)
	for _, v := range out.Data {
		if v != 0 && v != 100 {
			// covered pixels read ~100, uncovered stay 0; never NaN/garbage.
			t.Fatalf("unexpected reconstructed value %d", v)
		}
	}
}

func TestFT12DPanics(t *testing.T) {
	img := cv.NewMat(8, 8, 2)
	small := cv.NewMat(4, 4, 1)
	expectPanic(t, "FT12D empty", func() { FT12DComponents(&cv.Mat{}, CreateKernel(LinearBasis, 2), nil) })
	expectPanic(t, "FT12D mask size", func() { FT12DComponents(img, CreateKernel(LinearBasis, 2), small) })
	expectPanic(t, "FT12DPolynomial channel", func() { FT12DPolynomial(img, CreateKernel(LinearBasis, 2), nil, 9) })
	comps := FT12DComponents(img, CreateKernel(LinearBasis, 2), nil)
	expectPanic(t, "CoeffPlane which", func() { comps.CoeffPlane(5, 0) })
	expectPanic(t, "CoeffPlane channel", func() { comps.CoeffPlane(0, 9) })
}

// --- two-vector / 1-D kernels ---------------------------------------------

func TestCreateKernelVecMatchesCreateKernel(t *testing.T) {
	for _, fn := range []BasisFunction{LinearBasis, SinusBasis} {
		for _, r := range []int{1, 2, 4} {
			v := CreateKernel1D(fn, r)
			if len(v) != 2*r+1 {
				t.Fatalf("1D length %d want %d", len(v), 2*r+1)
			}
			got := CreateKernelVec(v)
			want := CreateKernel(fn, r)
			if got.Rows != want.Rows || got.Cols != want.Cols {
				t.Fatalf("size mismatch")
			}
			for i := range want.Data {
				if math.Abs(got.Data[i]-want.Data[i]) > 1e-15 {
					t.Fatalf("%v r=%d: CreateKernelVec disagrees at %d", fn, r, i)
				}
			}
		}
	}
}

func TestCreateKernelABIsOuterProduct(t *testing.T) {
	a := CreateKernel1D(LinearBasis, 2) // length 5
	b := CreateKernel1D(SinusBasis, 3)  // length 7
	k := CreateKernelAB(a, b)
	if k.Rows != len(b) || k.Cols != len(a) {
		t.Fatalf("kernel size %dx%d want %dx%d", k.Rows, k.Cols, len(b), len(a))
	}
	for y := 0; y < len(b); y++ {
		for x := 0; x < len(a); x++ {
			if math.Abs(k.At(y, x)-b[y]*a[x]) > 1e-15 {
				t.Fatalf("element (%d,%d) not outer product", y, x)
			}
		}
	}
}

func TestCreateKernelABPanics(t *testing.T) {
	good := CreateKernel1D(LinearBasis, 2)
	expectPanic(t, "even a", func() { CreateKernelAB([]float64{1, 2, 3, 4}, good) })
	expectPanic(t, "short b", func() { CreateKernelAB(good, []float64{1}) })
}

// --- fast FL variants ------------------------------------------------------

func TestFLProcessMatchesDense(t *testing.T) {
	img := ramp(37, 41, 1)
	dense := FT02DProcess(img, CreateKernel(LinearBasis, 3), nil)
	fast := FT02DFLProcess(img, LinearBasis, 3)
	if fast.Rows != dense.Rows || fast.Cols != dense.Cols {
		t.Fatalf("shape mismatch")
	}
	// Same math, different accumulation order: agree to within a rounding unit.
	if e := TransformError(dense, fast); e.MaxAbs > 1 {
		t.Errorf("FL vs dense max error %.0f exceeds 1", e.MaxAbs)
	}
}

func TestFLProcessConstantAndColor(t *testing.T) {
	img := cv.NewMat(20, 25, 3)
	img.SetTo(140)
	out := FT02DFLProcess(img, SinusBasis, 3)
	for i, v := range out.Data {
		if v != 140 {
			t.Fatalf("FL constant not preserved at %d: %d", i, v)
		}
	}
}

func TestFLProcessFloat(t *testing.T) {
	img := ramp(30, 30, 1)
	f := FT02DFLProcessFloat(img, LinearBasis, 3)
	if f.Rows != 30 || f.Cols != 30 {
		t.Fatalf("float shape %dx%d", f.Rows, f.Cols)
	}
	// The rounded float result matches the byte FL result exactly.
	b := FT02DFLProcess(img, LinearBasis, 3)
	for p := 0; p < 30*30; p++ {
		if clampByte(f.Data[p]) != b.Data[p] {
			t.Fatalf("float/byte mismatch at %d: %.3f vs %d", p, f.Data[p], b.Data[p])
		}
	}
}

func TestFLPanics(t *testing.T) {
	color := cv.NewMat(8, 8, 3)
	expectPanic(t, "FL empty", func() { FT02DFLProcess(&cv.Mat{}, LinearBasis, 2) })
	expectPanic(t, "FL radius", func() { FT02DFLProcess(color, LinearBasis, 0) })
	expectPanic(t, "FLFloat channels", func() { FT02DFLProcessFloat(color, LinearBasis, 2) })
	expectPanic(t, "FLFloat empty", func() { FT02DFLProcessFloat(&cv.Mat{}, LinearBasis, 2) })
}

// --- single iteration ------------------------------------------------------

func TestFT02DIterationProgresses(t *testing.T) {
	ref := ramp(50, 50, 1)
	corrupt := ref.Clone()
	unknown := cv.NewMat(50, 50, 1)
	for y := 18; y < 34; y++ {
		for x := 18; x < 34; x++ {
			unknown.Data[y*50+x] = 255
			corrupt.Data[y*50+x] = 0
		}
	}
	kernel := CreateKernel(LinearBasis, 3)

	// First iteration fills a ring but leaves the centre; count strictly drops
	// and further iterations drive it to zero.
	prev := 16 * 16
	work := corrupt
	mask := unknown
	steps := 0
	for {
		out, rem, still := FT02DIteration(work, kernel, mask)
		if still >= prev && still != 0 {
			t.Fatalf("iteration made no progress: %d -> %d", prev, still)
		}
		work, mask, prev = out, rem, still
		steps++
		if still == 0 {
			break
		}
		if steps > 100 {
			t.Fatalf("did not converge")
		}
	}
	if e := maskedMAE(ref, work, unknown); e > 6.0 {
		t.Errorf("iterated single-steps MAE %.3f too high", e)
	}
}

func TestFT02DIterationPanics(t *testing.T) {
	img := cv.NewMat(8, 8, 1)
	small := cv.NewMat(4, 4, 1)
	expectPanic(t, "iter empty", func() { FT02DIteration(&cv.Mat{}, CreateKernel(LinearBasis, 2), img) })
	expectPanic(t, "iter mask", func() { FT02DIteration(img, CreateKernel(LinearBasis, 2), small) })
}

// --- multi-step inpaint ----------------------------------------------------

func TestInpaintMultiStepBeatsOneStep(t *testing.T) {
	ref := ramp(64, 64, 1)
	corrupt := ref.Clone()
	mask := cv.NewMat(64, 64, 1)
	for y := 16; y < 40; y++ { // a 24x24 hole, far wider than the kernel
		for x := 16; x < 40; x++ {
			mask.Data[y*64+x] = 255
			corrupt.Data[y*64+x] = 0
		}
	}
	radius := 3

	one := Inpaint(corrupt, mask, radius, LinearBasis, OneStep)
	multi := InpaintMultiStep(corrupt, mask, radius, LinearBasis)

	// Multi-step fills the whole hole; one-step leaves the centre at zero.
	for p := 0; p < 64*64; p++ {
		if mask.Data[p] != 0 && multi.Data[p] == 0 && ref.Data[p] != 0 {
			t.Fatalf("multi-step left pixel %d unfilled", p)
		}
	}
	eOne := MaskedError(ref, one, mask)
	eMulti := MaskedError(ref, multi, mask)
	if eMulti.MAE >= eOne.MAE {
		t.Errorf("multi-step MAE %.3f should beat one-step MAE %.3f", eMulti.MAE, eOne.MAE)
	}
	if eMulti.MAE > 6.0 {
		t.Errorf("multi-step MAE %.3f too high", eMulti.MAE)
	}
	// Known pixels preserved exactly.
	if multi.Data[0] != ref.Data[0] {
		t.Errorf("multi-step modified a known pixel")
	}
}

func TestInpaintMultiStepColor(t *testing.T) {
	ref := ramp(48, 48, 3)
	corrupt := ref.Clone()
	mask := cv.NewMat(48, 48, 1)
	for y := 14; y < 34; y++ {
		for x := 14; x < 34; x++ {
			mask.Data[y*48+x] = 255
			for c := 0; c < 3; c++ {
				corrupt.Data[(y*48+x)*3+c] = 0
			}
		}
	}
	out := InpaintMultiStep(corrupt, mask, 3, SinusBasis)
	if e := MaskedError(ref, out, mask); e.MAE > 7.0 {
		t.Errorf("colour multi-step MAE %.3f too high", e.MAE)
	}
}

func TestInpaintMultiStepPanics(t *testing.T) {
	img := cv.NewMat(8, 8, 1)
	small := cv.NewMat(4, 4, 1)
	expectPanic(t, "ms empty", func() { InpaintMultiStep(&cv.Mat{}, img, 2, LinearBasis) })
	expectPanic(t, "ms mask", func() { InpaintMultiStep(img, small, 2, LinearBasis) })
	expectPanic(t, "ms radius", func() { InpaintMultiStep(img, img, 0, LinearBasis) })
}

// --- quality reporting -----------------------------------------------------

func TestTransformErrorAndMasked(t *testing.T) {
	a := cv.NewMat(4, 4, 1)
	b := cv.NewMat(4, 4, 1)
	a.SetTo(100)
	b.SetTo(100)
	if e := TransformError(a, b); !math.IsInf(e.PSNR, 1) || e.MAE != 0 {
		t.Errorf("identical images: %+v", e)
	}
	b.Data[0] = 110 // one sample off by 10
	e := TransformError(a, b)
	if e.MaxAbs != 10 || e.Samples != 16 {
		t.Errorf("unexpected stats %+v", e)
	}
	if math.Abs(e.MAE-10.0/16) > 1e-12 {
		t.Errorf("MAE %.4f", e.MAE)
	}
	if e.String() == "" {
		t.Errorf("empty String()")
	}
	// Masked over the single differing pixel gives MAE 10.
	mask := cv.NewMat(4, 4, 1)
	mask.Data[0] = 255
	me := MaskedError(a, b, mask)
	if me.MAE != 10 || me.Samples != 1 {
		t.Errorf("masked stats %+v", me)
	}
}

func TestQualityPanics(t *testing.T) {
	a := cv.NewMat(4, 4, 1)
	b := cv.NewMat(4, 5, 1)
	expectPanic(t, "TE mismatch", func() { TransformError(a, b) })
	expectPanic(t, "TE nil", func() { TransformError(a, nil) })
	expectPanic(t, "ME mismatch", func() { MaskedError(a, b, a) })
}

// --- filter extensions -----------------------------------------------------

func TestFilterVariants(t *testing.T) {
	img := ramp(30, 32, 1)
	lin := FilterLinear(img, 3)
	sin := FilterSinus(img, 3)
	if mae(lin, Filter(img, LinearBasis, 3)) != 0 {
		t.Errorf("FilterLinear != Filter(LinearBasis)")
	}
	if mae(sin, Filter(img, SinusBasis, 3)) != 0 {
		t.Errorf("FilterSinus != Filter(SinusBasis)")
	}
	// Degree-1 filter preserves the gradient better than degree-0.
	d1 := FilterDegree1(img, LinearBasis, 3)
	if TransformError(img, d1).MAE >= TransformError(img, lin).MAE {
		t.Errorf("degree-1 filter should preserve the ramp better than degree-0")
	}
	// Multi-radius returns one image per radius, larger radii smooth more.
	outs := FilterMultiRadius(img, LinearBasis, []int{2, 4, 6})
	if len(outs) != 3 {
		t.Fatalf("got %d outputs", len(outs))
	}
}

func TestFilterExtPanics(t *testing.T) {
	img := cv.NewMat(8, 8, 1)
	expectPanic(t, "deg1 empty", func() { FilterDegree1(&cv.Mat{}, LinearBasis, 2) })
	expectPanic(t, "deg1 radius", func() { FilterDegree1(img, LinearBasis, 0) })
	expectPanic(t, "multi empty radii", func() { FilterMultiRadius(img, LinearBasis, nil) })
	expectPanic(t, "multi bad radius", func() { FilterMultiRadius(img, LinearBasis, []int{2, 0}) })
	expectPanic(t, "multi empty img", func() { FilterMultiRadius(&cv.Mat{}, LinearBasis, []int{2}) })
}
