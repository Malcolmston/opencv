package superres

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// approx reports whether a and b are within tol.
func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// constMat builds a rows×cols×ch image with every sample set to v.
func constMat(rows, cols, ch int, v uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, ch)
	for i := range m.Data {
		m.Data[i] = v
	}
	return m
}

// gray1 builds a single-channel image from a flat row-major slice.
func gray1(rows, cols int, vals []uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	copy(m.Data, vals)
	return m
}

func TestBilinearKnownAnswer(t *testing.T) {
	// A 1×2 row [0, 100] widened to 4 columns has known bilinear values.
	src := gray1(1, 2, []uint8{0, 100})
	got := BilinearResize(src, 4, 1)
	want := []uint8{0, 25, 75, 100}
	for i, w := range want {
		if got.Data[i] != w {
			t.Errorf("bilinear[%d] = %d, want %d", i, got.Data[i], w)
		}
	}
}

func TestNearestKnownAnswer(t *testing.T) {
	src := gray1(1, 2, []uint8{0, 100})
	got := NearestResize(src, 4, 1)
	want := []uint8{0, 0, 100, 100}
	for i, w := range want {
		if got.Data[i] != w {
			t.Errorf("nearest[%d] = %d, want %d", i, got.Data[i], w)
		}
	}
}

func TestKernelPointValues(t *testing.T) {
	ck := CatmullRomKernel()
	if !approx(ck.Weight(0), 1, 1e-12) {
		t.Errorf("cubic W(0) = %v, want 1", ck.Weight(0))
	}
	if !approx(ck.Weight(1), 0, 1e-12) {
		t.Errorf("cubic W(1) = %v, want 0", ck.Weight(1))
	}
	if !approx(ck.Weight(2), 0, 1e-12) {
		t.Errorf("cubic W(2) = %v, want 0", ck.Weight(2))
	}
	if !approx(ck.Weight(0.5), 0.5625, 1e-12) {
		t.Errorf("cubic W(0.5) = %v, want 0.5625", ck.Weight(0.5))
	}
	if !approx(Sinc(0), 1, 1e-12) {
		t.Errorf("Sinc(0) = %v, want 1", Sinc(0))
	}
	if !approx(Sinc(1), 0, 1e-12) {
		t.Errorf("Sinc(1) = %v, want 0", Sinc(1))
	}
	if !approx(Sinc(0.5), 2/math.Pi, 1e-12) {
		t.Errorf("Sinc(0.5) = %v, want 2/pi", Sinc(0.5))
	}
	mk := MitchellKernel(1.0/3.0, 1.0/3.0)
	if !approx(mk.Weight(0), 16.0/18.0, 1e-12) {
		t.Errorf("mitchell W(0) = %v, want 8/9", mk.Weight(0))
	}
	lk := LanczosKernel(3)
	if !approx(lk.Weight(0), 1, 1e-12) {
		t.Errorf("lanczos W(0) = %v, want 1", lk.Weight(0))
	}
	if lk.Radius != 3 {
		t.Errorf("lanczos3 radius = %v, want 3", lk.Radius)
	}
}

func TestKernelPartitionOfUnity(t *testing.T) {
	// The linear and Keys cubic kernels partition unity exactly at every phase;
	// the windowed-sinc Lanczos kernels only approximately (the resampler
	// normalises weights to compensate), so they are held to a looser bound.
	exact := []ResampleKernel{LinearKernel(), CatmullRomKernel()}
	approxK := []ResampleKernel{LanczosKernel(2), LanczosKernel(3)}
	check := func(k ResampleKernel, tol float64) {
		for _, phase := range []float64{0, 0.1, 0.25, 0.5, 0.73, 0.99} {
			var sum float64
			r := int(math.Ceil(k.Radius))
			for i := -r; i <= r; i++ {
				sum += k.Weight(phase - float64(i))
			}
			if !approx(sum, 1, tol) {
				t.Errorf("%s partition at phase %v = %v, want 1 (tol %v)", k.Name, phase, sum, tol)
			}
		}
	}
	for _, k := range exact {
		check(k, 1e-9)
	}
	for _, k := range approxK {
		check(k, 0.03)
	}
}

func TestResizeConstantPreserved(t *testing.T) {
	src := constMat(6, 6, 3, 100)
	interps := []Interpolation{
		InterpNearest, InterpBilinear, InterpBicubic, InterpMitchell,
		InterpBSpline, InterpLanczos2, InterpLanczos3,
	}
	for _, in := range interps {
		out := Resize(src, 9, 9, in)
		if out.Rows != 9 || out.Cols != 9 || out.Channels != 3 {
			t.Fatalf("interp %d: bad dims %dx%dx%d", in, out.Rows, out.Cols, out.Channels)
		}
		for i, v := range out.Data {
			if v != 100 {
				t.Fatalf("interp %d: sample %d = %d, want 100", in, i, v)
				break
			}
		}
	}
}

func TestResizeScaleDims(t *testing.T) {
	src := constMat(10, 20, 1, 50)
	up := ResizeScale(src, 1.5, InterpBicubic)
	if up.Rows != 15 || up.Cols != 30 {
		t.Errorf("ResizeScale dims = %dx%d, want 15x30", up.Rows, up.Cols)
	}
	down := ResizeScale(src, 0.5, InterpBilinear)
	if down.Rows != 5 || down.Cols != 10 {
		t.Errorf("ResizeScale down dims = %dx%d, want 5x10", down.Rows, down.Cols)
	}
}

func TestBoxDownscaleKnownAnswer(t *testing.T) {
	src := gray1(4, 4, []uint8{
		10, 10, 20, 20,
		10, 10, 20, 20,
		30, 30, 40, 40,
		30, 30, 40, 40,
	})
	got := BoxDownscale(src, 2)
	want := []uint8{10, 20, 30, 40}
	if got.Rows != 2 || got.Cols != 2 {
		t.Fatalf("dims = %dx%d, want 2x2", got.Rows, got.Cols)
	}
	for i, w := range want {
		if got.Data[i] != w {
			t.Errorf("box[%d] = %d, want %d", i, got.Data[i], w)
		}
	}
}

func TestGaussianDownscaleConstant(t *testing.T) {
	src := constMat(16, 16, 1, 77)
	got := GaussianDownscale(src, 4, 0)
	if got.Rows != 4 || got.Cols != 4 {
		t.Fatalf("dims = %dx%d, want 4x4", got.Rows, got.Cols)
	}
	for i, v := range got.Data {
		if v != 77 {
			t.Fatalf("sample %d = %d, want 77", i, v)
		}
	}
}

func TestGaussianKernel1D(t *testing.T) {
	k := GaussianKernel1D(1.5)
	var sum float64
	for _, v := range k {
		sum += v
	}
	if !approx(sum, 1, 1e-12) {
		t.Errorf("gaussian kernel sum = %v, want 1", sum)
	}
	// Symmetry.
	for i := 0; i < len(k)/2; i++ {
		if !approx(k[i], k[len(k)-1-i], 1e-12) {
			t.Errorf("gaussian kernel not symmetric at %d", i)
		}
	}
}

func TestSharpenConstantUnchanged(t *testing.T) {
	src := constMat(8, 8, 3, 120)
	ops := map[string]*cv.Mat{
		"gaussian": GaussianBlur(src, 1.2),
		"unsharp":  UnsharpMask(src, 1.0, 0.8, 0),
		"laplace":  SharpenLaplacian(src, 0.7),
		"adaptive": AdaptiveUnsharpMask(src, 1.0, 0.8, 10),
	}
	for name, out := range ops {
		for i, v := range out.Data {
			if v != 120 {
				t.Fatalf("%s changed constant at %d: %d", name, i, v)
			}
		}
	}
}

func TestUnsharpMaskIncreasesContrast(t *testing.T) {
	// A soft step edge should become steeper (larger max-min gradient) after
	// unsharp masking.
	src := gray1(1, 6, []uint8{100, 110, 130, 150, 160, 170})
	out := UnsharpMask(src, 1.0, 1.0, 0)
	// The central difference at the steepest point should grow.
	before := float64(src.Data[4]) - float64(src.Data[2])
	after := float64(out.Data[4]) - float64(out.Data[2])
	if after <= before {
		t.Errorf("unsharp did not increase contrast: before=%v after=%v", before, after)
	}
}

func TestUpscaleAndSharpenDims(t *testing.T) {
	src := constMat(8, 8, 1, 90)
	out := UpscaleAndSharpen(src, 16, 16, InterpBicubic, 1.0, 0.5)
	if out.Rows != 16 || out.Cols != 16 {
		t.Errorf("dims = %dx%d, want 16x16", out.Rows, out.Cols)
	}
	for _, v := range out.Data {
		if v != 90 {
			t.Fatal("upscale+sharpen changed constant image")
		}
	}
}
