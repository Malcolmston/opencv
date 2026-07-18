package cv

import (
	"math"
	"testing"
)

func TestLUT(t *testing.T) {
	src := NewMat(1, 3, 1)
	copy(src.Data, []uint8{0, 128, 255})
	var inv [256]uint8
	for i := range inv {
		inv[i] = uint8(255 - i)
	}
	dst := LUT(src, inv[:])
	if dst.Data[0] != 255 || dst.Data[1] != 127 || dst.Data[2] != 0 {
		t.Errorf("LUT = %v", dst.Data)
	}
}

func TestGetGaussianKernel(t *testing.T) {
	k := GetGaussianKernel(5, 1)
	var sum float64
	for _, v := range k {
		sum += v
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Errorf("kernel sum = %v, want 1", sum)
	}
	if k[0] != k[4] || k[1] != k[3] {
		t.Error("kernel not symmetric")
	}
	if k[2] <= k[1] {
		t.Error("center weight should be largest")
	}
}

func TestGetDerivKernels(t *testing.T) {
	kx, ky := GetDerivKernels(1, 0, 3)
	if !floatSliceEq(kx, []float64{-1, 0, 1}) {
		t.Errorf("kx = %v, want [-1 0 1]", kx)
	}
	if !floatSliceEq(ky, []float64{1, 2, 1}) {
		t.Errorf("ky = %v, want [1 2 1]", ky)
	}
	kx2, _ := GetDerivKernels(2, 0, 3)
	if !floatSliceEq(kx2, []float64{1, -2, 1}) {
		t.Errorf("2nd deriv kx = %v, want [1 -2 1]", kx2)
	}
}

func floatSliceEq(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > 1e-12 {
			return false
		}
	}
	return true
}

func TestSqrBoxFilterConstant(t *testing.T) {
	src := NewMat(5, 5, 1)
	for i := range src.Data {
		src.Data[i] = 10
	}
	out := SqrBoxFilter(src, 3, true)
	if math.Abs(out.At(2, 2)-100) > 1e-9 {
		t.Errorf("normalized sqr box = %v, want 100", out.At(2, 2))
	}
}

func TestStackBlurConstant(t *testing.T) {
	src := NewMat(6, 6, 1)
	for i := range src.Data {
		src.Data[i] = 77
	}
	out := StackBlur(src, 3)
	for _, v := range out.Data {
		if v != 77 {
			t.Fatalf("stack blur changed constant to %d", v)
		}
	}
}

func TestGetAffineTransform(t *testing.T) {
	src := [3]Point2f{{0, 0}, {1, 0}, {0, 1}}
	dst := [3]Point2f{{0, 0}, {2, 0}, {0, 2}}
	m := GetAffineTransform(src, dst)
	want := AffineMatrix{2, 0, 0, 0, 2, 0}
	for i := range want {
		if math.Abs(m[i]-want[i]) > 1e-9 {
			t.Errorf("affine[%d] = %v, want %v", i, m[i], want[i])
		}
	}
}

func TestInvertAffineTransform(t *testing.T) {
	m := AffineMatrix{1, 0, 5, 0, 1, 3}
	inv := InvertAffineTransform(m)
	want := AffineMatrix{1, 0, -5, 0, 1, -3}
	for i := range want {
		if math.Abs(inv[i]-want[i]) > 1e-9 {
			t.Errorf("inv[%d] = %v, want %v", i, inv[i], want[i])
		}
	}
}

func TestWarpPolarConstant(t *testing.T) {
	src := NewMat(20, 20, 1)
	for i := range src.Data {
		src.Data[i] = 123
	}
	out := WarpPolar(src, 16, 16, 10, 10, 8, WarpPolarLinear)
	if out.Rows != 16 || out.Cols != 16 {
		t.Fatalf("dims = %dx%d", out.Rows, out.Cols)
	}
	if out.At(5, 5, 0) != 123 {
		t.Errorf("constant not preserved: %d", out.At(5, 5, 0))
	}
}

func TestCornerMinEigenValConstant(t *testing.T) {
	src := NewMat(10, 10, 1)
	for i := range src.Data {
		src.Data[i] = 50
	}
	out := CornerMinEigenVal(src, 3, 3)
	for _, v := range out.Data {
		if math.Abs(v) > 1e-9 {
			t.Fatalf("constant image should have zero response, got %v", v)
		}
	}
}

func TestFloodFill(t *testing.T) {
	src := NewMat(3, 3, 1)
	for i := range src.Data {
		src.Data[i] = 50
	}
	out, count := FloodFill(src, 1, 1, 200, 0, 0)
	if count != 9 {
		t.Errorf("flood count = %d, want 9", count)
	}
	for _, v := range out.Data {
		if v != 200 {
			t.Fatalf("unfilled pixel: %d", v)
		}
	}
	// Two regions separated by value.
	seg := NewMat(1, 4, 1)
	copy(seg.Data, []uint8{10, 10, 200, 200})
	_, c := FloodFill(seg, 0, 0, 5, 0, 0)
	if c != 2 {
		t.Errorf("region count = %d, want 2", c)
	}
}

func TestDemosaicConstant(t *testing.T) {
	src := NewMat(4, 4, 1)
	for i := range src.Data {
		src.Data[i] = 100
	}
	out := Demosaic(src, BayerRG)
	if out.Channels != 3 {
		t.Fatalf("channels = %d", out.Channels)
	}
	for _, v := range out.Data {
		if v != 100 {
			t.Fatalf("demosaic constant produced %d", v)
		}
	}
}

func TestGammaCorrectIdentity(t *testing.T) {
	src := NewMat(1, 4, 1)
	copy(src.Data, []uint8{0, 64, 128, 255})
	out := GammaCorrect(src, 1)
	for i := range src.Data {
		if out.Data[i] != src.Data[i] {
			t.Errorf("gamma 1 changed %d to %d", src.Data[i], out.Data[i])
		}
	}
}

func TestHSVFullRoundTrip(t *testing.T) {
	src := NewMat(1, 2, 3)
	copy(src.Data, []uint8{255, 0, 0, 0, 255, 0})
	hsv := RGBToHSVFull(src)
	if hsv.At(0, 0, 1) != 255 || hsv.At(0, 0, 2) != 255 {
		t.Errorf("red HSV = %v", hsv.AtPixel(0, 0))
	}
	if hsv.At(0, 1, 0) != 85 {
		t.Errorf("green hue = %d, want 85", hsv.At(0, 1, 0))
	}
	back := HSVFullToRGB(hsv)
	if back.At(0, 0, 0) != 255 || back.At(0, 0, 1) != 0 {
		t.Errorf("round trip red = %v", back.AtPixel(0, 0))
	}
}

func TestVarianceEntropyMedian(t *testing.T) {
	m := NewMat(1, 4, 1)
	copy(m.Data, []uint8{2, 4, 4, 6})
	if math.Abs(VarianceMat(m)-2) > 1e-9 {
		t.Errorf("variance = %v, want 2", VarianceMat(m))
	}
	if math.Abs(StdDevMat(m)-math.Sqrt2) > 1e-9 {
		t.Errorf("stddev = %v", StdDevMat(m))
	}
	c := NewMat(2, 2, 1)
	for i := range c.Data {
		c.Data[i] = 9
	}
	if Entropy(c) != 0 {
		t.Errorf("constant entropy = %v, want 0", Entropy(c))
	}
	half := NewMat(1, 4, 1)
	copy(half.Data, []uint8{1, 1, 2, 2})
	if math.Abs(Entropy(half)-1) > 1e-9 {
		t.Errorf("half/half entropy = %v, want 1", Entropy(half))
	}
	med := NewMat(1, 5, 1)
	copy(med.Data, []uint8{1, 2, 3, 4, 5})
	if Median(med) != 3 {
		t.Errorf("median = %v, want 3", Median(med))
	}
}

func TestMSEAndMinMaxLocMat(t *testing.T) {
	a := NewMat(2, 2, 1)
	b := a.Clone()
	if MSE(a, b) != 0 {
		t.Error("identical MSE should be 0")
	}
	m := NewMat(1, 4, 1)
	copy(m.Data, []uint8{3, 9, 1, 7})
	mn, mx, _, _, mxX, _ := MinMaxLocMat(m)
	if mn != 1 || mx != 9 || mxX != 1 {
		t.Errorf("minmax = %v %v at x=%d", mn, mx, mxX)
	}
}

func TestTriangleThreshold(t *testing.T) {
	src := NewMat(1, 6, 1)
	copy(src.Data, []uint8{10, 12, 11, 200, 205, 203})
	out, thr := TriangleThreshold(src)
	if thr < 0 || thr > 255 {
		t.Errorf("threshold out of range: %v", thr)
	}
	for _, v := range out.Data {
		if v != 0 && v != 255 {
			t.Fatalf("non-binary output: %d", v)
		}
	}
}

func BenchmarkStackBlur(b *testing.B) {
	src := NewMat(64, 64, 3)
	for i := range src.Data {
		src.Data[i] = uint8(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		StackBlur(src, 5)
	}
}
