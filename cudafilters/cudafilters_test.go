package cudafilters

import (
	"image"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// testMat builds a deterministic channels-channel test image with a mix of
// gradients and a bright/dark speck so that every filter has visible work to do.
func testMat(rows, cols, channels int) *cv.Mat {
	m := cv.NewMat(rows, cols, channels)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for c := 0; c < channels; c++ {
				v := (y*13 + x*7 + c*31) % 256
				m.Set(y, x, c, uint8(v))
			}
		}
	}
	// A couple of impulses to exercise rank/median/morphology filters.
	m.Set(1, 1, 0, 255)
	m.Set(rows-2, cols-2, 0, 0)
	return m
}

func matEqual(t *testing.T, got, want *cv.Mat) {
	t.Helper()
	if got == nil || want == nil {
		t.Fatalf("nil Mat: got=%v want=%v", got == nil, want == nil)
	}
	if got.Rows != want.Rows || got.Cols != want.Cols || got.Channels != want.Channels {
		t.Fatalf("shape mismatch: got %dx%dx%d want %dx%dx%d",
			got.Rows, got.Cols, got.Channels, want.Rows, want.Cols, want.Channels)
	}
	if len(got.Data) != len(want.Data) {
		t.Fatalf("data length mismatch: got %d want %d", len(got.Data), len(want.Data))
	}
	for i := range want.Data {
		if got.Data[i] != want.Data[i] {
			t.Fatalf("data mismatch at %d: got %d want %d", i, got.Data[i], want.Data[i])
		}
	}
}

// applyFilter uploads src, applies f, and downloads the result.
func applyFilter(f Filter, src *cv.Mat) *cv.Mat {
	g := GpuMatFromMat(src)
	return f.Apply(g).Download()
}

func TestCreateBoxFilter(t *testing.T) {
	src := testMat(9, 11, 1)
	got := applyFilter(CreateBoxFilter(image.Pt(3, 3), AnchorCenter, BorderDefault), src)
	matEqual(t, got, cv.BoxFilter(src, 3, true))
}

func TestCreateBlurFilter(t *testing.T) {
	src := testMat(8, 8, 3)
	got := applyFilter(CreateBlurFilter(image.Pt(5, 5)), src)
	matEqual(t, got, cv.BoxFilter(src, 5, true))
}

func TestCreateLinearFilter(t *testing.T) {
	src := testMat(7, 9, 3)
	k := cv.NewKernel(3, 3, []float64{0, -1, 0, -1, 5, -1, 0, -1, 0}) // sharpen
	got := applyFilter(CreateLinearFilter(k, 0, AnchorCenter, BorderDefault), src)
	matEqual(t, got, cv.Filter2D(src, k, 0))
}

func TestCreateSeparableLinearFilter(t *testing.T) {
	src := testMat(9, 7, 1)
	kx := []float64{1, 2, 1}
	ky := []float64{1, 0, -1}
	got := applyFilter(CreateSeparableLinearFilter(kx, ky, 2, AnchorCenter, BorderDefault), src)
	matEqual(t, got, cv.Filter2DSep(src, kx, ky, 2))
}

func TestCreateGaussianFilter(t *testing.T) {
	src := testMat(10, 10, 3)
	got := applyFilter(CreateGaussianFilter(image.Pt(5, 5), 1.2, 1.2, BorderDefault), src)
	kx := cv.GaussianKernel1D(5, 1.2)
	ky := cv.GaussianKernel1D(5, 1.2)
	matEqual(t, got, cv.Filter2DSep(src, kx, ky, 0))
}

func TestCreateGaussianFilterAsymmetric(t *testing.T) {
	src := testMat(9, 11, 1)
	got := applyFilter(CreateGaussianFilter(image.Pt(5, 3), 1.0, 2.0, BorderDefault), src)
	kx := cv.GaussianKernel1D(5, 1.0)
	ky := cv.GaussianKernel1D(3, 2.0)
	matEqual(t, got, cv.Filter2DSep(src, kx, ky, 0))
}

func TestCreateGaussianFilterSigma2Default(t *testing.T) {
	src := testMat(8, 8, 1)
	got := applyFilter(CreateGaussianFilter(image.Pt(3, 3), 0.8, 0, BorderDefault), src)
	k := cv.GaussianKernel1D(3, 0.8)
	matEqual(t, got, cv.Filter2DSep(src, k, k, 0))
}

func TestCreateSobelFilter(t *testing.T) {
	src := testMat(9, 9, 1)
	got := applyFilter(CreateSobelFilter(1, 0, 3, 1, 0, BorderDefault), src)
	matEqual(t, got, cv.Sobel(src, 1, 0, 3, 1, 0))
}

func TestCreateScharrFilter(t *testing.T) {
	src := testMat(9, 9, 3)
	got := applyFilter(CreateScharrFilter(0, 1, 1, 0, BorderDefault), src)
	matEqual(t, got, cv.Scharr(src, 0, 1, 1, 0))
}

func TestCreateDerivFilterSobel(t *testing.T) {
	src := testMat(8, 10, 1)
	got := applyFilter(CreateDerivFilter(1, 1, 3, 1, 0, BorderDefault), src)
	matEqual(t, got, cv.Sobel(src, 1, 1, 3, 1, 0))
}

func TestCreateDerivFilterScharr(t *testing.T) {
	src := testMat(8, 10, 1)
	got := applyFilter(CreateDerivFilter(1, 0, -1, 1, 0, BorderDefault), src)
	matEqual(t, got, cv.Scharr(src, 1, 0, 1, 0))
}

func TestCreateLaplacianFilter(t *testing.T) {
	src := testMat(9, 9, 1)
	got := applyFilter(CreateLaplacianFilter(1, 1, 0, BorderDefault), src)
	matEqual(t, got, cv.Laplacian(src, 1, 1, 0))
}

func TestCreateBoxMaxFilter(t *testing.T) {
	src := testMat(9, 9, 1)
	got := applyFilter(CreateBoxMaxFilter(image.Pt(3, 3), AnchorCenter, BorderDefault), src)
	kernel := cv.GetStructuringElement(cv.MorphRect, 3, 3)
	matEqual(t, got, cv.Dilate(src, kernel, 1))
}

func TestCreateBoxMinFilter(t *testing.T) {
	src := testMat(9, 9, 1)
	got := applyFilter(CreateBoxMinFilter(image.Pt(3, 3), AnchorCenter, BorderDefault), src)
	kernel := cv.GetStructuringElement(cv.MorphRect, 3, 3)
	matEqual(t, got, cv.Erode(src, kernel, 1))
}

func TestCreateMedianFilter(t *testing.T) {
	src := testMat(9, 9, 1)
	got := applyFilter(CreateMedianFilter(3), src)
	matEqual(t, got, cv.MedianBlur(src, 3))
}

func TestCreateMorphologyFilterAllOps(t *testing.T) {
	src := testMat(10, 10, 1)
	kernel := cv.GetStructuringElement(cv.MorphEllipse, 3, 3)
	cases := []struct {
		op   MorphOp
		want cv.MorphType
	}{
		{MorphErode, cv.MorphErode},
		{MorphDilate, cv.MorphDilate},
		{MorphOpen, cv.MorphOpen},
		{MorphClose, cv.MorphClose},
		{MorphGradient, cv.MorphGradient},
		{MorphTophat, cv.MorphTophat},
		{MorphBlackhat, cv.MorphBlackhat},
	}
	for _, tc := range cases {
		got := applyFilter(CreateMorphologyFilter(tc.op, kernel, AnchorCenter, 1), src)
		matEqual(t, got, cv.MorphologyEx(src, kernel, tc.want, 1))
	}
}

func TestMorphologyShortcuts(t *testing.T) {
	src := testMat(9, 9, 1)
	kernel := cv.GetStructuringElement(cv.MorphRect, 3, 3)
	matEqual(t, applyFilter(CreateErodeFilter(kernel, AnchorCenter, 2), src), cv.MorphologyEx(src, kernel, cv.MorphErode, 2))
	matEqual(t, applyFilter(CreateDilateFilter(kernel, AnchorCenter, 1), src), cv.MorphologyEx(src, kernel, cv.MorphDilate, 1))
	matEqual(t, applyFilter(CreateOpenFilter(kernel, AnchorCenter, 1), src), cv.MorphologyEx(src, kernel, cv.MorphOpen, 1))
	matEqual(t, applyFilter(CreateCloseFilter(kernel, AnchorCenter, 1), src), cv.MorphologyEx(src, kernel, cv.MorphClose, 1))
	matEqual(t, applyFilter(CreateMorphologyGradientFilter(kernel, AnchorCenter, 1), src), cv.MorphologyEx(src, kernel, cv.MorphGradient, 1))
	matEqual(t, applyFilter(CreateTopHatFilter(kernel, AnchorCenter, 1), src), cv.MorphologyEx(src, kernel, cv.MorphTophat, 1))
	matEqual(t, applyFilter(CreateBlackHatFilter(kernel, AnchorCenter, 1), src), cv.MorphologyEx(src, kernel, cv.MorphBlackhat, 1))
}

func TestCreateRowSumFilter(t *testing.T) {
	src := testMat(7, 9, 1)
	got := applyFilter(CreateRowSumFilter(3), src)
	matEqual(t, got, cv.Filter2DSep(src, []float64{1, 1, 1}, []float64{1}, 0))
}

func TestCreateColumnSumFilter(t *testing.T) {
	src := testMat(9, 7, 1)
	got := applyFilter(CreateColumnSumFilter(3), src)
	matEqual(t, got, cv.Filter2DSep(src, []float64{1}, []float64{1, 1, 1}, 0))
}

// ---------------------------------------------------------------------------
// Convenience wrappers should match their factories.
// ---------------------------------------------------------------------------

func TestConvenienceWrappers(t *testing.T) {
	src := testMat(9, 9, 1)
	g := GpuMatFromMat(src)
	kernel := cv.GetStructuringElement(cv.MorphRect, 3, 3)
	k := cv.NewKernel(3, 3, []float64{0, -1, 0, -1, 5, -1, 0, -1, 0})

	matEqual(t, BoxFilter(g, image.Pt(3, 3)).Download(), cv.BoxFilter(src, 3, true))
	matEqual(t, Blur(g, image.Pt(3, 3)).Download(), cv.BoxFilter(src, 3, true))
	matEqual(t, Filter2D(g, k, 0).Download(), cv.Filter2D(src, k, 0))
	matEqual(t, SepFilter2D(g, []float64{1, 2, 1}, []float64{1, 0, -1}, 0).Download(), cv.Filter2DSep(src, []float64{1, 2, 1}, []float64{1, 0, -1}, 0))
	matEqual(t, GaussianBlur(g, image.Pt(3, 3), 1, 1).Download(), cv.Filter2DSep(src, cv.GaussianKernel1D(3, 1), cv.GaussianKernel1D(3, 1), 0))
	matEqual(t, Sobel(g, 1, 0, 3, 1, 0).Download(), cv.Sobel(src, 1, 0, 3, 1, 0))
	matEqual(t, Scharr(g, 1, 0, 1, 0).Download(), cv.Scharr(src, 1, 0, 1, 0))
	matEqual(t, Laplacian(g, 1, 1, 0).Download(), cv.Laplacian(src, 1, 1, 0))
	matEqual(t, MedianBlur(g, 3).Download(), cv.MedianBlur(src, 3))
	matEqual(t, BoxMax(g, image.Pt(3, 3)).Download(), cv.Dilate(src, kernel, 1))
	matEqual(t, BoxMin(g, image.Pt(3, 3)).Download(), cv.Erode(src, kernel, 1))
	matEqual(t, Erode(g, kernel, 1).Download(), cv.MorphologyEx(src, kernel, cv.MorphErode, 1))
	matEqual(t, Dilate(g, kernel, 1).Download(), cv.MorphologyEx(src, kernel, cv.MorphDilate, 1))
	matEqual(t, MorphologyEx(g, MorphOpen, kernel, 1).Download(), cv.MorphologyEx(src, kernel, cv.MorphOpen, 1))
	matEqual(t, RowSum(g, 3).Download(), cv.Filter2DSep(src, []float64{1, 1, 1}, []float64{1}, 0))
	matEqual(t, ColumnSum(g, 3).Download(), cv.Filter2DSep(src, []float64{1}, []float64{1, 1, 1}, 0))
}

// Reusing one filter on several inputs must be consistent.
func TestFilterReuse(t *testing.T) {
	f := CreateGaussianFilter(image.Pt(3, 3), 1.0, 1.0, BorderDefault)
	for _, dims := range [][2]int{{5, 5}, {8, 6}, {7, 11}} {
		src := testMat(dims[0], dims[1], 1)
		got := f.Apply(GpuMatFromMat(src)).Download()
		k := cv.GaussianKernel1D(3, 1.0)
		matEqual(t, got, cv.Filter2DSep(src, k, k, 0))
	}
}

func TestApplyIgnoresStream(t *testing.T) {
	src := testMat(6, 6, 1)
	f := CreateBoxFilter(image.Pt(3, 3), AnchorCenter, BorderDefault)
	s := NewStream()
	got := f.Apply(GpuMatFromMat(src), s).Download()
	matEqual(t, got, cv.BoxFilter(src, 3, true))
	s.WaitForCompletion()
	s.Release()
}
