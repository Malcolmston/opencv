package cudafilters_test

import (
	"fmt"
	"image"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudafilters"
)

// ramp builds a small single-channel gradient image for the examples.
func ramp(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Set(y, x, 0, uint8((x*20)%256))
		}
	}
	return m
}

// ExampleCreateGaussianFilter shows the object-style API: build a filter once
// and apply it to a GpuMat.
func ExampleCreateGaussianFilter() {
	src := cudafilters.GpuMatFromMat(ramp(5, 5))
	f := cudafilters.CreateGaussianFilter(image.Pt(3, 3), 1.0, 1.0, cudafilters.BorderDefault)
	dst := f.Apply(src)
	rows, cols := dst.Size()
	fmt.Printf("%dx%d\n", rows, cols)
	// Output: 5x5
}

// ExampleGaussianBlur shows the one-call convenience wrapper.
func ExampleGaussianBlur() {
	src := cudafilters.GpuMatFromMat(ramp(4, 4))
	dst := cudafilters.GaussianBlur(src, image.Pt(3, 3), 1.5, 1.5)
	fmt.Println(dst.Empty())
	// Output: false
}

// ExampleCreateMorphologyFilter shows a compound morphological operation driven
// by a structuring element from the root package.
func ExampleCreateMorphologyFilter() {
	src := cudafilters.GpuMatFromMat(ramp(6, 6))
	kernel := cv.GetStructuringElement(cv.MorphRect, 3, 3)
	f := cudafilters.CreateMorphologyFilter(cudafilters.MorphGradient, kernel, cudafilters.AnchorCenter, 1)
	dst := f.Apply(src)
	fmt.Println(dst.Channels())
	// Output: 1
}

// ExampleCreateSobelFilter computes a first-order horizontal derivative.
func ExampleCreateSobelFilter() {
	src := cudafilters.GpuMatFromMat(ramp(5, 5))
	f := cudafilters.CreateSobelFilter(1, 0, 3, 1, 0, cudafilters.BorderDefault)
	out := f.Apply(src).Download()
	// The ramp increases along x, so the horizontal derivative is a constant
	// positive value in the interior.
	fmt.Println(out.At(2, 2, 0))
	// Output: 160
}
