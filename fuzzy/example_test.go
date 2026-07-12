package fuzzy_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/fuzzy"
)

func ExampleCreateKernel() {
	// A radius-2 linear kernel is 5x5, peaks at 1 in the centre and tapers to 0.
	k := fuzzy.CreateKernel(fuzzy.LinearBasis, 2)
	fmt.Printf("%dx%d centre=%.1f corner=%.1f\n", k.Rows, k.Cols, k.At(2, 2), k.At(0, 0))
	// Output: 5x5 centre=1.0 corner=0.0
}

func ExampleFilter() {
	// The F-transform reproduces a flat image exactly (partition of unity).
	img := cv.NewMat(6, 6, 1)
	img.SetTo(120)
	out := fuzzy.Filter(img, fuzzy.LinearBasis, 2)
	fmt.Println(out.At(3, 3, 0))
	// Output: 120
}

func ExampleInpaint() {
	// Reconstruct a single corrupted pixel from a uniform field.
	img := cv.NewMat(5, 5, 1)
	img.SetTo(90)
	img.Set(2, 2, 0, 0) // corrupted sample
	mask := cv.NewMat(5, 5, 1)
	mask.Set(2, 2, 0, 255) // non-zero marks the unknown pixel
	out := fuzzy.Inpaint(img, mask, 2, fuzzy.LinearBasis, fuzzy.OneStep)
	fmt.Println(out.At(2, 2, 0))
	// Output: 90
}

func ExampleFT02DProcess() {
	// FT02DProcess exposes the kernel directly; here it smooths a small image.
	img := cv.NewMat(8, 8, 1)
	img.SetTo(50)
	kernel := fuzzy.CreateKernel(fuzzy.SinusBasis, 2)
	out := fuzzy.FT02DProcess(img, kernel, nil)
	fmt.Println(out.At(4, 4, 0))
	// Output: 50
}
