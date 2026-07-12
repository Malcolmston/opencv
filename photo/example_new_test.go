package photo

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

func ExampleDomainTransformFilter() {
	// A perfectly flat image is unchanged by edge-preserving smoothing.
	img := cv.NewMat(6, 6, 1)
	img.SetTo(100)
	out := DomainTransformFilter(img, RecursFilter, 40, 0.3)
	fmt.Println(out.At(3, 3, 0))
	// Output: 100
}

func ExampleGammaCorrection() {
	// Gamma of 1 is the identity tone curve.
	img := cv.NewMat(2, 2, 1)
	img.SetTo(100)
	out := GammaCorrection(img, 1)
	fmt.Println(out.At(0, 0, 0))
	// Output: 100
}

func ExamplePencilSketch() {
	// A flat field dodges to near-white (only edges leave strokes).
	img := cv.NewMat(8, 8, 1)
	img.SetTo(120)
	gray, _ := PencilSketch(img, 40, 0.07, 0.02)
	fmt.Println(gray.At(4, 4, 0))
	// Output: 250
}

func ExampleGrayWorldWhiteBalance() {
	// An already-neutral image is left unchanged by gray-world balancing.
	img := cv.NewMat(3, 3, 3)
	img.SetTo(100)
	out := GrayWorldWhiteBalance(img)
	fmt.Println(out.At(1, 1, 0), out.At(1, 1, 1), out.At(1, 1, 2))
	// Output: 100 100 100
}

func ExampleHistogramStretch() {
	// The darkest sample maps to 0 and the brightest to 255.
	img := cv.NewMat(1, 3, 1)
	img.Set(0, 0, 0, 80)
	img.Set(0, 1, 0, 130)
	img.Set(0, 2, 0, 180)
	out := HistogramStretch(img)
	fmt.Println(out.At(0, 0, 0), out.At(0, 2, 0))
	// Output: 0 255
}
