package intensity_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/intensity"
)

// ExampleGammaLUT builds a power-law table and reads a few of its entries.
func ExampleGammaLUT() {
	lut := intensity.GammaLUT(2.0)
	fmt.Println(lut[0], lut[128], lut[255])
	// Output: 0 64 255
}

// ExampleAutoGammaValue reports the automatic exponent for a dark patch, which
// is < 1 (a brightening correction).
func ExampleAutoGammaValue() {
	dark := cv.NewMat(1, 1, 1)
	dark.Set(0, 0, 0, 64)
	fmt.Printf("%.2f\n", intensity.AutoGammaValue(dark))
	// Output: 0.50
}

// ExampleToneCurveLUT shows that the spline reproduces its control points.
func ExampleToneCurveLUT() {
	lut := intensity.ToneCurveLUT([]intensity.CurvePoint{
		{In: 0, Out: 0}, {In: 128, Out: 200}, {In: 255, Out: 255},
	})
	fmt.Println(lut[0], lut[128], lut[255])
	// Output: 0 200 255
}

// ExampleAutoContrast stretches a low-range grayscale row across [0,255].
func ExampleAutoContrast() {
	m := cv.NewMat(1, 4, 1)
	m.Set(0, 0, 0, 50)
	m.Set(0, 1, 0, 100)
	m.Set(0, 2, 0, 150)
	m.Set(0, 3, 0, 200)
	out := intensity.AutoContrast(m, 0)
	fmt.Println(out.At(0, 0, 0), out.At(0, 1, 0), out.At(0, 2, 0), out.At(0, 3, 0))
	// Output: 0 85 170 255
}
