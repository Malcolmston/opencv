package intensity_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/intensity"
)

// rampRow builds a 1x256 single-channel image whose column x holds intensity x.
func rampRow() *cv.Mat {
	m := cv.NewMat(1, 256, 1)
	for x := 0; x < 256; x++ {
		m.Set(0, x, 0, uint8(x))
	}
	return m
}

// ExampleGammaCorrection shows that gamma < 1 lifts a midtone toward white
// while gamma == 1 is the identity.
func ExampleGammaCorrection() {
	mid := cv.NewMat(1, 1, 1)
	mid.Set(0, 0, 0, 128)
	fmt.Println(intensity.GammaCorrection(mid, 1.0).At(0, 0, 0))
	fmt.Println(intensity.GammaCorrection(mid, 0.5).At(0, 0, 0))
	// Output:
	// 128
	// 181
}

// ExampleLogTransform shows the endpoints are preserved and low values lifted.
func ExampleLogTransform() {
	out := intensity.LogTransform(rampRow())
	fmt.Println(out.At(0, 0, 0), out.At(0, 64, 0), out.At(0, 255, 0))
	// Output: 0 192 255
}

// ExampleContrastStretching reproduces the control points exactly.
func ExampleContrastStretching() {
	out := intensity.ContrastStretching(rampRow(), 50, 30, 200, 220)
	fmt.Println(out.At(0, 50, 0), out.At(0, 200, 0))
	// Output: 30 220
}

// ExampleBitPlaneSlicing shows the most-significant bit plane of a ramp is a
// single step from 0 to 255 at the midpoint.
func ExampleBitPlaneSlicing() {
	out := intensity.BitPlaneSlicing(rampRow(), 7)
	fmt.Println(out.At(0, 127, 0), out.At(0, 128, 0))
	// Output: 0 255
}

// ExampleInvert produces the photographic negative.
func ExampleInvert() {
	out := intensity.Invert(rampRow())
	fmt.Println(out.At(0, 0, 0), out.At(0, 255, 0))
	// Output: 255 0
}

// ExampleAutoscaleContrast stretches a [50,200] ramp onto the full range.
func ExampleAutoscaleContrast() {
	m := cv.NewMat(1, 4, 1)
	m.Set(0, 0, 0, 50)
	m.Set(0, 1, 0, 100)
	m.Set(0, 2, 0, 150)
	m.Set(0, 3, 0, 200)
	out := intensity.AutoscaleContrast(m)
	fmt.Println(out.At(0, 0, 0), out.At(0, 3, 0))
	// Output: 0 255
}
