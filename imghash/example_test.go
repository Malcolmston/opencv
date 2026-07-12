package imghash_test

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/imghash"
)

// ramp builds a small single-channel diagonal gradient for the examples.
func ramp(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Set(y, x, 0, uint8((x*255/cols+y*255/rows)/2))
		}
	}
	return m
}

// photo builds a small single-channel image with broadband structure (a few
// sinusoids) so its frequency-domain hash is well conditioned, unlike a smooth
// ramp. Values stay clear of 0 and 255 so a brightness shift does not clamp.
func photo(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		fy := float64(y) / float64(rows)
		for x := 0; x < cols; x++ {
			fx := float64(x) / float64(cols)
			v := 110 + 45*math.Sin(2*math.Pi*2*fx) +
				35*math.Sin(2*math.Pi*3*fy) +
				25*math.Sin(2*math.Pi*(fx+fy))
			m.Set(y, x, 0, uint8(v))
		}
	}
	return m
}

// ExampleAverageHash shows the basic compute-then-compare workflow: an image
// hashed against a copy of itself has Hamming distance zero.
func ExampleAverageHash() {
	img := ramp(32, 32)
	h := imghash.NewAverageHash()
	a := h.Compute(img)
	b := h.Compute(img.Clone())
	fmt.Printf("bytes=%d distance=%.0f\n", len(a), h.Compare(a, b))
	// Output: bytes=8 distance=0
}

// ExamplePHash demonstrates that the perceptual hash is invariant to a uniform
// brightness shift: adding a constant to every pixel leaves the hash unchanged.
func ExamplePHash() {
	img := photo(32, 32)
	bright := img.Clone()
	for i := range bright.Data {
		if v := int(bright.Data[i]) + 20; v <= 255 {
			bright.Data[i] = uint8(v)
		}
	}
	h := imghash.NewPHash()
	fmt.Printf("%.0f\n", h.Compare(h.Compute(img), h.Compute(bright)))
	// Output: 0
}

// ExampleDifference contrasts a blurred copy (perceptually similar, small
// distance) with a checkerboard (structurally different, large distance) using
// the difference hash convenience function.
func ExampleDifference() {
	base := ramp(64, 64)
	blurred := cv.GaussianBlur(base, 5, 0)

	checker := cv.NewMat(64, 64, 1)
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			if ((x/8)+(y/8))%2 == 0 {
				checker.Set(y, x, 0, 255)
			}
		}
	}

	h := imghash.NewDHash()
	hb := imghash.Difference(base)
	dBlur := h.Compare(hb, imghash.Difference(blurred))
	dChecker := h.Compare(hb, imghash.Difference(checker))
	fmt.Println(dBlur < dChecker)
	// Output: true
}

// ExampleColorMomentHash shows a real-valued hash: its L1 distance to a copy of
// the same image is exactly zero.
func ExampleColorMomentHash() {
	img := ramp(48, 48)
	h := imghash.NewColorMomentHash()
	fmt.Printf("%.1f\n", h.Compare(h.Compute(img), h.Compute(img.Clone())))
	// Output: 0.0
}
