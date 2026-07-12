package quality_test

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/quality"
)

// ramp builds a small single-channel gradient image for the examples.
func ramp(rows, cols int) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			m.Set(y, x, 0, uint8((x*255/cols+y*255/rows)/2))
		}
	}
	return m
}

// ExamplePSNR shows that comparing an image with itself yields an infinite
// peak signal-to-noise ratio (zero error).
func ExamplePSNR() {
	img := ramp(16, 16)
	p := quality.PSNR(img, img.Clone())
	fmt.Println(math.IsInf(p, 1))
	// Output: true
}

// ExampleSSIM reports the mean structural similarity of an image with itself,
// which is exactly 1.
func ExampleSSIM() {
	img := ramp(32, 32)
	mean, qmap := quality.SSIM(img, img.Clone())
	fmt.Printf("%.3f %dx%d\n", mean, qmap.Rows, qmap.Cols)
	// Output: 1.000 32x32
}

// ExampleGMSD shows that the gradient magnitude similarity deviation of an
// image with itself is zero.
func ExampleGMSD() {
	img := ramp(24, 24)
	dev, _ := quality.GMSD(img, img.Clone())
	fmt.Printf("%.3f\n", dev)
	// Output: 0.000
}

// ExampleSharpness contrasts the variance-of-Laplacian sharpness of a sharp
// checkerboard with a blurred copy of it.
func ExampleSharpness() {
	sharp := cv.NewMat(32, 32, 1)
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			if ((x/4)+(y/4))%2 == 0 {
				sharp.Set(y, x, 0, 255)
			}
		}
	}
	blurred := cv.GaussianBlur(sharp, 7, 0)
	fmt.Println(quality.Sharpness(blurred) < quality.Sharpness(sharp))
	// Output: true
}

// ExampleNewQualitySSIM demonstrates the OpenCV-style object form: build the
// metric once with a reference, then score a candidate.
func ExampleNewQualitySSIM() {
	ref := ramp(32, 32)
	q := quality.NewQualitySSIM(ref)
	score := q.Compute(ref.Clone())
	fmt.Printf("%.3f\n", score[0])
	// Output: 1.000
}
