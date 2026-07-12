package quality_test

import (
	"fmt"
	"math"

	"github.com/malcolmston/opencv/quality"
)

// ExampleRMSE shows that the root-mean-squared error of an image with itself is
// exactly zero.
func ExampleRMSE() {
	img := ramp(16, 16)
	fmt.Printf("%.3f\n", quality.RMSE(img, img.Clone())[0])
	// Output: 0.000
}

// ExampleFSIM reports the feature-similarity index of an image with itself,
// which is exactly 1.
func ExampleFSIM() {
	img := ramp(32, 32)
	fmt.Printf("%.3f\n", quality.FSIM(img, img.Clone()))
	// Output: 1.000
}

// ExampleCWSSIM shows that the complex-wavelet SSIM of an image with itself is 1.
func ExampleCWSSIM() {
	img := ramp(32, 32)
	fmt.Printf("%.3f\n", quality.CWSSIM(img, img.Clone()))
	// Output: 1.000
}

// ExampleVIFP shows that the visual information fidelity of an image with itself
// is (to within numerical tolerance) 1.
func ExampleVIFP() {
	img := ramp(32, 32)
	fmt.Println(quality.VIFP(img, img.Clone()) > 0.99)
	// Output: true
}

// ExampleEntropyDiff shows that two images with the same luminance histogram —
// here an image and its own copy — have zero entropy difference.
func ExampleEntropyDiff() {
	img := ramp(24, 24)
	fmt.Printf("%.3f\n", quality.EntropyDiff(img, img.Clone()))
	// Output: 0.000
}

// ExampleSNR shows that the signal-to-noise ratio of an image with itself is
// infinite (there is no noise).
func ExampleSNR() {
	img := ramp(16, 16)
	fmt.Println(math.IsInf(quality.SNR(img, img.Clone()), 1))
	// Output: true
}
