package cudafilters_test

import (
	"fmt"
	"image"

	"github.com/malcolmston/opencv/cudafilters"
)

// Example uploads a small gradient image, smooths it on the (CPU-backed) device
// with a one-call Gaussian filter, and downloads the result — the
// upload/filter/download idiom of OpenCV's cudafilters module.
func Example() {
	src := cudafilters.GpuMatFromMat(ramp(5, 5))
	out := cudafilters.GaussianBlur(src, image.Pt(3, 3), 1.0, 1.0).Download()

	fmt.Printf("%dx%d\n", out.Rows, out.Cols)
	// Output: 5x5
}
