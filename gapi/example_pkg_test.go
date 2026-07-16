package gapi_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/gapi"
)

// Example builds the canonical G-API edge-detection pipeline
// Canny(GaussianBlur(RGB2Gray(in))), compiles it once, and applies the compiled
// graph to a synthetic RGB image, printing the single-channel output size.
func Example() {
	// Describe the pipeline symbolically; no pixels are touched yet.
	in := gapi.NewMat()
	gray := gapi.RGB2Gray(in)
	blurred := gapi.GaussianBlur(gray, 5, 1.4)
	edges := gapi.Canny(blurred, 50, 100)

	// Capture the graph and compile it into a reusable executable form.
	cc := gapi.NewComputation(in, edges).Compile()

	// Feed a concrete 3-channel image and run the whole graph once.
	img := cv.NewMat(16, 16, 3)
	for i := range img.Data {
		img.Data[i] = uint8(i % 256)
	}
	out := cc.Apply(img)[0]

	fmt.Printf("%dx%dx%d\n", out.Rows, out.Cols, out.Channels)
	// Output: 16x16x1
}
