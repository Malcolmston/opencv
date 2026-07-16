package cudawarping_test

import (
	"fmt"
	"image"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudawarping"
)

// Example uploads a host image, resizes it on the (CPU-backed) device with
// bilinear interpolation, and downloads the enlarged result.
func Example() {
	host := cv.NewMat(2, 2, 1)
	host.SetTo(7)

	g := cudawarping.Upload(host)
	out := g.Resize(image.Point{X: 4, Y: 4}, 0, 0, cudawarping.InterLinear, nil).Download()

	fmt.Printf("%dx%d value=%d\n", out.Rows, out.Cols, out.At(0, 0, 0))
	// Output: 4x4 value=7
}
