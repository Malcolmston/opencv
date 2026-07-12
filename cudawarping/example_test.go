package cudawarping_test

import (
	"fmt"
	"image"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudawarping"
)

// ExampleUpload shows the upload/operate/download round-trip that mirrors the
// cv::cuda GpuMat workflow.
func ExampleUpload() {
	host := cv.NewMat(4, 4, 1)
	host.SetTo(7)

	g := cudawarping.Upload(host) // host -> "device"
	up := g.PyrUp(nil)            // run a warp on the "device"
	result := up.Download()       // "device" -> host

	fmt.Printf("%dx%d\n", result.Rows, result.Cols)
	// Output: 8x8
}

// ExampleGpuMat_Resize doubles an image with bilinear interpolation.
func ExampleGpuMat_Resize() {
	g := cudawarping.NewGpuMat(2, 2, 1)
	out := g.Resize(image.Point{X: 4, Y: 4}, 0, 0, cudawarping.InterLinear, nil)
	r, c := out.Size()
	fmt.Printf("%dx%d\n", r, c)
	// Output: 4x4
}

// ExampleGpuMat_WarpAffine translates an image so a known pixel moves by a fixed
// offset.
func ExampleGpuMat_WarpAffine() {
	src := cv.NewMat(5, 5, 1)
	src.Set(1, 1, 0, 255) // bright at x=1, y=1
	g := cudawarping.Upload(src)

	// Forward translation by (dx, dy) = (2, 1).
	m := cv.AffineMatrix{1, 0, 2, 0, 1, 1}
	out := g.WarpAffine(m, image.Point{X: 5, Y: 5}, int(cudawarping.InterNearest), cudawarping.BorderConstant, 0, nil).Download()

	fmt.Println(out.At(2, 3, 0)) // bright moved to x=3, y=2
	// Output: 255
}

// ExampleGpuMat_Rotate90 rotates an image 90° clockwise, swapping its
// dimensions.
func ExampleGpuMat_Rotate90() {
	g := cudawarping.NewGpuMat(3, 5, 1)
	out := g.Rotate90(cudawarping.Rotate90CW, nil)
	r, c := out.Size()
	fmt.Printf("%dx%d\n", r, c)
	// Output: 5x3
}

// ExampleGpuMat_LinearPolar unwraps an image into (radius, angle) space; a
// feature at angle 0 lands on the first row.
func ExampleGpuMat_LinearPolar() {
	src := cv.NewMat(21, 21, 1)
	src.Set(10, 15, 0, 255) // 5 px right of the centre (10,10)
	g := cudawarping.Upload(src)

	polar := g.LinearPolar(image.Point{X: 10, Y: 8}, cudawarping.Point2f{X: 10, Y: 10}, 10, int(cudawarping.InterNearest), nil).Download()
	fmt.Println(polar.At(0, 5, 0))
	// Output: 255
}
