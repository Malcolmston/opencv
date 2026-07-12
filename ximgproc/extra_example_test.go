package ximgproc_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/ximgproc"
)

// ExampleDTFilter smooths a flat region while preserving a step edge using the
// recursive-filtering domain transform, self-guided.
func ExampleDTFilter() {
	src := cv.NewMat(16, 16, 1)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			v := uint8(60)
			if x >= 8 {
				v = 200
			}
			src.Data[y*16+x] = v
		}
	}
	out := ximgproc.DTFilter(src, nil, 12, 25, ximgproc.DTFilterRF, 3)
	fmt.Println(out.Rows, out.Cols, out.Channels)
	// The step across the middle is preserved.
	fmt.Println(out.Data[8*16+2] < 100 && out.Data[8*16+13] > 160)
	// Output:
	// 16 16 1
	// true
}

// ExampleJointBilateralFilter transfers a clean guide's edge onto a noisy input.
func ExampleJointBilateralFilter() {
	guide := cv.NewMat(8, 8, 1)
	src := cv.NewMat(8, 8, 1)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			v := uint8(40)
			if x >= 4 {
				v = 200
			}
			guide.Data[y*8+x] = v
			src.Data[y*8+x] = v
		}
	}
	out := ximgproc.JointBilateralFilter(guide, src, 5, 30, 4)
	fmt.Println(out.Channels)
	// Output:
	// 1
}

// ExampleFastGlobalSmootherFilter smooths an image while keeping its edges.
func ExampleFastGlobalSmootherFilter() {
	src := cv.NewMat(12, 12, 1)
	for i := range src.Data {
		src.Data[i] = 128
	}
	out := ximgproc.FastGlobalSmootherFilter(src, nil, 30, 25, 3)
	fmt.Println(out.Data[70]) // a flat image stays flat
	// Output:
	// 128
}

// ExampleRollingGuidanceFilter removes fine texture from an image.
func ExampleRollingGuidanceFilter() {
	src := cv.NewMat(16, 16, 1)
	for y := 0; y < 16; y++ {
		for x := 0; x < 16; x++ {
			v := 120
			if (x+y)%2 == 0 {
				v += 30
			}
			src.Data[y*16+x] = uint8(v)
		}
	}
	out := ximgproc.RollingGuidanceFilter(src, 7, 40, 4, 4)
	fmt.Println(out.Rows, out.Cols)
	// Output:
	// 16 16
}

// ExampleWeightedMedianFilter removes salt-and-pepper noise.
func ExampleWeightedMedianFilter() {
	src := cv.NewMat(10, 10, 1)
	for i := range src.Data {
		src.Data[i] = 100
	}
	src.Data[55] = 0   // pepper
	src.Data[44] = 255 // salt
	// A large sigma makes the weights near-uniform, i.e. an ordinary median,
	// which rejects the impulses.
	out := ximgproc.WeightedMedianFilter(src, nil, 2, 1000)
	fmt.Println(out.Data[55], out.Data[44])
	// Output:
	// 100 100
}

// ExampleGradientDericheX reports the sign of the recursive gradient across a
// left-to-right rising edge.
func ExampleGradientDericheX() {
	img := cv.NewMat(8, 8, 1)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			v := uint8(30)
			if x >= 4 {
				v = 220
			}
			img.Data[y*8+x] = v
		}
	}
	g := ximgproc.GradientDericheX(img, 1.0)
	fmt.Println(g.At(4, 3) > 0) // rising edge -> positive gradient
	// Output:
	// true
}

// ExampleCovarianceEstimation prints the shape of the estimated covariance.
func ExampleCovarianceEstimation() {
	img := cv.NewMat(10, 10, 1)
	for i := range img.Data {
		img.Data[i] = uint8(i % 7 * 30)
	}
	cov := ximgproc.CovarianceEstimation(img, 2, 2)
	fmt.Println(len(cov), len(cov[0])) // 4x4 for a 2x2 window
	// Output:
	// 4 4
}

// ExampleSuperpixelLSC segments a colour image into compact superpixels.
func ExampleSuperpixelLSC() {
	img := cv.NewMat(32, 32, 3)
	for y := 0; y < 32; y++ {
		for x := 0; x < 32; x++ {
			i := (y*32 + x) * 3
			img.Data[i+0] = uint8((x / 8) * 60)
			img.Data[i+1] = uint8((y / 8) * 60)
			img.Data[i+2] = 128
		}
	}
	labels, n := ximgproc.SuperpixelLSC(img, 8, 0.075)
	fmt.Println(labels.Rows, labels.Cols)
	fmt.Println(n > 0)
	// Output:
	// 32 32
	// true
}

// ExampleEdgeBoxes proposes object bounding boxes from an edge map.
func ExampleEdgeBoxes() {
	img := cv.NewMat(40, 40, 1)
	img.SetTo(20)
	for y := 10; y < 30; y++ {
		for x := 12; x < 28; x++ {
			img.Data[y*40+x] = 200
		}
	}
	edges := ximgproc.StructuredEdgeDetectionLite(img)
	boxes := ximgproc.EdgeBoxes(edges, 5)
	fmt.Println(len(boxes) > 0)
	// Output:
	// true
}
