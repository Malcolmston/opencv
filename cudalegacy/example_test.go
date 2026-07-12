package cudalegacy_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudalegacy"
)

// ExampleGpuMat_upload shows the upload/download idiom: a host frame is uploaded
// to a GpuMat and downloaded back, mirroring the OpenCV CUDA API. The Stream is
// a no-op.
func ExampleGpuMat_upload() {
	stream := cudalegacy.NewStream()
	host := cv.NewMat(4, 4, 1)
	host.SetTo(42)

	device := cudalegacy.NewGpuMat()
	device.Upload(host, stream)
	back := device.Download(stream)

	fmt.Println(back.At(2, 2, 0))
	// Output: 42
}

// ExampleGraphCut segments a strip whose left half prefers the source terminal
// and right half the sink, recovering a clean vertical cut.
func ExampleGraphCut() {
	src := cv.NewFloatMat(1, 4)
	snk := cv.NewFloatMat(1, 4)
	src.Data = []float64{9, 9, 1, 1}
	snk.Data = []float64{1, 1, 9, 9}

	labels := cudalegacy.GraphCut(src, snk, 1.0, nil)
	for _, v := range labels.Mat.Data {
		fmt.Printf("%d ", v)
	}
	fmt.Println()
	// Output: 255 255 0 0
}

// ExampleProjectPoints projects a single 3D point lying on the optical axis; it
// lands exactly on the principal point regardless of lens distortion.
func ExampleProjectPoints() {
	K := [3][3]float64{{500, 0, 320}, {0, 500, 240}, {0, 0, 1}}
	pts := cudalegacy.ProjectPoints(
		[][3]float64{{0, 0, 10}},
		[3]float64{}, [3]float64{},
		K, nil)
	fmt.Printf("%.0f %.0f\n", pts[0][0], pts[0][1])
	// Output: 320 240
}

// ExampleCompactPoints drops the masked-out correspondences from two parallel
// point sets.
func ExampleCompactPoints() {
	p0 := [][2]float64{{0, 0}, {1, 1}, {2, 2}}
	p1 := [][2]float64{{5, 5}, {6, 6}, {7, 7}}
	mask := []uint8{1, 0, 1}

	o0, o1 := cudalegacy.CompactPoints(p0, p1, mask)
	fmt.Println(len(o0), o0, o1)
	// Output: 2 [[0 0] [2 2]] [[5 5] [7 7]]
}

// ExampleImagePyramid builds a three-level Gaussian pyramid and reads back the
// sizes of its levels.
func ExampleImagePyramid() {
	img := cv.NewMat(32, 32, 1)
	pyr := cudalegacy.NewImagePyramid(cudalegacy.GpuMatFromMat(img), 3, nil)

	for i := 0; i < pyr.NumLayers(); i++ {
		r, c := pyr.Layer(i).Size()
		fmt.Printf("level %d: %dx%d\n", i, r, c)
	}
	// Output:
	// level 0: 32x32
	// level 1: 16x16
	// level 2: 8x8
}
