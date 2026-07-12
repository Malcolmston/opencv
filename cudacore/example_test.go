package cudacore_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudacore"
)

// ExampleGpuMat_Upload shows the upload / operate / download flow that mirrors
// OpenCV's cuda API. No GPU is involved; the "device" matrix lives in host
// memory, so the round-trip preserves samples exactly.
func ExampleGpuMat_Upload() {
	src := cv.NewMat(1, 3, 1)
	copy(src.Data, []uint8{10, 20, 30})

	var g cudacore.GpuMat
	g.Upload(src)
	out := g.Download()

	fmt.Println(ints(out.Data))
	// Output: [10 20 30]
}

// ExampleGpuMat_ConvertTo scales samples with saturation, the CV_8U analogue of
// cv::cuda::GpuMat::convertTo.
func ExampleGpuMat_ConvertTo() {
	src := cv.NewMat(1, 4, 1)
	copy(src.Data, []uint8{0, 50, 100, 200})

	g := cudacore.NewGpuMat(src)
	out := g.ConvertTo(1.5, 0).Download() // 200*1.5 = 300 saturates to 255

	fmt.Println(ints(out.Data))
	// Output: [0 75 150 255]
}

// ExampleGpuMat_CopyMakeBorder pads a 1x1 matrix with a constant border.
func ExampleGpuMat_CopyMakeBorder() {
	g := cudacore.NewGpuMat(cv.NewMat(1, 1, 1))
	g.SetTo(cv.NewScalar(5))

	out := g.CopyMakeBorder(1, 1, 1, 1, cudacore.BorderConstant, cv.NewScalar(0))
	rows, cols := out.Size()
	fmt.Printf("%dx%d\n", rows, cols)
	fmt.Println(ints(out.Download().Data))
	// Output:
	// 3x3
	// [0 0 0 0 5 0 0 0 0]
}

// ExampleGetCudaEnabledDeviceCount shows that this build reports no CUDA device.
func ExampleGetCudaEnabledDeviceCount() {
	fmt.Println(cudacore.GetCudaEnabledDeviceCount())
	// Output: 0
}

// ExampleStream shows that a Stream and Event can be threaded through code for
// source compatibility; every operation still completes synchronously.
func ExampleStream() {
	stream := cudacore.NewStream()
	done := cudacore.NewEvent()

	g := cudacore.NewGpuMat(cv.NewMat(2, 2, 1))
	g.SetTo(cv.NewScalar(7))
	done.Record(stream)
	stream.WaitForCompletion()

	fmt.Println(ints(g.Download().Data), cudacore.ElapsedTime(done, done))
	// Output: [7 7 7 7] 0
}

// ints converts a byte slice to an int slice so examples print numeric values
// without staticcheck suggesting a string conversion.
func ints(b []uint8) []int {
	out := make([]int, len(b))
	for i, v := range b {
		out[i] = int(v)
	}
	return out
}
