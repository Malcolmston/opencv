package cudabgsegm_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/bgsegm"
	"github.com/malcolmston/opencv/cudabgsegm"
)

// fill returns a rows×cols single-channel Mat filled with val.
func fill(rows, cols int, val uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	m.SetTo(val)
	return m
}

// fg counts foreground pixels in a mask GpuMat.
func fg(mask *cudabgsegm.GpuMat) int {
	n := 0
	for _, v := range mask.Mat.Data {
		if v == bgsegm.ForegroundValue {
			n++
		}
	}
	return n
}

// ExampleBackgroundSubtractorMOG2 warms the CUDA-style MOG2 model on a static
// background, then applies a frame with a bright 3×3 moving blob and reports the
// foreground area. Frames and masks travel as GpuMat values and a Stream is
// threaded through each call, mirroring the OpenCV CUDA API.
func ExampleBackgroundSubtractorMOG2() {
	sub := cudabgsegm.CreateBackgroundSubtractorMOG2(10, 16, false)
	stream := cudabgsegm.NewStream()

	for i := 0; i < 30; i++ {
		sub.Apply(cudabgsegm.GpuMatFromMat(fill(8, 8, 40)), -1, stream)
	}

	frame := fill(8, 8, 40)
	for y := 2; y < 5; y++ {
		for x := 3; x < 6; x++ {
			frame.Set(y, x, 0, 220)
		}
	}
	mask := sub.Apply(cudabgsegm.GpuMatFromMat(frame), -1, stream)
	fmt.Println("foreground pixels:", fg(mask))
	// Output: foreground pixels: 9
}

// ExampleBackgroundSubtractorMOG shows the upload/download idiom: the frame is
// uploaded to a GpuMat, classified, and the mask downloaded back to host memory.
func ExampleBackgroundSubtractorMOG() {
	sub := cudabgsegm.CreateBackgroundSubtractorMOG(10, 5, 0.7, 15)
	stream := cudabgsegm.NewStream()

	device := cudabgsegm.NewGpuMat()
	for i := 0; i < 30; i++ {
		device.Upload(fill(8, 8, 40), stream)
		sub.Apply(device, -1, stream)
	}

	frame := fill(8, 8, 40)
	for y := 2; y < 5; y++ {
		for x := 3; x < 6; x++ {
			frame.Set(y, x, 0, 220)
		}
	}
	device.Upload(frame, stream)
	host := sub.Apply(device, -1, stream).Download(stream)

	count := 0
	for _, v := range host.Data {
		if v == bgsegm.ForegroundValue {
			count++
		}
	}
	fmt.Println("foreground pixels:", count)
	// Output: foreground pixels: 9
}

// ExampleBackgroundSubtractorGMG classifies the same blob with the GMG model
// after its initial learning period.
func ExampleBackgroundSubtractorGMG() {
	sub := cudabgsegm.CreateBackgroundSubtractorGMG(15, 0.8)
	stream := cudabgsegm.NewStream()

	for i := 0; i < 30; i++ {
		sub.Apply(cudabgsegm.GpuMatFromMat(fill(8, 8, 40)), -1, stream)
	}

	frame := fill(8, 8, 40)
	for y := 2; y < 5; y++ {
		for x := 3; x < 6; x++ {
			frame.Set(y, x, 0, 220)
		}
	}
	mask := sub.Apply(cudabgsegm.GpuMatFromMat(frame), -1, stream)
	fmt.Println("foreground pixels:", fg(mask))
	// Output: foreground pixels: 9
}
