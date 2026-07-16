package cudabgsegm_test

import (
	"fmt"

	"github.com/malcolmston/opencv/cudabgsegm"
)

// Example trains the CUDA-style MOG2 background model on a static background,
// then applies a frame containing a bright 3x3 moving blob and reports the
// foreground area of the downloaded mask. Frames and masks travel as GpuMat
// values with a Stream threaded through each call, mirroring OpenCV's CUDA API.
func Example() {
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
