package cudacore_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudacore"
)

// Example shows the core cudacore device workflow: query the CUDA-enabled device
// count (zero in this CPU-backed build), upload a host matrix into a device
// matrix, and download it back unchanged.
func Example() {
	fmt.Println("cuda devices:", cudacore.GetCudaEnabledDeviceCount())

	src := cv.NewMat(1, 3, 1)
	copy(src.Data, []uint8{10, 20, 30})

	var g cudacore.GpuMat
	g.Upload(src)
	out := g.Download()

	fmt.Println("round-trip:", ints(out.Data))
	// Output:
	// cuda devices: 0
	// round-trip: [10 20 30]
}
