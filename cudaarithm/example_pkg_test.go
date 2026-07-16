package cudaarithm_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudaarithm"
)

// Example shows the end-to-end cudaarithm flow: upload two host matrices into
// device matrices, add them with saturation, threshold the sum, then download
// both results back to host memory.
func Example() {
	a := cv.NewMat(1, 4, 1)
	copy(a.Data, []uint8{10, 90, 130, 250})
	b := cv.NewMat(1, 4, 1)
	copy(b.Data, []uint8{5, 30, 40, 10})

	sum := cudaarithm.Add(cudaarithm.NewGpuMat(a), cudaarithm.NewGpuMat(b))
	mask, _ := cudaarithm.Threshold(sum, 120, 255, cv.ThreshBinary)

	fmt.Println(ints(sum.Download().Data))
	fmt.Println(ints(mask.Download().Data))
	// Output:
	// [15 120 170 255]
	// [0 0 255 255]
}
