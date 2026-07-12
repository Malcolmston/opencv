package cudaarithm_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudaarithm"
)

// ExampleAdd shows the upload / operate / download flow that mirrors OpenCV's
// cuda API. No GPU is involved; the "device" matrices live in host memory.
func ExampleAdd() {
	a := cv.NewMat(1, 3, 1)
	copy(a.Data, []uint8{10, 20, 250})
	b := cv.NewMat(1, 3, 1)
	copy(b.Data, []uint8{5, 5, 10})

	ga := cudaarithm.NewGpuMat(a)
	gb := cudaarithm.NewGpuMat(b)

	sum := cudaarithm.Add(ga, gb).Download() // 250+10 saturates to 255
	fmt.Println(ints(sum.Data))
	// Output: [15 25 255]
}

// ExampleStream shows that a Stream can be threaded through operations for
// source compatibility; work still completes synchronously.
func ExampleStream() {
	src := cv.NewMat(1, 4, 1)
	copy(src.Data, []uint8{0, 4, 9, 16})

	stream := cudaarithm.NewStream()
	roots := cudaarithm.Sqrt(cudaarithm.NewGpuMat(src), stream)
	stream.WaitForCompletion()

	fmt.Println(ints(roots.Download().Data))
	// Output: [0 2 3 4]
}

// ExampleMinMaxLoc reports the extreme sample values and their locations.
func ExampleMinMaxLoc() {
	src := cv.NewMat(2, 3, 1)
	copy(src.Data, []uint8{5, 9, 2, 7, 1, 8})

	lo, hi, minX, minY, maxX, maxY := cudaarithm.MinMaxLoc(cudaarithm.NewGpuMat(src))
	fmt.Printf("min %v at (%d,%d), max %v at (%d,%d)\n", lo, minX, minY, hi, maxX, maxY)
	// Output: min 1 at (1,1), max 9 at (1,0)
}

// ExampleGemm multiplies two small matrices with saturation.
func ExampleGemm() {
	a := cv.NewMat(2, 2, 1)
	copy(a.Data, []uint8{1, 2, 0, 1})
	b := cv.NewMat(2, 2, 1)
	copy(b.Data, []uint8{1, 1, 0, 1})

	prod := cudaarithm.Gemm(cudaarithm.NewGpuMat(a), cudaarithm.NewGpuMat(b), 1, nil, 0)
	fmt.Println(ints(prod.Download().Data))
	// Output: [1 3 0 1]
}

// ints converts a byte slice to an int slice so the examples print numeric
// values without staticcheck suggesting a string conversion.
func ints(b []uint8) []int {
	out := make([]int, len(b))
	for i, v := range b {
		out[i] = int(v)
	}
	return out
}
