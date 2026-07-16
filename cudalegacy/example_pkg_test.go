package cudalegacy_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudalegacy"
)

// Example uploads a host image to a device matrix and builds a three-level
// Gaussian image pyramid on it, reading back the size of each level — a
// representative cudalegacy multi-resolution flow.
func Example() {
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
