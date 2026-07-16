package hfs_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/hfs"
)

// Example runs the full HFS pipeline on a synthetic four-region image: it builds
// a segmenter for the image size, segments it on the CPU, and prints how many
// regions the hierarchical merge recovered.
func Example() {
	// A 64×64 RGB image split into four solid-colour quadrants.
	const size = 64
	img := cv.NewMat(size, size, 3)
	colors := [4][3]uint8{{220, 30, 30}, {30, 200, 30}, {30, 30, 220}, {230, 220, 40}}
	half := size / 2
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			q := 0
			if x >= half {
				q++
			}
			if y >= half {
				q += 2
			}
			img.SetPixel(y, x, colors[q][:])
		}
	}

	seg := hfs.CreateWithDefaults(img.Rows, img.Cols)
	drawn := seg.PerformSegmentCpu(img, true)

	fmt.Printf("regions=%d output=%dx%dx%d\n", seg.NumSegments(), drawn.Rows, drawn.Cols, drawn.Channels)
	// Output: regions=4 output=64x64x3
}
