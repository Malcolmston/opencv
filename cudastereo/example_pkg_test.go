package cudastereo_test

import (
	"fmt"

	"github.com/malcolmston/opencv/cudastereo"
)

// Example builds a rectified stereo pair whose right-hand region is shifted by a
// known disparity, uploads both images to device matrices, runs block-matching
// stereo, and downloads the disparity map to read back the recovered value.
func Example() {
	left, right := makePair(64, 24, 8)
	l := cudastereo.NewGpuMatFromMat(left)
	r := cudastereo.NewGpuMatFromMat(right)

	bm := cudastereo.CreateStereoBM(16, 7)
	disp := bm.Compute(l, r, nil).Download()

	fmt.Println("disparity:", disp.Data[12*disp.Cols+50])
	// Output: disparity: 8
}
