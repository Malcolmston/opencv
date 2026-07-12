package stereo_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/stereo"
)

// buildPair makes a tiny rectified pair whose right half is shifted right by a
// known disparity, so the matchers should recover that disparity there.
func buildPair(w, h, disp int) (left, right *cv.Mat) {
	tex := func(x, y int) uint8 { return uint8((x*167 + y*83 + (x*x)%91) % 256) }
	right = cv.NewMat(h, w, 1)
	left = cv.NewMat(h, w, 1)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			right.Data[y*w+x] = tex(x, y)
			sx := x
			if x >= w/2 {
				sx = x - disp
			}
			if sx < 0 {
				sx = 0
			}
			left.Data[y*w+x] = tex(sx, y)
		}
	}
	return left, right
}

// ExampleStereoBM recovers the disparity of a shifted region with block matching.
func ExampleStereoBM() {
	left, right := buildPair(64, 24, 8)
	d := stereo.StereoBM{NumDisparities: 16, BlockSize: 7}.Compute(left, right)
	fmt.Println(d.Data[12*d.Cols+50])
	// Output: 8
}

// ExampleStereoSGBM recovers the same disparity with four-path aggregation.
func ExampleStereoSGBM() {
	left, right := buildPair(64, 24, 8)
	d := stereo.StereoSGBM{NumDisparities: 16, BlockSize: 5}.Compute(left, right)
	fmt.Println(d.Data[12*d.Cols+50])
	// Output: 8
}

// ExampleReprojectImageTo3D shows that a constant disparity reprojects to a
// constant depth through a rectification matrix Q.
func ExampleReprojectImageTo3D() {
	d := cv.NewMat(2, 3, 1)
	d.SetTo(8)
	Q := [4][4]float64{
		{1, 0, 0, -1},
		{0, 1, 0, -1},
		{0, 0, 0, 500},
		{0, 0, 10, 0},
	}
	pts := stereo.ReprojectImageTo3D(d, Q)
	fmt.Printf("Z=%.2f\n", pts[0][2])
	// Output: Z=6.25
}

// ExampleFilterSpecklesDisparity removes a small isolated blob while keeping a
// large connected region.
func ExampleFilterSpecklesDisparity() {
	d := cv.NewMat(6, 6, 1)
	for i := 0; i < 6*4; i++ { // top four rows: a big blob of disparity 12
		d.Data[i] = 12
	}
	d.Data[5*6+5] = 40 // a lone speckle

	stereo.FilterSpecklesDisparity(d, stereo.InvalidDisparity, 4, 4)
	fmt.Println(d.Data[0], d.Data[5*6+5])
	// Output: 12 0
}
