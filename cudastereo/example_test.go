package cudastereo_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/cudastereo"
)

// makePair builds a tiny rectified pair whose right half is shifted by a known
// disparity, so a matcher should recover that disparity there.
func makePair(w, h, disp int) (left, right *cv.Mat) {
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

// ExampleStereoBM recovers the disparity of a shifted region on the (CPU-backed)
// device, showing the upload / compute / download round-trip.
func ExampleStereoBM() {
	left, right := makePair(64, 24, 8)
	l := cudastereo.NewGpuMatFromMat(left)
	r := cudastereo.NewGpuMatFromMat(right)

	bm := cudastereo.CreateStereoBM(16, 7)
	disp := bm.Compute(l, r, nil).Download()

	fmt.Println(disp.Data[12*disp.Cols+50])
	// Output: 8
}

// ExampleStereoBeliefPropagation recovers the same disparity with hierarchical
// loopy belief propagation.
func ExampleStereoBeliefPropagation() {
	left, right := makePair(64, 24, 8)
	l := cudastereo.NewGpuMatFromMat(left)
	r := cudastereo.NewGpuMatFromMat(right)

	bp := cudastereo.CreateStereoBeliefPropagation(16, 5, 3)
	disp := bp.Compute(l, r, nil).Download()

	fmt.Println(disp.Data[12*disp.Cols+50])
	// Output: 8
}

// ExampleStereoConstantSpaceBP recovers the disparity with constant-space belief
// propagation, which keeps only a few disparity hypotheses per pixel.
func ExampleStereoConstantSpaceBP() {
	left, right := makePair(64, 24, 8)
	l := cudastereo.NewGpuMatFromMat(left)
	r := cudastereo.NewGpuMatFromMat(right)

	csbp := cudastereo.CreateStereoConstantSpaceBP(16, 8, 4, 4)
	disp := csbp.Compute(l, r, nil).Download()

	fmt.Println(disp.Data[12*disp.Cols+50])
	// Output: 8
}

// ExampleReprojectImageTo3D shows that a constant disparity reprojects to a
// constant depth through a rectification matrix Q.
func ExampleReprojectImageTo3D() {
	d := cv.NewMat(2, 3, 1)
	d.SetTo(8)
	g := cudastereo.NewGpuMatFromMat(d)

	Q := [4][4]float64{
		{1, 0, 0, -1},
		{0, 1, 0, -1},
		{0, 0, 0, 500},
		{0, 0, 10, 0},
	}
	pts := cudastereo.ReprojectImageTo3D(g, Q, nil)
	fmt.Printf("Z=%.2f\n", pts[0][2])
	// Output: Z=6.25
}

// ExampleStreo_estimateRecommendedParams shows the belief-propagation parameter
// heuristic for a given image size.
func ExampleStereoBeliefPropagation_estimateRecommendedParams() {
	bp := cudastereo.CreateStereoBeliefPropagation(0, 0, 0)
	ndisp, iters, _ := bp.EstimateRecommendedParams(640, 480)
	fmt.Println(ndisp, iters)
	// Output: 160 8
}
