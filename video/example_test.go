package video

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// ExampleBuildOpticalFlowPyramid shows the coarse-to-fine level sizes of a
// Gaussian pyramid; each level halves both dimensions.
func ExampleBuildOpticalFlowPyramid() {
	img := cv.NewMat(64, 48, 1)
	pyr := BuildOpticalFlowPyramid(img, 3)
	for i, lv := range pyr {
		fmt.Printf("level %d: %dx%d\n", i, lv.Rows, lv.Cols)
	}
	// Output:
	// level 0: 64x48
	// level 1: 32x24
	// level 2: 16x12
	// level 3: 8x6
}

// ExampleKalmanFilter tracks a constant-velocity trajectory. After feeding a
// few exact position measurements the filter recovers both the position and the
// hidden velocity component of the state.
func ExampleKalmanFilter() {
	kf := NewKalmanFilter(2, 1)
	kf.TransitionMatrix = [][]float64{{1, 1}, {0, 1}}
	kf.MeasurementMatrix = [][]float64{{1, 0}}
	kf.ProcessNoiseCov = [][]float64{{1e-4, 0}, {0, 1e-4}}
	kf.MeasurementNoiseCov = [][]float64{{0.1}}

	for k := 1; k <= 20; k++ {
		kf.Predict()
		kf.Correct([]float64{2 * float64(k)}) // position = 2*k
	}
	fmt.Printf("pos=%.1f vel=%.1f\n", kf.StatePost[0], kf.StatePost[1])
	// Output: pos=40.0 vel=2.0
}

// ExampleCalcOpticalFlowFarneback recovers a uniform one-pixel horizontal shift
// as a dense flow field and reports its interior mean displacement.
func ExampleCalcOpticalFlowFarneback() {
	// A vertical-stripe pattern shifted right by one pixel.
	prev := cv.NewMat(24, 24, 1)
	next := cv.NewMat(24, 24, 1)
	for y := 0; y < 24; y++ {
		for x := 0; x < 24; x++ {
			prev.Set(y, x, 0, uint8((x*37)%251))
			nx := x - 1
			if nx < 0 {
				nx = 0
			}
			next.Set(y, x, 0, uint8((nx*37)%251))
		}
	}
	flow := CalcOpticalFlowFarneback(prev, next, 2, 2)
	dx, dy := flow.MeanFlow(4)
	fmt.Printf("mean flow = (%.1f, %.1f)\n", dx, dy)
	// Output: mean flow = (1.0, 0.0)
}
