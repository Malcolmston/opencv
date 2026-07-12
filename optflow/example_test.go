package optflow_test

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/optflow"
)

// ExampleCalcOpticalFlowDenseHS recovers a uniform one-pixel horizontal shift
// from a textured pair as a dense Horn-Schunck flow field and reports its
// interior mean displacement.
func ExampleCalcOpticalFlowDenseHS() {
	rows, cols := 40, 40
	prev := cv.NewMat(rows, cols, 1)
	next := cv.NewMat(rows, cols, 1)
	pattern := func(x, y float64) uint8 {
		v := 128 + 80*math.Sin(2*math.Pi*x/16) + 40*math.Cos(2*math.Pi*y/13)
		if v < 0 {
			v = 0
		}
		if v > 255 {
			v = 255
		}
		return uint8(math.Round(v))
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			prev.Set(y, x, 0, pattern(float64(x), float64(y)))
			next.Set(y, x, 0, pattern(float64(x-1), float64(y))) // shifted right by 1
		}
	}
	flow := optflow.CalcOpticalFlowDenseHS(prev, next, 15, 200)
	u, v := flow.MeanFlow(6)
	// Snap near-zero components to avoid a signed-zero print (-0.0).
	if math.Abs(v) < 0.05 {
		v = 0
	}
	fmt.Printf("mean flow = (%.1f, %.1f)\n", u, v)
	// Output: mean flow = (1.0, 0.0)
}

// ExampleFlowToColor visualises a flow field and reports the channel count of
// the Middlebury colour-wheel image.
func ExampleFlowToColor() {
	flow := optflow.NewFlowField(10, 10)
	for y := 0; y < 10; y++ {
		for x := 0; x < 10; x++ {
			flow.Set(y, x, float64(x-5), float64(y-5))
		}
	}
	img := optflow.FlowToColor(flow)
	fmt.Printf("%dx%d, %d channels\n", img.Rows, img.Cols, img.Channels)
	// Output: 10x10, 3 channels
}

// ExampleWarpByFlow shows that warping the previous frame by a constant flow
// reproduces a shifted next frame. With u = 1 the output samples img at column
// x-1, so column 3 takes the value from img's column 2.
func ExampleWarpByFlow() {
	img := cv.NewMat(8, 8, 1)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			img.Set(y, x, 0, uint8(x*20))
		}
	}
	flow := optflow.NewFlowField(8, 8)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			flow.Set(y, x, 1, 0) // shift right by one
		}
	}
	warped := optflow.WarpByFlow(img, flow)
	fmt.Printf("%d\n", warped.At(0, 3, 0))
	// Output: 40
}
