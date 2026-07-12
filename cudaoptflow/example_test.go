package cudaoptflow

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// roundMean prints the interior mean flow rounded to whole pixels, which is
// stable across platforms for the deterministic synthetic frames used here.
func roundMean(u, v float64) string {
	return fmt.Sprintf("%.0f %.0f", math.Round(u), math.Round(v))
}

func ExampleFarnebackOpticalFlow() {
	prev, next := denseGpuPair(64, 64, 2, 1)
	flow := NewFarnebackOpticalFlow(4, 4).Calc(prev, next, NewStream())
	u, v := flow.MeanFlow(8)
	fmt.Println(roundMean(u, v))
	// Output: 2 1
}

func ExampleOpticalFlowDualTVL1() {
	prev, next := denseGpuPair(64, 64, 3, 2)
	flow := NewOpticalFlowDualTVL1().Calc(prev, next, nil)
	u, v := flow.MeanFlow(8)
	fmt.Println(roundMean(u, v))
	// Output: 3 2
}

func ExampleBroxOpticalFlow() {
	prev, next := denseGpuPair(64, 64, 3, 2)
	flow := NewBroxOpticalFlow(0.2, 50, 0.5, 3, 12, 20).Calc(prev, next, nil)
	u, v := flow.MeanFlow(8)
	fmt.Println(roundMean(u, v))
	// Output: 3 2
}

func ExampleNvidiaHWOpticalFlow() {
	// NvidiaHWOpticalFlow mirrors the OpenCV class that drives the NVIDIA
	// hardware optical-flow accelerator; here it computes a real dense flow on
	// the CPU with Dense Inverse Search.
	prev, next := denseGpuPair(64, 64, 2, 3)
	flow := NewNvidiaHWOpticalFlow(4, 2, 3).Calc(prev, next, nil)
	u, v := flow.MeanFlow(8)
	fmt.Println(roundMean(u, v))
	// Output: 2 3
}

func ExampleSparsePyrLKOpticalFlow() {
	prev, next := denseGpuPair(80, 80, 3, 2)
	pts := []cv.Point{{X: 30, Y: 30}, {X: 50, Y: 50}}
	nextPts, status, _ := NewSparsePyrLKOpticalFlow(15, 2, 30).Calc(prev, next, pts, NewStream())
	for i := range pts {
		fmt.Printf("pt%d status=%d -> (%d,%d)\n", i, status[i], nextPts[i].X, nextPts[i].Y)
	}
	// Output:
	// pt0 status=1 -> (33,32)
	// pt1 status=1 -> (53,52)
}

func ExampleFlowToGpuMat() {
	prev, next := denseGpuPair(64, 64, 2, 1)
	flow := NewFarnebackOpticalFlow(4, 4).Calc(prev, next, nil)

	// Pack the float flow into an OpenCV-style two-channel uint8 GpuMat and
	// decode it back.
	packed := FlowToGpuMat(flow)
	r, c := packed.Size()
	decoded := packed.ToFlowField()
	u, v := decoded.MeanFlow(8)
	fmt.Printf("%dx%d chans=%d mean=%s\n", r, c, packed.Channels(), roundMean(u, v))
	// Output: 64x64 chans=2 mean=2 1
}
