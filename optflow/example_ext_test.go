package optflow_test

import (
	"bytes"
	"fmt"
	"image"
	"math"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/optflow"
)

// makeShift builds a textured grayscale pair where next is prev translated by
// (dx, dy), used by the runnable examples below.
func makeShift(rows, cols int, dx, dy float64) (prev, next *cv.Mat) {
	prev = cv.NewMat(rows, cols, 1)
	next = cv.NewMat(rows, cols, 1)
	pattern := func(x, y float64) uint8 {
		v := 128 + 60*math.Sin(2*math.Pi*x/17) + 50*math.Cos(2*math.Pi*y/13)
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
			next.Set(y, x, 0, pattern(float64(x)-dx, float64(y)-dy))
		}
	}
	return prev, next
}

// ExampleCalcOpticalFlowDenseTVL1 recovers a uniform two-pixel horizontal shift
// as a dense TV-L1 flow field and reports its interior mean displacement.
func ExampleCalcOpticalFlowDenseTVL1() {
	prev, next := makeShift(64, 64, 2, 0)
	p := optflow.DefaultTVL1Params()
	p.Scales = 4
	flow := optflow.CalcOpticalFlowDenseTVL1(prev, next, p)
	u, v := flow.MeanFlow(10)
	if math.Abs(v) < 0.1 {
		v = 0
	}
	fmt.Printf("mean flow = (%.1f, %.1f)\n", u, v)
	// Output: mean flow = (2.0, 0.0)
}

// ExampleDualTVL1OpticalFlow shows the stateful estimator object producing the
// same result as the package-level function.
func ExampleDualTVL1OpticalFlow() {
	prev, next := makeShift(48, 48, 1, 0)
	est := optflow.NewDualTVL1OpticalFlow(optflow.DefaultTVL1Params())
	flow := est.Calc(prev, next)
	u, _ := flow.MeanFlow(8)
	fmt.Printf("u ≈ %.1f\n", u)
	// Output: u ≈ 1.0
}

// ExampleWriteFlow round-trips a flow field through the Middlebury .flo binary
// format in memory.
func ExampleWriteFlow() {
	flow := optflow.NewFlowField(3, 3)
	flow.Set(1, 1, 2.5, -1.25)

	var buf bytes.Buffer
	if err := optflow.WriteFlow(&buf, flow); err != nil {
		fmt.Println("write:", err)
		return
	}
	got, err := optflow.ReadFlow(&buf)
	if err != nil {
		fmt.Println("read:", err)
		return
	}
	u, v := got.At(1, 1)
	fmt.Printf("%dx%d, centre = (%.2f, %.2f)\n", got.Rows, got.Cols, u, v)
	// Output: 3x3, centre = (2.50, -1.25)
}

// ExampleAverageEndpointError scores an estimated flow against a known
// ground-truth translation.
func ExampleAverageEndpointError() {
	est := optflow.NewFlowField(2, 2)
	gt := optflow.NewFlowField(2, 2)
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			gt.Set(y, x, 1, 0)
			est.Set(y, x, 1, 0)
		}
	}
	est.Set(0, 0, 1, 1) // one pixel off by 1 in v
	fmt.Printf("AEE = %.2f\n", optflow.AverageEndpointError(est, gt))
	// Output: AEE = 0.25
}

// ExampleInterpolateFlow densifies two sparse samples with inverse-distance
// (Shepard) weighting, reproducing each sample exactly at its own location.
func ExampleInterpolateFlow() {
	pts := []optflow.PointF{{X: 0, Y: 0}, {X: 9, Y: 9}}
	vecs := []optflow.PointF{{X: 2, Y: 0}, {X: -2, Y: 0}}
	flow := optflow.InterpolateFlow(10, 10, pts, vecs, 0)
	u0, _ := flow.At(0, 0)
	u1, _ := flow.At(9, 9)
	fmt.Printf("corners: %.1f, %.1f\n", u0, u1)
	// Output: corners: 2.0, -2.0
}

// ExampleCalcOpticalFlowSparseRLOF tracks a single point through a translation.
func ExampleCalcOpticalFlowSparseRLOF() {
	prev, next := makeShift(48, 48, 2, 0)
	pts := []image.Point{{X: 24, Y: 24}}
	nextPts, status := optflow.CalcOpticalFlowSparseRLOF(prev, next, pts, optflow.DefaultRLOFParams())
	du := nextPts[0].X - 24
	fmt.Printf("tracked=%v du≈%.1f\n", status[0], du)
	// Output: tracked=true du≈2.0
}
