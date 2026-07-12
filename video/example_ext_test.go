package video

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// exampleTexture is a small deterministic non-periodic texture shifted by
// (ox, oy), used by the runnable examples below.
func exampleTexture(rows, cols int, ox, oy float64) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	kx := 2 * math.Pi / float64(cols) * 1.3
	ky := 2 * math.Pi / float64(rows) * 1.1
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			fx := float64(x) - ox
			fy := float64(y) - oy
			v := 128 + 70*math.Sin(kx*fx+0.5) + 55*math.Cos(ky*fy+1.1) + 45*math.Sin(0.5*(kx*fx+ky*fy))
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			m.Set(y, x, 0, uint8(v+0.5))
		}
	}
	return m
}

// ExampleFindTransformECC recovers a known translation between two frames by
// maximising the enhanced correlation coefficient.
func ExampleFindTransformECC() {
	tmpl := exampleTexture(64, 64, 0, 0)
	input := exampleTexture(64, 64, 3, 2)
	cc, warp := FindTransformECC(tmpl, input, nil, MotionTranslation, NewTermCriteria(60, 1e-8))
	fmt.Printf("cc>0.99=%v tx=%.1f ty=%.1f\n", cc > 0.99, math.Round(warp[0][2]), math.Round(warp[1][2]))
	// Output: cc>0.99=true tx=3.0 ty=2.0
}

// ExampleEstimateAffinePartial2D recovers a similarity transform (scale,
// rotation, translation) from a handful of point correspondences.
func ExampleEstimateAffinePartial2D() {
	from := []PointF{{0, 0}, {10, 0}, {0, 10}, {10, 10}}
	// Apply scale 2, rotation 90 degrees CCW, translation (5, 1).
	to := []PointF{{5, 1}, {5, 21}, {-15, 1}, {-15, 21}}
	tf, ok := EstimateAffinePartial2D(from, to)
	fmt.Printf("ok=%v scale=%.1f angleDeg=%.0f t=(%.0f,%.0f)\n",
		ok, tf.Scale, tf.Angle*180/math.Pi, tf.Tx, tf.Ty)
	// Output: ok=true scale=2.0 angleDeg=90 t=(5,1)
}

// ExampleMeanShift recentres a search window on the mode of a probability image.
func ExampleMeanShift() {
	prob := cv.NewFloatMat(60, 60)
	for y := 0; y < 60; y++ {
		for x := 0; x < 60; x++ {
			dx := float64(x-40) / 5
			dy := float64(y-20) / 5
			prob.Data[y*60+x] = math.Exp(-0.5 * (dx*dx + dy*dy))
		}
	}
	_, w := MeanShift(prob, cv.Rect{X: 5, Y: 5, Width: 16, Height: 16}, NewTermCriteria(50, 0.05))
	cx := w.X + w.Width/2
	cy := w.Y + w.Height/2
	fmt.Printf("near mode (40,20): %v\n", math.Abs(float64(cx-40)) <= 1 && math.Abs(float64(cy-20)) <= 1)
	// Output: near mode (40,20): true
}

// ExampleDISOpticalFlow computes a dense flow field and reports the interior
// mean displacement for a uniformly shifted frame.
func ExampleDISOpticalFlow() {
	prev := exampleTexture(80, 80, 0, 0)
	next := exampleTexture(80, 80, 2, 0)
	dis := NewDISOpticalFlow(DISPresetMedium)
	dis.FinestScale = 0
	flow := dis.Calc(prev, next)
	dx, dy := flow.MeanFlow(12)
	fmt.Printf("mean flow ~ (%.0f, %.0f)\n", math.Round(dx), math.Round(dy))
	// Output: mean flow ~ (2, 0)
}

// ExampleKalmanFilter_PredictControl drives a Kalman filter with a control input
// (constant acceleration) and recovers the position and velocity.
func ExampleKalmanFilter_PredictControl() {
	kf := NewKalmanFilter(2, 1)
	kf.SetTransitionMatrix([][]float64{{1, 1}, {0, 1}})
	kf.SetMeasurementMatrix([][]float64{{1, 0}})
	kf.SetProcessNoiseCov([][]float64{{1e-5, 0}, {0, 1e-5}})
	kf.SetMeasurementNoiseCov([][]float64{{0.05}})
	kf.SetControlMatrix([][]float64{{0.5}, {1}})
	kf.SetState([]float64{0, 0})

	pos, vel := 0.0, 0.0
	for k := 0; k < 30; k++ {
		kf.PredictControl([]float64{1})
		pos += vel + 0.5
		vel++
		kf.Correct([]float64{pos})
	}
	fmt.Printf("pos~%.0f vel~%.0f\n", math.Round(kf.StatePost[0]), math.Round(kf.StatePost[1]))
	// Output: pos~450 vel~30
}
