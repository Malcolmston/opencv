package tracking

import "testing"

func TestConstantVelocityKalmanTracksLine(t *testing.T) {
	// True motion: start at (0,0), velocity (1, 0.5) per step, no noise.
	kf := NewConstantVelocityKalman2D(0, 0, 1, 0.01, 1.0)
	var state []float64
	for step := 1; step <= 30; step++ {
		kf.Predict(nil)
		tx := float64(step) * 1.0
		ty := float64(step) * 0.5
		state = kf.Correct([]float64{tx, ty})
	}
	// state = (x, y, vx, vy).
	requireTrue(t, approx(state[2], 1.0, 0.1), "vx = %v, want ~1", state[2])
	requireTrue(t, approx(state[3], 0.5, 0.1), "vy = %v, want ~0.5", state[3])
	requireTrue(t, approx(state[0], 30.0, 0.5), "x = %v, want ~30", state[0])
	requireTrue(t, approx(state[1], 15.0, 0.5), "y = %v, want ~15", state[1])

	// One more prediction should extrapolate forward along the velocity.
	pred := kf.Predict(nil)
	requireTrue(t, approx(pred[0], 31.0, 0.5), "predicted x = %v, want ~31", pred[0])
	requireTrue(t, approx(pred[1], 15.5, 0.5), "predicted y = %v, want ~15.5", pred[1])
}

func TestKalmanScalarConverges(t *testing.T) {
	// 1-D static model: state and measurement are the same scalar.
	kf := NewKalmanFilter(1, 1, 0)
	kf.TransitionMatrix = MatrixFromRows([][]float64{{1}})
	kf.MeasurementMatrix = MatrixFromRows([][]float64{{1}})
	kf.ProcessNoiseCov = MatrixFromRows([][]float64{{1e-5}})
	kf.MeasurementNoiseCov = MatrixFromRows([][]float64{{0.1}})
	kf.ErrorCovPost = MatrixFromRows([][]float64{{1}})
	kf.StatePost = MatrixFromRows([][]float64{{0}})

	var s []float64
	for i := 0; i < 50; i++ {
		kf.Predict(nil)
		s = kf.Correct([]float64{5})
	}
	requireTrue(t, approx(s[0], 5, 0.05), "converged estimate = %v, want ~5", s[0])
}

func TestKalmanControlInput(t *testing.T) {
	// State advances by a control input each step, no dynamics of its own.
	kf := NewKalmanFilter(1, 1, 1)
	kf.TransitionMatrix = MatrixFromRows([][]float64{{1}})
	kf.ControlMatrix = MatrixFromRows([][]float64{{1}})
	kf.MeasurementMatrix = MatrixFromRows([][]float64{{1}})
	pred := kf.Predict([]float64{2})
	requireTrue(t, approx(pred[0], 2, 1e-9), "control-driven prediction = %v, want 2", pred[0])
}
