package video

import "fmt"

// PredictControl advances the filter one time step including an external control
// input u (length ControlDim) applied through the control model B. It computes
//
//	x' = A * x + B * u
//	P' = A * P * Aᵀ + Q
//
// updating StatePre and ErrorCovPre and, as with [KalmanFilter.Predict],
// mirroring the prediction into StatePost / ErrorCovPost so a Predict without a
// following Correct still advances the estimate. The returned slice is a fresh
// copy of the predicted state.
//
// A control model must have been installed with [KalmanFilter.SetControlMatrix]
// first; otherwise, or if len(u) != ControlDim, it panics. Passing a nil or
// empty u with no control model is not allowed — use [KalmanFilter.Predict] for
// the control-free case.
func (kf *KalmanFilter) PredictControl(u []float64) []float64 {
	if kf.ControlMatrix == nil || kf.ControlDim == 0 {
		panic("video: PredictControl requires a control model; call SetControlMatrix first")
	}
	if len(u) != kf.ControlDim {
		panic(fmt.Sprintf("video: PredictControl control vector has length %d, want %d", len(u), kf.ControlDim))
	}
	// x' = A x + B u
	kf.StatePre = vecAdd(matVec(kf.TransitionMatrix, kf.StatePost), matVec(kf.ControlMatrix, u))
	// P' = A P Aᵀ + Q
	ap := matMul(kf.TransitionMatrix, kf.ErrorCovPost)
	apat := matMul(ap, transpose(kf.TransitionMatrix))
	kf.ErrorCovPre = matAdd(apat, kf.ProcessNoiseCov)
	kf.StatePost = cloneVec(kf.StatePre)
	kf.ErrorCovPost = cloneMat(kf.ErrorCovPre)
	return cloneVec(kf.StatePre)
}

// SetTransitionMatrix installs the n x n state-transition model A. It panics
// unless a is exactly StateDim x StateDim.
func (kf *KalmanFilter) SetTransitionMatrix(a [][]float64) {
	checkDims("TransitionMatrix", a, kf.StateDim, kf.StateDim)
	kf.TransitionMatrix = cloneMat(a)
}

// SetMeasurementMatrix installs the m x n measurement model H. It panics unless
// h is exactly MeasureDim x StateDim.
func (kf *KalmanFilter) SetMeasurementMatrix(h [][]float64) {
	checkDims("MeasurementMatrix", h, kf.MeasureDim, kf.StateDim)
	kf.MeasurementMatrix = cloneMat(h)
}

// SetProcessNoiseCov installs the n x n process-noise covariance Q. It panics
// unless q is exactly StateDim x StateDim.
func (kf *KalmanFilter) SetProcessNoiseCov(q [][]float64) {
	checkDims("ProcessNoiseCov", q, kf.StateDim, kf.StateDim)
	kf.ProcessNoiseCov = cloneMat(q)
}

// SetMeasurementNoiseCov installs the m x m measurement-noise covariance R. It
// panics unless r is exactly MeasureDim x MeasureDim.
func (kf *KalmanFilter) SetMeasurementNoiseCov(r [][]float64) {
	checkDims("MeasurementNoiseCov", r, kf.MeasureDim, kf.MeasureDim)
	kf.MeasurementNoiseCov = cloneMat(r)
}

// SetControlMatrix installs the n x c control model B used by
// [KalmanFilter.PredictControl] and records ControlDim = c. b must have exactly
// StateDim rows and at least one column; passing nil clears the control model.
func (kf *KalmanFilter) SetControlMatrix(b [][]float64) {
	if b == nil {
		kf.ControlMatrix = nil
		kf.ControlDim = 0
		return
	}
	if len(b) != kf.StateDim || len(b[0]) < 1 {
		panic(fmt.Sprintf("video: ControlMatrix must be %d x c with c >= 1", kf.StateDim))
	}
	cols := len(b[0])
	for i, row := range b {
		if len(row) != cols {
			panic(fmt.Sprintf("video: ControlMatrix row %d has %d columns, want %d", i, len(row), cols))
		}
	}
	kf.ControlMatrix = cloneMat(b)
	kf.ControlDim = cols
}

// SetState overwrites both StatePre and StatePost with the given state vector,
// providing an initial estimate. It panics unless len(x) == StateDim.
func (kf *KalmanFilter) SetState(x []float64) {
	if len(x) != kf.StateDim {
		panic(fmt.Sprintf("video: SetState vector has length %d, want %d", len(x), kf.StateDim))
	}
	kf.StatePre = cloneVec(x)
	kf.StatePost = cloneVec(x)
}

// checkDims panics if a is not exactly rows x cols (with every row of equal
// length).
func checkDims(name string, a [][]float64, rows, cols int) {
	if len(a) != rows {
		panic(fmt.Sprintf("video: %s must have %d rows, got %d", name, rows, len(a)))
	}
	for i, row := range a {
		if len(row) != cols {
			panic(fmt.Sprintf("video: %s row %d must have %d columns, got %d", name, i, cols, len(row)))
		}
	}
}
