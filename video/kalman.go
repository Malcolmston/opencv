package video

// KalmanFilter is a classic linear Kalman filter, mirroring cv::KalmanFilter.
// It estimates the state of a linear dynamical system observed through noisy
// measurements by alternating a [KalmanFilter.Predict] step (project the state
// and its covariance forward through the transition model) and a
// [KalmanFilter.Correct] step (fuse a new measurement).
//
// All matrices are dense [][]float64 in row-major order. For a system with an
// n-dimensional state and m-dimensional measurement:
//
//   - StatePre, StatePost are length-n state vectors (x' and x).
//   - TransitionMatrix (A) is n x n: the state transition model.
//   - MeasurementMatrix (H) is m x n: maps state to measurement space.
//   - ProcessNoiseCov (Q) is n x n: additive process noise covariance.
//   - MeasurementNoiseCov (R) is m x m: measurement noise covariance.
//   - ErrorCovPre, ErrorCovPost (P' and P) are n x n error covariances.
//
// Construct a filter with [NewKalmanFilter], which installs sensible identity
// defaults, then set TransitionMatrix, MeasurementMatrix and the noise
// covariances (and optionally the initial StatePost / ErrorCovPost) to describe
// your model.
type KalmanFilter struct {
	// StateDim is the dimension n of the state vector.
	StateDim int
	// MeasureDim is the dimension m of the measurement vector.
	MeasureDim int

	// StatePre is the predicted state x' = A*x (updated by Predict).
	StatePre []float64
	// StatePost is the corrected state x (updated by Correct).
	StatePost []float64

	// TransitionMatrix is the n x n state-transition model A.
	TransitionMatrix [][]float64
	// MeasurementMatrix is the m x n measurement model H.
	MeasurementMatrix [][]float64
	// ProcessNoiseCov is the n x n process-noise covariance Q.
	ProcessNoiseCov [][]float64
	// MeasurementNoiseCov is the m x m measurement-noise covariance R.
	MeasurementNoiseCov [][]float64
	// ErrorCovPre is the predicted n x n error covariance P'.
	ErrorCovPre [][]float64
	// ErrorCovPost is the corrected n x n error covariance P.
	ErrorCovPost [][]float64

	// ControlMatrix is the optional n x c control model B applied by
	// [KalmanFilter.PredictControl]. It is nil until a control model is
	// installed with [KalmanFilter.SetControlMatrix]; a plain [KalmanFilter.Predict]
	// never consults it. ControlDim records its column count c.
	ControlMatrix [][]float64
	// ControlDim is the dimension c of the control-input vector (0 when no
	// control model is installed).
	ControlDim int
}

// NewKalmanFilter creates a filter for an n-dimensional state and an
// m-dimensional measurement. TransitionMatrix, ProcessNoiseCov, MeasurementNoiseCov
// and ErrorCovPost are initialised to identity matrices, MeasurementMatrix to a
// zero m x n matrix, and both state vectors to zero. Callers are expected to
// overwrite the matrices that describe their model. It panics unless n >= 1 and
// m >= 1.
func NewKalmanFilter(stateDim, measureDim int) *KalmanFilter {
	if stateDim < 1 || measureDim < 1 {
		panic("video: NewKalmanFilter requires stateDim >= 1 and measureDim >= 1")
	}
	return &KalmanFilter{
		StateDim:            stateDim,
		MeasureDim:          measureDim,
		StatePre:            make([]float64, stateDim),
		StatePost:           make([]float64, stateDim),
		TransitionMatrix:    identity(stateDim),
		MeasurementMatrix:   zeros(measureDim, stateDim),
		ProcessNoiseCov:     identity(stateDim),
		MeasurementNoiseCov: identity(measureDim),
		ErrorCovPre:         identity(stateDim),
		ErrorCovPost:        identity(stateDim),
	}
}

// Predict advances the filter one time step using the transition model and
// returns the predicted state x'. It computes
//
//	x' = A * x
//	P' = A * P * Aᵀ + Q
//
// updating StatePre and ErrorCovPre. The returned slice is a fresh copy.
func (kf *KalmanFilter) Predict() []float64 {
	// x' = A x
	kf.StatePre = matVec(kf.TransitionMatrix, kf.StatePost)
	// P' = A P Aᵀ + Q
	ap := matMul(kf.TransitionMatrix, kf.ErrorCovPost)
	apat := matMul(ap, transpose(kf.TransitionMatrix))
	kf.ErrorCovPre = matAdd(apat, kf.ProcessNoiseCov)
	// OpenCV leaves statePost equal to statePre until Correct runs; mirror that
	// so a Predict without a following Correct still advances the estimate.
	kf.StatePost = cloneVec(kf.StatePre)
	kf.ErrorCovPost = cloneMat(kf.ErrorCovPre)
	return cloneVec(kf.StatePre)
}

// Correct fuses a measurement z (length MeasureDim) and returns the corrected
// state x. It computes the Kalman gain and updates the state and covariance:
//
//	y = z - H * x'
//	S = H * P' * Hᵀ + R
//	K = P' * Hᵀ * S⁻¹
//	x = x' + K * y
//	P = (I - K * H) * P'
//
// updating StatePost and ErrorCovPost. It panics if len(z) != MeasureDim or the
// innovation covariance S is singular. The returned slice is a fresh copy.
func (kf *KalmanFilter) Correct(z []float64) []float64 {
	if len(z) != kf.MeasureDim {
		panic("video: KalmanFilter.Correct measurement has wrong length")
	}
	h := kf.MeasurementMatrix
	ht := transpose(h)
	// S = H P' Hᵀ + R
	pHt := matMul(kf.ErrorCovPre, ht)
	s := matAdd(matMul(h, pHt), kf.MeasurementNoiseCov)
	sInv, ok := matInverse(s)
	if !ok {
		panic("video: KalmanFilter.Correct innovation covariance is singular")
	}
	// K = P' Hᵀ S⁻¹
	k := matMul(pHt, sInv)
	// y = z - H x'
	y := vecSub(z, matVec(h, kf.StatePre))
	// x = x' + K y
	kf.StatePost = vecAdd(kf.StatePre, matVec(k, y))
	// P = (I - K H) P'
	kh := matMul(k, h)
	ikh := matSub(identity(kf.StateDim), kh)
	kf.ErrorCovPost = matMul(ikh, kf.ErrorCovPre)
	return cloneVec(kf.StatePost)
}

// cloneVec returns a copy of v.
func cloneVec(v []float64) []float64 {
	out := make([]float64, len(v))
	copy(out, v)
	return out
}
