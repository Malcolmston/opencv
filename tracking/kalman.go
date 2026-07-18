package tracking

// KalmanFilter is a linear Kalman filter following the same layout as OpenCV's
// cv::KalmanFilter. The exported matrices may be inspected and overwritten
// directly to configure the model before the first predict/correct cycle:
//
//	x_pre  = F·x_post + B·u                 (state prediction)
//	P_pre  = F·P_post·Fᵀ + Q                (covariance prediction)
//	K      = P_pre·Hᵀ·(H·P_pre·Hᵀ + R)⁻¹    (Kalman gain)
//	x_post = x_pre + K·(z − H·x_pre)        (state correction)
//	P_post = (I − K·H)·P_pre                (covariance correction)
//
// where F is TransitionMatrix, B is ControlMatrix, H is MeasurementMatrix, Q is
// ProcessNoiseCov and R is MeasurementNoiseCov. All state and covariance
// matrices are stored as column vectors / square matrices of float64.
type KalmanFilter struct {
	// StatePre is the predicted state x_pre (DynamParams x 1).
	StatePre *Matrix
	// StatePost is the corrected state x_post (DynamParams x 1).
	StatePost *Matrix
	// TransitionMatrix is the state-transition model F (DynamParams x DynamParams).
	TransitionMatrix *Matrix
	// ControlMatrix is the control model B (DynamParams x ControlParams); nil
	// when the filter has no control input.
	ControlMatrix *Matrix
	// MeasurementMatrix is the observation model H (MeasureParams x DynamParams).
	MeasurementMatrix *Matrix
	// ProcessNoiseCov is the process-noise covariance Q (DynamParams x DynamParams).
	ProcessNoiseCov *Matrix
	// MeasurementNoiseCov is the measurement-noise covariance R (MeasureParams x MeasureParams).
	MeasurementNoiseCov *Matrix
	// ErrorCovPre is the predicted error covariance P_pre.
	ErrorCovPre *Matrix
	// ErrorCovPost is the corrected error covariance P_post.
	ErrorCovPost *Matrix

	dynamParams   int
	measureParams int
	controlParams int
}

// NewKalmanFilter constructs a Kalman filter with the given state, measurement
// and control dimensions (controlParams may be 0 for no control input). The
// transition, measurement and control matrices are zeroed, the process and
// measurement noise covariances are set to the identity, and the posterior error
// covariance to the identity. Callers configure the model by writing the
// exported matrices before running the predict/correct cycle. It panics if
// dynamParams or measureParams is not positive.
func NewKalmanFilter(dynamParams, measureParams, controlParams int) *KalmanFilter {
	if dynamParams <= 0 || measureParams <= 0 {
		panic("tracking: NewKalmanFilter requires positive dynamParams and measureParams")
	}
	if controlParams < 0 {
		panic("tracking: NewKalmanFilter requires controlParams >= 0")
	}
	kf := &KalmanFilter{
		StatePre:            NewMatrix(dynamParams, 1),
		StatePost:           NewMatrix(dynamParams, 1),
		TransitionMatrix:    NewMatrix(dynamParams, dynamParams),
		MeasurementMatrix:   NewMatrix(measureParams, dynamParams),
		ProcessNoiseCov:     IdentityMatrix(dynamParams),
		MeasurementNoiseCov: IdentityMatrix(measureParams),
		ErrorCovPre:         NewMatrix(dynamParams, dynamParams),
		ErrorCovPost:        IdentityMatrix(dynamParams),
		dynamParams:         dynamParams,
		measureParams:       measureParams,
		controlParams:       controlParams,
	}
	if controlParams > 0 {
		kf.ControlMatrix = NewMatrix(dynamParams, controlParams)
	}
	return kf
}

// Predict advances the filter one step and returns the predicted state as a
// fresh slice. control supplies the control vector; pass nil (or an empty slice)
// when the filter has no control input. It panics if control is non-empty but
// its length does not match the configured control dimension.
func (kf *KalmanFilter) Predict(control []float64) []float64 {
	// x_pre = F * x_post (+ B * u).
	kf.StatePre = kf.TransitionMatrix.Mul(kf.StatePost)
	if len(control) > 0 {
		if kf.ControlMatrix == nil || len(control) != kf.controlParams {
			panic("tracking: KalmanFilter.Predict control vector has wrong length")
		}
		u := columnVector(control)
		kf.StatePre = kf.StatePre.Add(kf.ControlMatrix.Mul(u))
	}
	// P_pre = F * P_post * Fᵀ + Q.
	ft := kf.TransitionMatrix.Transpose()
	kf.ErrorCovPre = kf.TransitionMatrix.Mul(kf.ErrorCovPost).Mul(ft).Add(kf.ProcessNoiseCov)
	// Default the posterior to the prediction until a correction arrives.
	kf.StatePost = kf.StatePre.Clone()
	kf.ErrorCovPost = kf.ErrorCovPre.Clone()
	return matColumnToSlice(kf.StatePre)
}

// Correct updates the predicted state with the measurement vector and returns
// the corrected state as a fresh slice. It panics if measurement's length does
// not match the configured measurement dimension.
func (kf *KalmanFilter) Correct(measurement []float64) []float64 {
	if len(measurement) != kf.measureParams {
		panic("tracking: KalmanFilter.Correct measurement vector has wrong length")
	}
	h := kf.MeasurementMatrix
	ht := h.Transpose()
	// S = H * P_pre * Hᵀ + R.
	s := h.Mul(kf.ErrorCovPre).Mul(ht).Add(kf.MeasurementNoiseCov)
	// K = P_pre * Hᵀ * S⁻¹.
	k := kf.ErrorCovPre.Mul(ht).Mul(s.Inverse())
	z := columnVector(measurement)
	// y = z - H * x_pre.
	innov := z.Sub(h.Mul(kf.StatePre))
	// x_post = x_pre + K * y.
	kf.StatePost = kf.StatePre.Add(k.Mul(innov))
	// P_post = (I - K * H) * P_pre.
	ident := IdentityMatrix(kf.dynamParams)
	kf.ErrorCovPost = ident.Sub(k.Mul(h)).Mul(kf.ErrorCovPre)
	return matColumnToSlice(kf.StatePost)
}

// StateVector returns the current corrected state as a fresh slice.
func (kf *KalmanFilter) StateVector() []float64 {
	return matColumnToSlice(kf.StatePost)
}

// NewConstantVelocityKalman2D constructs a Kalman filter for tracking a 2-D
// point under a constant-velocity motion model. The state is (x, y, vx, vy) and
// the measurement is the observed position (x, y). dt is the time step between
// updates; processNoise scales the process-noise covariance Q (uncertainty in
// the constant-velocity assumption) and measureNoise scales the measurement
// covariance R (sensor noise). The filter is initialised at position (x0, y0)
// with zero velocity. Larger measureNoise makes the filter trust its model more
// and smooth measurements harder.
func NewConstantVelocityKalman2D(x0, y0, dt, processNoise, measureNoise float64) *KalmanFilter {
	kf := NewKalmanFilter(4, 2, 0)
	kf.TransitionMatrix = MatrixFromRows([][]float64{
		{1, 0, dt, 0},
		{0, 1, 0, dt},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	})
	kf.MeasurementMatrix = MatrixFromRows([][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
	})
	kf.ProcessNoiseCov = IdentityMatrix(4).Scale(processNoise)
	kf.MeasurementNoiseCov = IdentityMatrix(2).Scale(measureNoise)
	kf.ErrorCovPost = IdentityMatrix(4)
	kf.StatePost = MatrixFromRows([][]float64{{x0}, {y0}, {0}, {0}})
	return kf
}

// columnVector wraps a slice as an n-by-1 matrix.
func columnVector(v []float64) *Matrix {
	m := NewMatrix(len(v), 1)
	copy(m.Data, v)
	return m
}

// matColumnToSlice copies a column-vector matrix into a fresh slice.
func matColumnToSlice(m *Matrix) []float64 {
	out := make([]float64, m.Rows)
	for i := 0; i < m.Rows; i++ {
		out[i] = m.Data[i*m.Cols]
	}
	return out
}
