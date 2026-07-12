package video

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// synthTexture builds a deterministic, well-textured single-channel image whose
// intensity at (x, y) is a smooth function shifted by (ox, oy). Sampling the
// same analytic function at a fractional offset yields an exact translated image
// so optical-flow accuracy can be measured against a known ground truth.
func synthTexture(rows, cols int, ox, oy float64) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			fx := float64(x) - ox
			fy := float64(y) - oy
			// Two orthogonal sinusoids plus a cross term give strong gradients
			// in both directions and a well-conditioned structure tensor.
			v := 128 +
				50*math.Sin(0.30*fx) +
				50*math.Cos(0.28*fy) +
				25*math.Sin(0.15*(fx+fy))
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

func TestBuildOpticalFlowPyramidHalves(t *testing.T) {
	img := synthTexture(64, 64, 0, 0)
	pyr := BuildOpticalFlowPyramid(img, 3)
	if len(pyr) != 4 {
		t.Fatalf("pyramid length = %d, want 4", len(pyr))
	}
	wantR := []int{64, 32, 16, 8}
	wantC := []int{64, 32, 16, 8}
	for i, lv := range pyr {
		if lv.Rows != wantR[i] || lv.Cols != wantC[i] {
			t.Errorf("level %d dims = %dx%d, want %dx%d", i, lv.Rows, lv.Cols, wantR[i], wantC[i])
		}
	}
	// Every level must be single-channel grayscale.
	for i, lv := range pyr {
		if lv.Channels != 1 {
			t.Errorf("level %d channels = %d, want 1", i, lv.Channels)
		}
	}
}

func TestBuildOpticalFlowPyramidNonSquare(t *testing.T) {
	img := synthTexture(30, 48, 0, 0)
	pyr := BuildOpticalFlowPyramid(img, 2)
	// PyrDown rounds dimensions up: 30->15->8, 48->24->12.
	wantR := []int{30, 15, 8}
	wantC := []int{48, 24, 12}
	for i, lv := range pyr {
		if lv.Rows != wantR[i] || lv.Cols != wantC[i] {
			t.Errorf("level %d dims = %dx%d, want %dx%d", i, lv.Rows, lv.Cols, wantR[i], wantC[i])
		}
	}
}

func TestCalcOpticalFlowPyrLKTranslation(t *testing.T) {
	const (
		rows = 80
		cols = 80
		dx   = 2.0
		dy   = 1.0
	)
	prev := synthTexture(rows, cols, 0, 0)
	next := synthTexture(rows, cols, dx, dy)

	// A grid of interior feature points, kept away from the border.
	var pts []cv.Point
	for y := 20; y <= 60; y += 10 {
		for x := 20; x <= 60; x += 10 {
			pts = append(pts, cv.Point{X: x, Y: y})
		}
	}

	nextPts, status, errs := CalcOpticalFlowPyrLK(prev, next, pts, 15, 3)
	if len(nextPts) != len(pts) || len(status) != len(pts) || len(errs) != len(pts) {
		t.Fatalf("output lengths = %d/%d/%d, want %d", len(nextPts), len(status), len(errs), len(pts))
	}

	good := 0
	for i, p := range pts {
		if !status[i] {
			continue
		}
		ex := float64(p.X) + dx
		ey := float64(p.Y) + dy
		if math.Abs(float64(nextPts[i].X)-ex) <= 1 && math.Abs(float64(nextPts[i].Y)-ey) <= 1 {
			good++
		}
	}
	// Require the large majority of tracked points to be within ~1px.
	if good < (len(pts)*8)/10 {
		t.Errorf("only %d/%d points tracked within 1px", good, len(pts))
	}
}

func TestCalcOpticalFlowPyrLKZeroMotion(t *testing.T) {
	img := synthTexture(60, 60, 0, 0)
	pts := []cv.Point{{X: 30, Y: 30}, {X: 20, Y: 40}}
	nextPts, status, _ := CalcOpticalFlowPyrLK(img, img.Clone(), pts, 15, 2)
	for i, p := range pts {
		if !status[i] {
			t.Errorf("point %d not tracked on identical frames", i)
			continue
		}
		if nextPts[i].X != p.X || nextPts[i].Y != p.Y {
			t.Errorf("point %d moved to %v on identical frames, want %v", i, nextPts[i], p)
		}
	}
}

func TestCalcOpticalFlowFarnebackShift(t *testing.T) {
	const (
		rows = 48
		cols = 48
		dx   = 1.0
		dy   = 0.0
	)
	prev := synthTexture(rows, cols, 0, 0)
	next := synthTexture(rows, cols, dx, dy)
	flow := CalcOpticalFlowFarneback(prev, next, 3, 3)
	if flow.Rows != rows || flow.Cols != cols {
		t.Fatalf("flow dims = %dx%d, want %dx%d", flow.Rows, flow.Cols, rows, cols)
	}
	mx, my := flow.MeanFlow(6)
	if math.Abs(mx-dx) > 0.5 || math.Abs(my-dy) > 0.5 {
		t.Errorf("mean flow = (%.2f, %.2f), want ~(%.1f, %.1f)", mx, my, dx, dy)
	}
}

func TestKalmanFilterLinearTrajectory(t *testing.T) {
	// Constant-velocity model: state = [position, velocity].
	kf := NewKalmanFilter(2, 1)
	kf.TransitionMatrix = [][]float64{
		{1, 1},
		{0, 1},
	}
	kf.MeasurementMatrix = [][]float64{
		{1, 0},
	}
	// Small process noise, moderate measurement noise.
	kf.ProcessNoiseCov = [][]float64{
		{1e-4, 0},
		{0, 1e-4},
	}
	kf.MeasurementNoiseCov = [][]float64{{0.25}}
	kf.ErrorCovPost = [][]float64{
		{1, 0},
		{0, 1},
	}
	// Start with a deliberately wrong initial state to test convergence.
	kf.StatePost = []float64{0, 0}

	const velocity = 2.0
	var lastPos, lastVel float64
	for k := 1; k <= 60; k++ {
		kf.Predict()
		truePos := velocity * float64(k)
		kf.Correct([]float64{truePos})
		lastPos = kf.StatePost[0]
		lastVel = kf.StatePost[1]
	}
	trueFinal := velocity * 60.0
	if math.Abs(lastPos-trueFinal) > 0.5 {
		t.Errorf("final position estimate = %.3f, want ~%.1f", lastPos, trueFinal)
	}
	if math.Abs(lastVel-velocity) > 0.2 {
		t.Errorf("final velocity estimate = %.3f, want ~%.1f", lastVel, velocity)
	}
}

func TestKalmanFilterConstantMeasurement(t *testing.T) {
	// A 1D position-only filter fed a constant measurement should converge to it.
	kf := NewKalmanFilter(1, 1)
	kf.MeasurementMatrix = [][]float64{{1}}
	kf.ProcessNoiseCov = [][]float64{{1e-2}}
	kf.MeasurementNoiseCov = [][]float64{{1}}
	kf.StatePost = []float64{0}

	const target = 10.0
	for i := 0; i < 100; i++ {
		kf.Predict()
		kf.Correct([]float64{target})
	}
	if math.Abs(kf.StatePost[0]-target) > 0.05 {
		t.Errorf("estimate = %.4f, want ~%.1f", kf.StatePost[0], target)
	}
}

func TestMatInverseIdentity(t *testing.T) {
	a := [][]float64{
		{4, 3},
		{6, 3},
	}
	inv, ok := matInverse(a)
	if !ok {
		t.Fatal("matInverse reported singular for an invertible matrix")
	}
	prod := matMul(a, inv)
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			want := 0.0
			if i == j {
				want = 1.0
			}
			if math.Abs(prod[i][j]-want) > 1e-9 {
				t.Errorf("a*inv[%d][%d] = %.6f, want %.1f", i, j, prod[i][j], want)
			}
		}
	}
	if _, ok := matInverse([][]float64{{1, 2}, {2, 4}}); ok {
		t.Error("matInverse should report singular for a rank-deficient matrix")
	}
}
