package video

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// --- Kalman control input & setters ---

func TestKalmanFilterControlInput(t *testing.T) {
	// State = [position, velocity]; control u = constant acceleration applied
	// through B = [0.5; 1].
	kf := NewKalmanFilter(2, 1)
	kf.SetTransitionMatrix([][]float64{{1, 1}, {0, 1}})
	kf.SetMeasurementMatrix([][]float64{{1, 0}})
	kf.SetProcessNoiseCov([][]float64{{1e-5, 0}, {0, 1e-5}})
	kf.SetMeasurementNoiseCov([][]float64{{0.05}})
	kf.SetControlMatrix([][]float64{{0.5}, {1}})
	kf.SetState([]float64{0, 0})

	if kf.ControlDim != 1 {
		t.Fatalf("ControlDim = %d, want 1", kf.ControlDim)
	}

	// Ground-truth constant-acceleration motion: a=1.
	const a = 1.0
	pos, vel := 0.0, 0.0
	for k := 0; k < 40; k++ {
		kf.PredictControl([]float64{a})
		// True dynamics for the measurement.
		pos += vel + 0.5*a
		vel += a
		kf.Correct([]float64{pos})
	}
	if math.Abs(kf.StatePost[0]-pos) > 0.5 {
		t.Errorf("position estimate = %.3f, want ~%.3f", kf.StatePost[0], pos)
	}
	if math.Abs(kf.StatePost[1]-vel) > 0.5 {
		t.Errorf("velocity estimate = %.3f, want ~%.3f", kf.StatePost[1], vel)
	}
}

func TestKalmanSettersValidateDims(t *testing.T) {
	kf := NewKalmanFilter(2, 1)
	assertPanics(t, "bad transition dims", func() {
		kf.SetTransitionMatrix([][]float64{{1, 0, 0}})
	})
	assertPanics(t, "bad control dims", func() {
		kf.SetControlMatrix([][]float64{{1}}) // needs 2 rows
	})
	assertPanics(t, "bad state length", func() {
		kf.SetState([]float64{1, 2, 3})
	})
	assertPanics(t, "predictcontrol without model", func() {
		NewKalmanFilter(2, 1).PredictControl([]float64{1})
	})
}

// --- MeanShift / CamShift ---

// blobProb builds a probability image with a single Gaussian blob.
func blobProb(rows, cols int, cx, cy, sx, sy float64) *cv.FloatMat {
	f := cv.NewFloatMat(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			dx := (float64(x) - cx) / sx
			dy := (float64(y) - cy) / sy
			f.Data[y*cols+x] = math.Exp(-0.5 * (dx*dx + dy*dy))
		}
	}
	return f
}

func TestMeanShiftConverges(t *testing.T) {
	prob := blobProb(80, 80, 55, 30, 6, 6)
	// Start the window away from the blob.
	start := cv.Rect{X: 20, Y: 40, Width: 20, Height: 20}
	iters, w := MeanShift(prob, start, NewTermCriteria(50, 0.1))
	cx := float64(w.X) + float64(w.Width)/2
	cy := float64(w.Y) + float64(w.Height)/2
	if math.Abs(cx-55) > 2 || math.Abs(cy-30) > 2 {
		t.Errorf("mean-shift converged to (%.1f, %.1f), want ~(55, 30) after %d iters", cx, cy, iters)
	}
	if iters < 1 {
		t.Errorf("iterations = %d, want >= 1", iters)
	}
}

func TestCamShiftOrientation(t *testing.T) {
	// An elongated horizontal blob: sx > sy, principal axis roughly horizontal.
	prob := blobProb(80, 100, 50, 40, 14, 5)
	start := cv.Rect{X: 30, Y: 25, Width: 30, Height: 30}
	box, _ := CamShift(prob, start, NewTermCriteria(50, 0.1))
	if math.Abs(box.CenterX-50) > 3 || math.Abs(box.CenterY-40) > 3 {
		t.Errorf("CamShift centre = (%.1f, %.1f), want ~(50, 40)", box.CenterX, box.CenterY)
	}
	// The larger extent should correspond to the horizontal (wider) axis.
	if box.Height < box.Width {
		t.Errorf("expected the long axis (Height) to exceed Width, got W=%.1f H=%.1f", box.Width, box.Height)
	}
}

// --- FindTransformECC ---

func TestFindTransformECCTranslation(t *testing.T) {
	const rows, cols = 72, 72
	tmpl := synthTexture(rows, cols, 0, 0)
	const dx, dy = 2.0, 1.0
	input := synthTexture(rows, cols, dx, dy)

	cc, warp := FindTransformECC(tmpl, input, nil, MotionTranslation, NewTermCriteria(60, 1e-8))
	if cc < 0.95 {
		t.Fatalf("ECC correlation = %.4f, want > 0.95", cc)
	}
	// warp maps template -> input; recovered translation should match (dx, dy).
	if math.Abs(warp[0][2]-dx) > 0.3 || math.Abs(warp[1][2]-dy) > 0.3 {
		t.Errorf("recovered translation = (%.3f, %.3f), want ~(%.1f, %.1f)", warp[0][2], warp[1][2], dx, dy)
	}
}

func TestFindTransformECCEuclidean(t *testing.T) {
	const rows, cols = 80, 80
	tmpl := synthTexture(rows, cols, 0, 0)
	// Build the input as a small rotation of the template about its centre.
	m := cv.GetRotationMatrix2D(float64(cols)/2, float64(rows)/2, 3.0, 1.0)
	input := cv.WarpAffine(tmpl, m, cols, rows, cv.InterLinear)

	cc, warp := FindTransformECC(tmpl, input, nil, MotionEuclidean, NewTermCriteria(100, 1e-9))
	if cc < 0.9 {
		t.Fatalf("ECC correlation = %.4f, want > 0.9", cc)
	}
	// Recovered rotation magnitude should be ~3 degrees.
	theta := math.Atan2(warp[1][0], warp[0][0]) * 180 / math.Pi
	if math.Abs(math.Abs(theta)-3.0) > 1.0 {
		t.Errorf("recovered rotation = %.3f deg, want ~±3 deg", theta)
	}
}

func TestFindTransformECCAffine(t *testing.T) {
	const rows, cols = 80, 80
	tmpl := synthTexture(rows, cols, 0, 0)
	// A mild affine: slight anisotropic scale + shear + translation.
	m := cv.AffineMatrix{1.03, 0.02, 1.5, -0.015, 0.98, -1.0}
	input := cv.WarpAffine(tmpl, m, cols, rows, cv.InterLinear)

	cc, warp := FindTransformECC(tmpl, input, nil, MotionAffine, NewTermCriteria(120, 1e-10))
	if cc < 0.9 {
		t.Fatalf("ECC correlation = %.4f, want > 0.9", cc)
	}
	// The recovered 2x3 warp should be close to the applied one.
	want := [][]float64{{1.03, 0.02, 1.5}, {-0.015, 0.98, -1.0}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			tol := 0.05
			if j == 2 {
				tol = 0.6 // translation in pixels
			}
			if math.Abs(warp[i][j]-want[i][j]) > tol {
				t.Errorf("warp[%d][%d] = %.4f, want ~%.4f", i, j, warp[i][j], want[i][j])
			}
		}
	}
}

// --- DIS optical flow ---

func TestDISOpticalFlowShift(t *testing.T) {
	const rows, cols = 96, 96
	const dx, dy = 2.0, 1.0
	prev := synthTexture(rows, cols, 0, 0)
	next := synthTexture(rows, cols, dx, dy)

	dis := NewDISOpticalFlow(DISPresetMedium)
	dis.FinestScale = 0 // full-resolution result for accuracy
	flow := dis.Calc(prev, next)
	if flow.Rows != rows || flow.Cols != cols {
		t.Fatalf("flow dims = %dx%d, want %dx%d", flow.Rows, flow.Cols, rows, cols)
	}
	mx, my := flow.MeanFlow(12)
	if math.Abs(mx-dx) > 0.7 || math.Abs(my-dy) > 0.7 {
		t.Errorf("DIS mean flow = (%.2f, %.2f), want ~(%.1f, %.1f)", mx, my, dx, dy)
	}
}

// scene builds a deterministic, non-periodic textured image shifted by (ox, oy).
// Unlike synthTexture its dominant spatial frequencies span barely more than one
// cycle across the image, so there is no translational ambiguity and pyramidal
// Lucas-Kanade tracks large shifts without aliasing. It is used by the tracker
// and stabilizer tests, which need every point to track reliably.
func scene(rows, cols int, ox, oy float64) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	kx := 2 * math.Pi / float64(cols) * 1.3
	ky := 2 * math.Pi / float64(rows) * 1.1
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			fx := float64(x) - ox
			fy := float64(y) - oy
			v := 128 +
				70*math.Sin(kx*fx+0.5) +
				55*math.Cos(ky*fy+1.1) +
				45*math.Sin(0.5*(kx*fx+ky*fy))
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

// --- TrackerFeaturePyrLK ---

func TestTrackerFeaturePyrLK(t *testing.T) {
	const rows, cols = 90, 90
	f0 := scene(rows, cols, 0, 0)
	f1 := scene(rows, cols, 2, 1)
	f2 := scene(rows, cols, 4, 2)

	pts := []cv.Point{{X: 30, Y: 30}, {X: 50, Y: 45}, {X: 60, Y: 30}}
	tr := NewTrackerFeaturePyrLK(15, 3)
	tr.Init(f0, pts)

	got1, st1 := tr.Update(f1)
	if len(got1) != len(pts) || len(st1) != len(got1) {
		t.Fatalf("update1 returned %d points", len(got1))
	}
	for i := range got1 {
		ex := pts[i].X + 2
		ey := pts[i].Y + 1
		if abs(float64(got1[i].X-ex)) > 1 || abs(float64(got1[i].Y-ey)) > 1 {
			t.Errorf("point %d tracked to %v, want ~(%d,%d)", i, got1[i], ex, ey)
		}
	}
	got2, _ := tr.Update(f2)
	for i := range got2 {
		ex := pts[i].X + 4
		ey := pts[i].Y + 2
		if abs(float64(got2[i].X-ex)) > 1 || abs(float64(got2[i].Y-ey)) > 1 {
			t.Errorf("frame2 point %d tracked to %v, want ~(%d,%d)", i, got2[i], ex, ey)
		}
	}
}

// --- EstimateAffinePartial2D ---

func TestEstimateAffinePartial2D(t *testing.T) {
	const scale, angle = 1.2, 0.3 // radians
	tx, ty := 5.0, -3.0
	c, s := math.Cos(angle), math.Sin(angle)
	var from, to []PointF
	for _, p := range [][2]float64{{0, 0}, {10, 0}, {0, 10}, {10, 10}, {5, 8}} {
		from = append(from, PointF{X: p[0], Y: p[1]})
		to = append(to, PointF{
			X: scale*(c*p[0]-s*p[1]) + tx,
			Y: scale*(s*p[0]+c*p[1]) + ty,
		})
	}
	tf, ok := EstimateAffinePartial2D(from, to)
	if !ok {
		t.Fatal("EstimateAffinePartial2D failed on a well-posed problem")
	}
	if math.Abs(tf.Scale-scale) > 1e-6 || math.Abs(tf.Angle-angle) > 1e-6 ||
		math.Abs(tf.Tx-tx) > 1e-6 || math.Abs(tf.Ty-ty) > 1e-6 {
		t.Errorf("got scale=%.4f angle=%.4f t=(%.4f,%.4f), want scale=%.1f angle=%.1f t=(%.1f,%.1f)",
			tf.Scale, tf.Angle, tf.Tx, tf.Ty, scale, angle, tx, ty)
	}
	if _, ok := EstimateAffinePartial2D([]PointF{{0, 0}}, []PointF{{1, 1}}); ok {
		t.Error("expected failure with a single correspondence")
	}
}

// --- VideoStabilizer ---

func TestVideoStabilizerReducesJitter(t *testing.T) {
	const rows, cols = 96, 96
	jitter := []float64{0, 3, -2, 4, -3, 2, -4, 3, -1, 2}
	base := scene(rows, cols, 0, 0)

	stab := NewVideoStabilizer(9)
	var rawErr, stabErr float64
	for i, jx := range jitter {
		frame := scene(rows, cols, jx, 0)
		out := stab.Stabilize(frame)
		if out.Rows != rows || out.Cols != cols {
			t.Fatalf("frame %d output dims = %dx%d", i, out.Rows, out.Cols)
		}
		if i == 0 {
			continue // reference frame
		}
		rawErr += interiorMAD(frame, base)
		stabErr += interiorMAD(out, base)
	}
	if stabErr >= rawErr {
		t.Errorf("stabilized error %.2f not below raw error %.2f", stabErr, rawErr)
	}
}

// interiorMAD is the mean absolute grayscale difference over the interior region
// (avoiding warp border artefacts).
func interiorMAD(a, b *cv.Mat) float64 {
	ga := toGray(a)
	gb := toGray(b)
	const m = 12
	var sum float64
	var n int
	for y := m; y < ga.Rows-m; y++ {
		for x := m; x < ga.Cols-m; x++ {
			d := float64(ga.Data[y*ga.Cols+x]) - float64(gb.Data[y*gb.Cols+x])
			sum += math.Abs(d)
			n++
		}
	}
	return sum / float64(n)
}

// --- Background subtraction ---

func TestBackgroundSubtractorMOG2(t *testing.T) {
	const rows, cols = 40, 40
	bg := synthTexture(rows, cols, 0, 0)
	sub := NewBackgroundSubtractorMOG2()
	// Learn the static background.
	for i := 0; i < 30; i++ {
		sub.Apply(bg)
	}
	// A frame with a bright square object.
	frame := bg.Clone()
	for y := 15; y < 25; y++ {
		for x := 15; x < 25; x++ {
			frame.Set(y, x, 0, 255)
		}
	}
	mask := sub.Apply(frame)

	fgInObject := 0
	for y := 16; y < 24; y++ {
		for x := 16; x < 24; x++ {
			if mask.At(y, x, 0) == 255 {
				fgInObject++
			}
		}
	}
	if fgInObject < 40 { // 8x8 = 64 pixels, most should fire
		t.Errorf("only %d/64 object pixels flagged as foreground", fgInObject)
	}
	// The static background corners should be mostly background.
	fgInBg := 0
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			if mask.At(y, x, 0) == 255 {
				fgInBg++
			}
		}
	}
	if fgInBg > 12 {
		t.Errorf("%d/64 background pixels wrongly flagged as foreground", fgInBg)
	}
}

func TestBackgroundSubtractorKNN(t *testing.T) {
	const rows, cols = 40, 40
	bg := synthTexture(rows, cols, 0, 0)
	sub := NewBackgroundSubtractorKNN()
	sub.KNNSamples = 2
	for i := 0; i < 20; i++ {
		sub.Apply(bg)
	}
	frame := bg.Clone()
	for y := 15; y < 25; y++ {
		for x := 15; x < 25; x++ {
			frame.Set(y, x, 0, 255)
		}
	}
	mask := sub.Apply(frame)
	fg := 0
	for y := 16; y < 24; y++ {
		for x := 16; x < 24; x++ {
			if mask.At(y, x, 0) == 255 {
				fg++
			}
		}
	}
	if fg < 40 {
		t.Errorf("only %d/64 object pixels flagged as foreground", fg)
	}
}

// --- Sub-pixel optical flow ---

func TestCalcOpticalFlowPyrLKFSubpixel(t *testing.T) {
	const rows, cols = 80, 80
	const dx, dy = 1.5, 0.5
	prev := synthTexture(rows, cols, 0, 0)
	next := synthTexture(rows, cols, dx, dy)

	pts := []PointF{{X: 30, Y: 30}, {X: 45, Y: 40}, {X: 55, Y: 35}}
	got, status, _ := CalcOpticalFlowPyrLKF(prev, next, pts, 15, 3)
	for i, p := range pts {
		if !status[i] {
			t.Errorf("point %d not tracked", i)
			continue
		}
		if math.Abs(got[i].X-(p.X+dx)) > 0.3 || math.Abs(got[i].Y-(p.Y+dy)) > 0.3 {
			t.Errorf("point %d -> (%.3f, %.3f), want ~(%.2f, %.2f)", i, got[i].X, got[i].Y, p.X+dx, p.Y+dy)
		}
	}
}

func TestCalcOpticalFlowFarnebackSubpixel(t *testing.T) {
	const rows, cols = 56, 56
	const dx = 1.5
	prev := synthTexture(rows, cols, 0, 0)
	next := synthTexture(rows, cols, dx, 0)
	flow := CalcOpticalFlowFarnebackSubpixel(prev, next, 3, 3)
	mx, my := flow.MeanFlow(8)
	if math.Abs(mx-dx) > 0.4 || math.Abs(my) > 0.4 {
		t.Errorf("subpixel Farneback mean flow = (%.2f, %.2f), want ~(%.1f, 0)", mx, my, dx)
	}
}

// assertPanics fails the test if fn does not panic.
func assertPanics(t *testing.T, name string, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic, got none", name)
		}
	}()
	fn()
}
