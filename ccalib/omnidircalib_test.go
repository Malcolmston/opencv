package ccalib

import (
	"math"
	"math/rand"
	"testing"
)

// planarGrid returns a centred planar target of rows×cols points spaced by step.
func planarGrid(rows, cols int, step float64) [][3]float64 {
	var g [][3]float64
	ox := float64(cols-1) * step / 2
	oy := float64(rows-1) * step / 2
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			g = append(g, [3]float64{float64(j)*step - ox, float64(i)*step - oy, 0})
		}
	}
	return g
}

// calibViews is a fixed set of wide-angle-covering poses that break the
// focal/Xi degeneracy of the unified model.
func calibViews() [][2][3]float64 {
	return [][2][3]float64{
		{{0.05, 0.02, 0}, {-0.8, -0.6, 2.2}},
		{{-0.35, 0.1, 0.05}, {0.5, -0.4, 2.6}},
		{{0.4, -0.12, -0.03}, {-0.3, 0.5, 2.4}},
		{{0.1, 0.45, 0.02}, {0.2, 0.3, 2.0}},
		{{-0.3, -0.38, 0.04}, {-0.6, 0.2, 2.3}},
		{{0.28, -0.42, -0.05}, {0.4, -0.5, 2.5}},
	}
}

func TestPoseFromRays(t *testing.T) {
	m := testModel()
	grid := planarGrid(6, 6, 0.4)
	rvec := [3]float64{0.2, -0.15, 0.05}
	tvec := [3]float64{0.3, -0.2, 2.4}
	img := Omnidir.ProjectPoints(grid, rvec, tvec, m.K(), m.Xi, m.Dist())
	dirs := make([][3]float64, len(img))
	for i, p := range img {
		d, ok := liftToSphere(p[0], p[1], m.Xi, m.Fx, m.Fy, m.Cx, m.Cy, m.Skew, m.K1, m.K2, m.P1, m.P2)
		if !ok {
			t.Fatalf("lift failed")
		}
		dirs[i] = d
	}
	r, tt, ok := poseFromRays(grid, dirs)
	if !ok {
		t.Fatalf("poseFromRays failed")
	}
	if norm3(sub3(r, rvec)) > 0.02 {
		t.Fatalf("rvec off: got %v want %v", r, rvec)
	}
	if norm3(sub3(tt, tvec)) > 0.05 {
		t.Fatalf("tvec off: got %v want %v", tt, tvec)
	}
}

func TestCalibrateRecoversFocal(t *testing.T) {
	m := testModel()
	grid := planarGrid(7, 7, 0.5)
	views := calibViews()
	rng := rand.New(rand.NewSource(42))
	var objs [][][3]float64
	var imgs [][][2]float64
	for _, v := range views {
		pts := Omnidir.ProjectPoints(grid, v[0], v[1], m.K(), m.Xi, m.Dist())
		// Add small deterministic pixel noise.
		for i := range pts {
			pts[i][0] += rng.NormFloat64() * 0.1
			pts[i][1] += rng.NormFloat64() * 0.1
		}
		objs = append(objs, grid)
		imgs = append(imgs, pts)
	}
	// Deliberately poor focal/centre guess; Xi is a known lens constant held
	// fixed (the CALIB_FIX_XI equivalent) so the focal estimate is well posed.
	guess := OmniModel{Fx: 230, Fy: 230, Cx: 330, Cy: 250, Xi: m.Xi}
	model, rvecs, tvecs, rms, ok := Omnidir.Calibrate(objs, imgs, 640, 480, guess, true)
	if !ok {
		t.Fatalf("calibration failed")
	}
	if len(rvecs) != len(views) || len(tvecs) != len(views) {
		t.Fatalf("expected %d extrinsics", len(views))
	}
	if rms > 0.5 {
		t.Fatalf("reprojection RMS too high: %.3f", rms)
	}
	fxErr := 100 * math.Abs(model.Fx-m.Fx) / m.Fx
	fyErr := 100 * math.Abs(model.Fy-m.Fy) / m.Fy
	if fxErr > 3 || fyErr > 3 {
		t.Fatalf("focal not recovered within 3%%: Fx err %.2f%%, Fy err %.2f%% (got Fx=%.2f Fy=%.2f)", fxErr, fyErr, model.Fx, model.Fy)
	}
	if math.Abs(model.Xi-m.Xi) > 0.05 {
		t.Fatalf("Xi not recovered: got %.4f want %.4f", model.Xi, m.Xi)
	}
}

func TestCalibrateFreeXiNoiseFree(t *testing.T) {
	m := testModel()
	grid := planarGrid(7, 7, 0.5)
	views := calibViews()
	var objs [][][3]float64
	var imgs [][][2]float64
	for _, v := range views {
		objs = append(objs, grid)
		imgs = append(imgs, Omnidir.ProjectPoints(grid, v[0], v[1], m.K(), m.Xi, m.Dist()))
	}
	guess := OmniModel{Fx: 240, Fy: 240, Cx: 320, Cy: 240, Xi: 0.95}
	model, _, _, rms, ok := Omnidir.Calibrate(objs, imgs, 640, 480, guess, false)
	if !ok {
		t.Fatalf("calibration failed")
	}
	if rms > 0.05 {
		t.Fatalf("noise-free RMS too high: %.4f", rms)
	}
	// With wide-angle, noise-free data the focal/Xi valley is broken and both
	// recover well even with Xi free.
	if 100*math.Abs(model.Fx-m.Fx)/m.Fx > 3 {
		t.Fatalf("free-Xi focal not recovered: got Fx=%.2f", model.Fx)
	}
	if math.Abs(model.Xi-m.Xi) > 0.05 {
		t.Fatalf("free-Xi mirror not recovered: got Xi=%.4f want %.4f", model.Xi, m.Xi)
	}
}

func TestCalibrateRejectsTooFewViews(t *testing.T) {
	grid := planarGrid(5, 5, 0.4)
	objs := [][][3]float64{grid, grid}
	imgs := [][][2]float64{
		Omnidir.ProjectPoints(grid, [3]float64{0, 0, 0}, [3]float64{0, 0, 2}, testModel().K(), 1.05, nil),
		Omnidir.ProjectPoints(grid, [3]float64{0, 0, 0}, [3]float64{0.1, 0, 2}, testModel().K(), 1.05, nil),
	}
	if _, _, _, _, ok := Omnidir.Calibrate(objs, imgs, 640, 480, OmniModel{}, false); ok {
		t.Fatalf("expected calibration to reject fewer than 3 views")
	}
}

func TestStereoCalibrate(t *testing.T) {
	m1 := testModel()
	m2 := OmniModel{Fx: 300, Fy: 300, Cx: 320, Cy: 240, Xi: 1.0, K1: -0.01}
	grid := planarGrid(6, 6, 0.5)
	// True relative transform camera1 -> camera2.
	Rtrue := rodriguesToMatrix([3]float64{0.03, 0.12, -0.02})
	Ttrue := [3]float64{0.25, -0.05, 0.1}
	views := calibViews()
	var objs [][][3]float64
	var i1, i2 [][][2]float64
	for _, v := range views {
		R1 := rodriguesToMatrix(v[0])
		t1 := v[1]
		// Camera2 pose = relative ∘ camera1 pose.
		R2 := mul3(Rtrue, R1)
		t2 := add3(matVec3(Rtrue, t1), Ttrue)
		objs = append(objs, grid)
		i1 = append(i1, Omnidir.ProjectPoints(grid, v[0], t1, m1.K(), m1.Xi, m1.Dist()))
		i2 = append(i2, Omnidir.ProjectPoints(grid, rodriguesToVector(R2), t2, m2.K(), m2.Xi, m2.Dist()))
	}
	R, T, rms, ok := Omnidir.StereoCalibrate(objs, i1, i2, m1, m2)
	if !ok {
		t.Fatalf("stereo calibration failed")
	}
	rErr := norm3(sub3(rodriguesToVector(R), rodriguesToVector(Rtrue)))
	tErr := norm3(sub3(T, Ttrue))
	if rErr > 0.01 {
		t.Fatalf("relative rotation off by %.4f rad", rErr)
	}
	if tErr > 0.02 {
		t.Fatalf("relative translation off by %.4f, got %v want %v", tErr, T, Ttrue)
	}
	if rms > 0.5 {
		t.Fatalf("stereo RMS too high: %.3f", rms)
	}
}

func TestMultiCameraCalibration(t *testing.T) {
	m := testModel()
	models := []OmniModel{m, m, m}
	// Ground-truth camera poses relative to camera 0.
	camPoses := []Pose{
		IdentityPose(),
		{R: rodriguesToMatrix([3]float64{0, 0.3, 0}), T: [3]float64{0.5, 0, 0.05}},
		{R: rodriguesToMatrix([3]float64{0.1, -0.2, 0.05}), T: [3]float64{-0.4, 0.1, 0.1}},
	}
	grid := planarGrid(6, 6, 0.5)
	framePoses := []Pose{
		{R: rodriguesToMatrix([3]float64{0.05, 0.02, 0}), T: [3]float64{-0.3, -0.2, 2.3}},
		{R: rodriguesToMatrix([3]float64{-0.2, 0.1, 0.03}), T: [3]float64{0.2, -0.1, 2.5}},
		{R: rodriguesToMatrix([3]float64{0.15, -0.15, 0}), T: [3]float64{0, 0.2, 2.1}},
		{R: rodriguesToMatrix([3]float64{0.1, 0.2, 0.04}), T: [3]float64{0.1, 0.1, 2.4}},
	}
	var obs []MultiCameraObservation
	for c := range models {
		for f := range framePoses {
			// Pattern-to-camera pose = camPose ∘ framePose.
			pcf := camPoses[c].Compose(framePoses[f])
			img := Omnidir.ProjectPoints(grid, pcf.Rvec(), pcf.T, models[c].K(), models[c].Xi, models[c].Dist())
			obs = append(obs, MultiCameraObservation{Camera: c, Frame: f, ObjectPoints: grid, ImagePoints: img})
		}
	}
	res, ok := Omnidir.MultiCameraCalibration(models, len(framePoses), obs)
	if !ok {
		t.Fatalf("multi-camera calibration failed")
	}
	for c := range models {
		if !res.CameraKnown[c] {
			t.Fatalf("camera %d not resolved", c)
		}
		rErr := norm3(sub3(res.CameraPoses[c].Rvec(), camPoses[c].Rvec()))
		tErr := norm3(sub3(res.CameraPoses[c].T, camPoses[c].T))
		if rErr > 0.01 || tErr > 0.02 {
			t.Fatalf("camera %d pose off: rErr=%.4f tErr=%.4f", c, rErr, tErr)
		}
	}
	for f := range framePoses {
		if !res.FrameKnown[f] {
			t.Fatalf("frame %d not resolved", f)
		}
		tErr := norm3(sub3(res.FramePoses[f].T, framePoses[f].T))
		if tErr > 0.02 {
			t.Fatalf("frame %d translation off by %.4f", f, tErr)
		}
	}
}

func TestPoseComposeInverse(t *testing.T) {
	p := Pose{R: rodriguesToMatrix([3]float64{0.2, -0.1, 0.3}), T: [3]float64{1, -2, 3}}
	id := p.Compose(p.Inverse())
	v := id.Apply([3]float64{4, 5, 6})
	if norm3(sub3(v, [3]float64{4, 5, 6})) > 1e-9 {
		t.Fatalf("p ∘ p^{-1} is not identity: %v", v)
	}
}
