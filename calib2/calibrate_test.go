package calib2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestApplyAndFindHomographyExact(t *testing.T) {
	h := [3][3]float64{
		{1.1, 0.05, 12},
		{0.03, 1.08, 8},
		{0.0002, 0.0001, 1},
	}
	src := []cv.Point2f{{X: 0, Y: 0}, {X: 100, Y: 0}, {X: 100, Y: 100}, {X: 0, Y: 100}, {X: 40, Y: 70}, {X: 90, Y: 20}}
	dst := make([]cv.Point2f, len(src))
	for i, p := range src {
		dst[i] = ApplyHomography(h, p)
	}
	got, ok := FindHomography(src, dst)
	if !ok {
		t.Fatal("FindHomography failed")
	}
	// Homographies are equal up to scale; both are normalized to h[2][2]=1.
	if mat3MaxDiff(got, h) > 1e-6 {
		t.Errorf("recovered homography diff %g:\n%v", mat3MaxDiff(got, h), got)
	}
	// And it must map the points correctly.
	for i := range src {
		m := ApplyHomography(got, src[i])
		if math.Abs(m.X-dst[i].X) > 1e-6 || math.Abs(m.Y-dst[i].Y) > 1e-6 {
			t.Errorf("point %d mapped to %v want %v", i, m, dst[i])
		}
	}
}

// synthCalibViews builds noise-free image observations of a planar chessboard
// under several poses for a known camera, the standard way to validate Zhang's
// method by exact recovery.
func synthCalibViews(k CameraMatrix, d DistortionCoeffs) (obj [][][3]float64, img [][]cv.Point2f, poses []Pose) {
	board := GenerateChessboardCorners(6, 8, 1.0)
	// Center the board around the origin for well-conditioned homographies.
	centered := make([][3]float64, len(board))
	for i, p := range board {
		centered[i] = [3]float64{p[0] - 3.5, p[1] - 2.5, 0}
	}
	rvecs := [][3]float64{
		{0, 0, 0},
		{0.30, 0, 0},
		{0, 0.30, 0},
		{0.20, -0.20, 0.10},
		{-0.25, 0.15, -0.10},
		{0.12, 0.22, 0.30},
	}
	tvecs := [][3]float64{
		{0, 0, 12},
		{0.4, 0.2, 11},
		{-0.3, 0.1, 13},
		{0.2, -0.4, 12},
		{-0.2, 0.3, 12.5},
		{0.1, 0.1, 11.5},
	}
	for i := range rvecs {
		p := NewPoseFromRvec(rvecs[i], tvecs[i])
		obj = append(obj, centered)
		img = append(img, ProjectPoints(centered, p, k, d))
		poses = append(poses, p)
	}
	return obj, img, poses
}

func TestCalibrateCameraRecoversIntrinsics(t *testing.T) {
	k := CameraMatrix{Fx: 800, Fy: 800, Cx: 320, Cy: 240}
	obj, img, _ := synthCalibViews(k, DistortionCoeffs{})
	gk, gd, poses, rms, err := CalibrateCamera(obj, img)
	if err != nil {
		t.Fatalf("CalibrateCamera error: %v", err)
	}
	if len(poses) != len(obj) {
		t.Fatalf("expected %d poses, got %d", len(obj), len(poses))
	}
	if math.Abs(gk.Fx-k.Fx) > 0.5 || math.Abs(gk.Fy-k.Fy) > 0.5 {
		t.Errorf("focal lengths %g,%g want ~%g", gk.Fx, gk.Fy, k.Fx)
	}
	if math.Abs(gk.Cx-k.Cx) > 0.5 || math.Abs(gk.Cy-k.Cy) > 0.5 {
		t.Errorf("principal point %g,%g want %g,%g", gk.Cx, gk.Cy, k.Cx, k.Cy)
	}
	if math.Abs(gk.Skew) > 1e-2 {
		t.Errorf("skew %g should be ~0", gk.Skew)
	}
	if rms > 1e-3 {
		t.Errorf("RMS reprojection error %g too high for noise-free data", rms)
	}
	if !gd.IsZero() && (math.Abs(gd.K1) > 1e-3 || math.Abs(gd.K2) > 1e-3) {
		t.Errorf("recovered spurious distortion: %+v", gd)
	}
}

func TestIntrinsicsFromHomographies(t *testing.T) {
	k := CameraMatrix{Fx: 650, Fy: 640, Cx: 310, Cy: 250}
	obj, img, _ := synthCalibViews(k, DistortionCoeffs{})
	homs := make([][3][3]float64, len(obj))
	for v := range obj {
		src := make([]cv.Point2f, len(obj[v]))
		for i, p := range obj[v] {
			src[i] = cv.Point2f{X: p[0], Y: p[1]}
		}
		h, ok := FindHomography(src, img[v])
		if !ok {
			t.Fatalf("homography %d failed", v)
		}
		homs[v] = h
	}
	gk, err := IntrinsicsFromHomographies(homs)
	if err != nil {
		t.Fatalf("intrinsics error: %v", err)
	}
	if math.Abs(gk.Fx-k.Fx) > 0.5 || math.Abs(gk.Cy-k.Cy) > 0.5 {
		t.Errorf("intrinsics %+v want %+v", gk, k)
	}
}

func TestExtrinsicsFromHomography(t *testing.T) {
	k := CameraMatrix{Fx: 800, Fy: 800, Cx: 320, Cy: 240}
	pose := NewPoseFromRvec([3]float64{0.15, -0.1, 0.05}, [3]float64{0.3, -0.2, 10})
	board := GenerateChessboardCorners(6, 8, 1.0)
	centered := make([][3]float64, len(board))
	src := make([]cv.Point2f, len(board))
	for i, p := range board {
		centered[i] = [3]float64{p[0] - 3.5, p[1] - 2.5, 0}
		src[i] = cv.Point2f{X: centered[i][0], Y: centered[i][1]}
	}
	img := ProjectPoints(centered, pose, k, DistortionCoeffs{})
	h, ok := FindHomography(src, img)
	if !ok {
		t.Fatal("homography failed")
	}
	got := ExtrinsicsFromHomography(h, k)
	if mat3MaxDiff(got.R, pose.R) > 1e-6 {
		t.Errorf("extrinsic rotation diff %g", mat3MaxDiff(got.R, pose.R))
	}
	for i := 0; i < 3; i++ {
		if math.Abs(got.T[i]-pose.T[i]) > 1e-5 {
			t.Errorf("extrinsic translation %v want %v", got.T, pose.T)
		}
	}
}

func TestReprojectionErrorZero(t *testing.T) {
	k := CameraMatrix{Fx: 800, Fy: 800, Cx: 320, Cy: 240}
	pose := NewPoseFromRvec([3]float64{0.1, 0.1, 0}, [3]float64{0, 0, 10})
	obj := [][3]float64{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}
	img := ProjectPoints(obj, pose, k, DistortionCoeffs{})
	if e := ReprojectionError(obj, img, pose, k, DistortionCoeffs{}); e > 1e-12 {
		t.Errorf("reprojection error %g should be 0", e)
	}
	if e := ReprojectionErrorRMS(obj, img, pose, k, DistortionCoeffs{}); e > 1e-12 {
		t.Errorf("RMS reprojection error %g should be 0", e)
	}
	// Introduce a 3-pixel shift in one point -> mean error > 0.
	img[0].X += 3
	if e := ReprojectionError(obj, img, pose, k, DistortionCoeffs{}); e <= 0 {
		t.Error("expected positive reprojection error after perturbation")
	}
}

func TestGenerateChessboardCorners(t *testing.T) {
	c := GenerateChessboardCorners(2, 3, 2.0)
	if len(c) != 6 {
		t.Fatalf("expected 6 corners, got %d", len(c))
	}
	// Row-major: index 3 is row 1, col 0 -> (0, 2, 0).
	if c[3] != ([3]float64{0, 2, 0}) {
		t.Errorf("corner[3] = %v want (0,2,0)", c[3])
	}
	if c[2] != ([3]float64{4, 0, 0}) {
		t.Errorf("corner[2] = %v want (4,0,0)", c[2])
	}
}
