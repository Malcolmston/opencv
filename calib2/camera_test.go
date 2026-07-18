package calib2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestProjectPointKnown(t *testing.T) {
	k := CameraMatrix{Fx: 800, Fy: 800, Cx: 320, Cy: 240}
	pose := Pose{R: Mat3Identity()}
	// Point on the optical axis projects to the principal point.
	if p := ProjectPoint([3]float64{0, 0, 10}, pose, k, DistortionCoeffs{}); math.Abs(p.X-320) > 1e-9 || math.Abs(p.Y-240) > 1e-9 {
		t.Errorf("axis point projected to %v want (320,240)", p)
	}
	// Point at x=1,z=10 -> normalized 0.1 -> u = 800*0.1+320 = 400.
	if p := ProjectPoint([3]float64{1, 0, 10}, pose, k, DistortionCoeffs{}); math.Abs(p.X-400) > 1e-9 || math.Abs(p.Y-240) > 1e-9 {
		t.Errorf("projected to %v want (400,240)", p)
	}
}

func TestCameraMatrixInverse(t *testing.T) {
	k := CameraMatrix{Fx: 800, Fy: 750, Cx: 320, Cy: 240, Skew: 1.5}
	inv := k.Inverse()
	prod := Mat3Mul(k.Matrix(), inv)
	if mat3MaxDiff(prod, Mat3Identity()) > 1e-12 {
		t.Errorf("K·K⁻¹ != I")
	}
}

func TestNewCameraMatrixRoundTrip(t *testing.T) {
	k := CameraMatrix{Fx: 800, Fy: 750, Cx: 320, Cy: 240, Skew: 2}
	back := NewCameraMatrix(k.Matrix())
	if back != k {
		t.Errorf("round trip = %+v want %+v", back, k)
	}
}

func TestPoseApplyInverse(t *testing.T) {
	pose := NewPoseFromRvec([3]float64{0.1, 0.2, -0.3}, [3]float64{1, 2, 3})
	x := [3]float64{4, -5, 6}
	cam := pose.Apply(x)
	back := pose.Inverse().Apply(cam)
	for i := 0; i < 3; i++ {
		if math.Abs(back[i]-x[i]) > 1e-10 {
			t.Errorf("pose inverse round trip failed: %v vs %v", back, x)
		}
	}
}

func TestProjectionMatrixDecompose(t *testing.T) {
	k := CameraMatrix{Fx: 800, Fy: 820, Cx: 320, Cy: 240}
	pose := NewPoseFromRvec([3]float64{0.05, -0.1, 0.2}, [3]float64{0.5, -0.3, 8})
	p := ProjectionMatrix(k, pose)
	gk, gr, gt := DecomposeProjectionMatrix(p)
	if math.Abs(gk.Fx-k.Fx) > 1e-6 || math.Abs(gk.Fy-k.Fy) > 1e-6 ||
		math.Abs(gk.Cx-k.Cx) > 1e-6 || math.Abs(gk.Cy-k.Cy) > 1e-6 {
		t.Errorf("decomposed intrinsics %+v want %+v", gk, k)
	}
	if mat3MaxDiff(gr, pose.R) > 1e-8 {
		t.Errorf("decomposed rotation diff %g", mat3MaxDiff(gr, pose.R))
	}
	for i := 0; i < 3; i++ {
		if math.Abs(gt[i]-pose.T[i]) > 1e-6 {
			t.Errorf("decomposed translation %v want %v", gt, pose.T)
		}
	}
}

func TestProjectionMatrixConsistency(t *testing.T) {
	// P·[X;1] (homogeneous) must equal ProjectPoint for a distortion-free camera.
	k := CameraMatrix{Fx: 700, Fy: 700, Cx: 300, Cy: 200}
	pose := NewPoseFromRvec([3]float64{0.1, 0.1, 0}, [3]float64{0, 0, 6})
	p := ProjectionMatrix(k, pose)
	x := [3]float64{1.5, -0.5, 0.3}
	h := [4]float64{x[0], x[1], x[2], 1}
	var proj [3]float64
	for i := 0; i < 3; i++ {
		proj[i] = p[i][0]*h[0] + p[i][1]*h[1] + p[i][2]*h[2] + p[i][3]*h[3]
	}
	pm := cv.Point2f{X: proj[0] / proj[2], Y: proj[1] / proj[2]}
	direct := ProjectPoint(x, pose, k, DistortionCoeffs{})
	if math.Abs(pm.X-direct.X) > 1e-8 || math.Abs(pm.Y-direct.Y) > 1e-8 {
		t.Errorf("projection matrix %v vs ProjectPoint %v", pm, direct)
	}
}
