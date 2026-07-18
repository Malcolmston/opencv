package calib2

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestTriangulateKnown(t *testing.T) {
	k := CameraMatrix{Fx: 800, Fy: 800, Cx: 320, Cy: 240}
	pose1 := Pose{R: Mat3Identity()}
	// Second camera translated 1 unit along -x in its own frame (baseline).
	pose2 := Pose{R: Mat3Identity(), T: [3]float64{-1, 0, 0}}
	p1 := ProjectionMatrix(k, pose1)
	p2 := ProjectionMatrix(k, pose2)
	worlds := [][3]float64{{0.5, -0.3, 7}, {-1.2, 0.8, 5}, {0.1, 0.1, 9}}
	for _, w := range worlds {
		pt1 := ProjectPoint(w, pose1, k, DistortionCoeffs{})
		pt2 := ProjectPoint(w, pose2, k, DistortionCoeffs{})
		got := Triangulate(p1, p2, pt1, pt2)
		for i := 0; i < 3; i++ {
			if math.Abs(got[i]-w[i]) > 1e-6 {
				t.Errorf("triangulated %v want %v", got, w)
				break
			}
		}
	}
}

func TestTriangulatePointsBatch(t *testing.T) {
	k := CameraMatrix{Fx: 700, Fy: 700, Cx: 300, Cy: 200}
	pose1 := Pose{R: Mat3Identity()}
	pose2 := NewPoseFromRvec([3]float64{0.02, 0.05, 0}, [3]float64{-1, 0.1, 0})
	p1 := ProjectionMatrix(k, pose1)
	p2 := ProjectionMatrix(k, pose2)
	worlds := [][3]float64{{0, 0, 6}, {1, 1, 7}, {-1, 0.5, 8}}
	var i1, i2 []cv.Point2f
	for _, w := range worlds {
		i1 = append(i1, ProjectPoint(w, pose1, k, DistortionCoeffs{}))
		i2 = append(i2, ProjectPoint(w, pose2, k, DistortionCoeffs{}))
	}
	got := TriangulatePoints(p1, p2, i1, i2)
	for j, w := range worlds {
		for i := 0; i < 3; i++ {
			if math.Abs(got[j][i]-w[i]) > 1e-5 {
				t.Errorf("point %d = %v want %v", j, got[j], w)
				break
			}
		}
	}
}

func TestEpipolarConstraint(t *testing.T) {
	// For a correct fundamental matrix, x2ᵀ F x1 = 0 for all correspondences.
	k1 := CameraMatrix{Fx: 800, Fy: 800, Cx: 320, Cy: 240}
	k2 := CameraMatrix{Fx: 780, Fy: 790, Cx: 310, Cy: 250}
	pose := NewPoseFromRvec([3]float64{0.03, 0.08, -0.02}, [3]float64{-1, 0.05, 0.1})
	f := FundamentalMatrix(k1, k2, pose)
	worlds := [][3]float64{{0.5, -0.3, 7}, {-1.2, 0.8, 6}, {0.1, 0.1, 9}, {1.5, 1.0, 8}}
	for _, w := range worlds {
		x1 := ProjectPoint(w, Pose{R: Mat3Identity()}, k1, DistortionCoeffs{})
		x2 := ProjectPoint(w, pose, k2, DistortionCoeffs{})
		v1 := [3]float64{x1.X, x1.Y, 1}
		v2 := [3]float64{x2.X, x2.Y, 1}
		fv1 := Mat3VecMul(f, v1)
		res := Vec3Dot(v2, fv1)
		if math.Abs(res) > 1e-6 {
			t.Errorf("epipolar constraint violated: %g", res)
		}
	}
}

func TestEpilineThroughCorrespondence(t *testing.T) {
	// The epipolar line of x1 in image 2 must pass through x2.
	k := CameraMatrix{Fx: 800, Fy: 800, Cx: 320, Cy: 240}
	pose := NewPoseFromRvec([3]float64{0.02, 0.06, 0.01}, [3]float64{-1, 0.1, 0.05})
	f := FundamentalMatrix(k, k, pose)
	w := [3]float64{0.4, -0.2, 7}
	x1 := ProjectPoint(w, Pose{R: Mat3Identity()}, k, DistortionCoeffs{})
	x2 := ProjectPoint(w, pose, k, DistortionCoeffs{})
	lines := ComputeCorrespondEpilines([]cv.Point2f{x1}, 1, f)
	l := lines[0]
	if math.Abs(l[0]*l[0]+l[1]*l[1]-1) > 1e-9 {
		t.Errorf("epiline not normalized")
	}
	d := l[0]*x2.X + l[1]*x2.Y + l[2]
	if math.Abs(d) > 1e-6 {
		t.Errorf("point-to-epiline distance %g should be 0", d)
	}
}

func TestDecomposeEssentialMatrix(t *testing.T) {
	pose := NewPoseFromRvec([3]float64{0.1, -0.2, 0.15}, [3]float64{-0.8, 0.3, 0.2})
	// Normalize translation because E is only defined up to scale.
	tn := Vec3Normalize(pose.T)
	e := EssentialMatrix(Pose{R: pose.R, T: tn})
	r1, r2, tt := DecomposeEssentialMatrix(e)
	// One of the recovered rotations must match the true rotation.
	match := mat3MaxDiff(r1, pose.R) < 1e-6 || mat3MaxDiff(r2, pose.R) < 1e-6
	if !match {
		t.Errorf("neither decomposed rotation matches truth\nr1=%v\nr2=%v\nR=%v", r1, r2, pose.R)
	}
	// Translation direction matches up to sign.
	if 1-math.Abs(Vec3Dot(Vec3Normalize(tt), tn)) > 1e-6 {
		t.Errorf("translation direction mismatch: %v vs %v", tt, tn)
	}
}

func TestDisparityTo3DRoundTrip(t *testing.T) {
	// Build Q for a rectified pair, project a point to disparity, reproject.
	k := CameraMatrix{Fx: 800, Fy: 800, Cx: 320, Cy: 240}
	baseline := 0.12
	q := disparityToDepthMatrix(k, baseline)
	world := [3]float64{0.3, -0.15, 4.0}
	// Rectified stereo: left u = fx*X/Z + cx, right u' = fx*(X-B)/Z + cx.
	u := k.Fx*world[0]/world[2] + k.Cx
	v := k.Fy*world[1]/world[2] + k.Cy
	uR := k.Fx*(world[0]-baseline)/world[2] + k.Cx
	disp := u - uR
	got := DisparityTo3D(u, v, disp, q)
	for i := 0; i < 3; i++ {
		if math.Abs(got[i]-world[i]) > 1e-9 {
			t.Errorf("reprojected %v want %v", got, world)
			break
		}
	}
	if depth := DepthFromDisparity(disp, k.Fx, baseline); math.Abs(depth-world[2]) > 1e-9 {
		t.Errorf("DepthFromDisparity %g want %g", depth, world[2])
	}
}

func TestReprojectImageTo3D(t *testing.T) {
	k := CameraMatrix{Fx: 500, Fy: 500, Cx: 10, Cy: 10}
	q := disparityToDepthMatrix(k, 0.1)
	disp := [][]float64{
		{0, 5},
		{10, -1},
	}
	pts := ReprojectImageTo3D(disp, q)
	// Zero and negative disparities are invalid -> zero point.
	if pts[0][0] != ([3]float64{}) {
		t.Errorf("zero-disparity pixel should be zero point, got %v", pts[0][0])
	}
	if pts[1][1] != ([3]float64{}) {
		t.Errorf("negative-disparity pixel should be zero point")
	}
	// Positive disparity -> finite depth z = f*B/d.
	if z := pts[0][1][2]; math.Abs(z-500*0.1/5) > 1e-9 {
		t.Errorf("depth %g want %g", z, 500*0.1/5)
	}
}

func TestStereoRectifyHorizontal(t *testing.T) {
	k1 := CameraMatrix{Fx: 800, Fy: 800, Cx: 320, Cy: 240}
	k2 := k1
	// Camera 2 is translated mostly along x with a slight rotation (typical rig).
	pose := NewPoseFromRvec([3]float64{0.01, 0.02, 0}, [3]float64{-1, 0.02, 0.03})
	rect := StereoRectify(k1, k2, pose)
	if !IsRotationMatrix(rect.R1, 1e-9) || !IsRotationMatrix(rect.R2, 1e-9) {
		t.Fatal("rectifying transforms are not rotations")
	}
	if rect.Baseline <= 0 {
		t.Fatalf("baseline should be positive, got %g", rect.Baseline)
	}
	// After rectification, corresponding image points must share the same row.
	worlds := [][3]float64{{0.2, -0.1, 6}, {-0.3, 0.25, 7}, {0.5, 0.4, 8}}
	for _, w := range worlds {
		x1 := ProjectPoint(w, Pose{R: Mat3Identity()}, k1, DistortionCoeffs{})
		x2 := ProjectPoint(w, pose, k2, DistortionCoeffs{})
		// Map into normalized rays, apply rectifying rotation, reproject with knew.
		r1y := rectifiedRow(x1, k1, rect.R1)
		r2y := rectifiedRow(x2, k2, rect.R2)
		if math.Abs(r1y-r2y) > 1e-6 {
			t.Errorf("rectified rows differ: %g vs %g", r1y, r2y)
		}
	}
}

// rectifiedRow maps a pixel through the inverse intrinsics, applies the
// rectifying rotation and returns the rectified row coordinate.
func rectifiedRow(p cv.Point2f, k CameraMatrix, r [3][3]float64) float64 {
	x, y := pixelToNormalized(p.X, p.Y, k)
	ray := Mat3VecMul(r, [3]float64{x, y, 1})
	return k.Fy*(ray[1]/ray[2]) + k.Cy
}
