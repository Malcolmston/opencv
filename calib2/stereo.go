package calib2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Triangulate reconstructs a single 3D point from its projections in two views
// by the homogeneous linear (DLT) method. p1 and p2 are the 3×4 projection
// matrices of the two cameras and pt1, pt2 the corresponding image points. The
// returned point is in the world frame of the projection matrices.
func Triangulate(p1, p2 [3][4]float64, pt1, pt2 cv.Point2f) [3]float64 {
	a := NewMatrix(4, 4)
	setRow := func(r int, coeff [4]float64) {
		for j := 0; j < 4; j++ {
			a.Data[r*4+j] = coeff[j]
		}
	}
	setRow(0, [4]float64{
		pt1.X*p1[2][0] - p1[0][0],
		pt1.X*p1[2][1] - p1[0][1],
		pt1.X*p1[2][2] - p1[0][2],
		pt1.X*p1[2][3] - p1[0][3],
	})
	setRow(1, [4]float64{
		pt1.Y*p1[2][0] - p1[1][0],
		pt1.Y*p1[2][1] - p1[1][1],
		pt1.Y*p1[2][2] - p1[1][2],
		pt1.Y*p1[2][3] - p1[1][3],
	})
	setRow(2, [4]float64{
		pt2.X*p2[2][0] - p2[0][0],
		pt2.X*p2[2][1] - p2[0][1],
		pt2.X*p2[2][2] - p2[0][2],
		pt2.X*p2[2][3] - p2[0][3],
	})
	setRow(3, [4]float64{
		pt2.Y*p2[2][0] - p2[1][0],
		pt2.Y*p2[2][1] - p2[1][1],
		pt2.Y*p2[2][2] - p2[1][2],
		pt2.Y*p2[2][3] - p2[1][3],
	})
	x := smallestEigenvector(a.gram())
	w := x[3]
	if math.Abs(w) < 1e-300 {
		w = 1e-300
	}
	return [3]float64{x[0] / w, x[1] / w, x[2] / w}
}

// TriangulatePoints reconstructs many 3D points from two views by applying
// [Triangulate] to each correspondence. It panics if the point slices differ in
// length.
func TriangulatePoints(p1, p2 [3][4]float64, pts1, pts2 []cv.Point2f) [][3]float64 {
	if len(pts1) != len(pts2) {
		panic("calib2: TriangulatePoints length mismatch")
	}
	out := make([][3]float64, len(pts1))
	for i := range pts1 {
		out[i] = Triangulate(p1, p2, pts1[i], pts2[i])
	}
	return out
}

// EssentialMatrix returns the essential matrix E = [T]ₓ·R for the relative pose
// (R, T) that maps the first camera's coordinates to the second camera's
// coordinates. It encodes the epipolar geometry of a calibrated stereo pair.
func EssentialMatrix(pose Pose) [3][3]float64 {
	return Mat3Mul(skew3(pose.T), pose.R)
}

// FundamentalMatrix returns the fundamental matrix F = K2⁻ᵀ·E·K1⁻¹ relating
// pixel coordinates in the two views, where E is the essential matrix of the
// relative pose and k1, k2 are the intrinsics of the first and second camera.
func FundamentalMatrix(k1, k2 CameraMatrix, pose Pose) [3][3]float64 {
	e := EssentialMatrix(pose)
	k1inv := k1.Inverse()
	k2invT := Mat3Transpose(k2.Inverse())
	return Mat3Mul(Mat3Mul(k2invT, e), k1inv)
}

// DecomposeEssentialMatrix factors an essential matrix into the two possible
// rotations and the (sign-ambiguous) unit translation direction of the relative
// pose, following Hartley & Zisserman. The four physically distinct
// (rotation, translation) combinations are (r1, +t), (r1, -t), (r2, +t) and
// (r2, -t); a cheirality test on triangulated points selects the correct one.
func DecomposeEssentialMatrix(e [3][3]float64) (r1, r2 [3][3]float64, t [3]float64) {
	u, _, v := svd3(e)
	// Enforce proper rotations for u and v.
	if Mat3Det(u) < 0 {
		for i := 0; i < 3; i++ {
			u[i][2] = -u[i][2]
		}
	}
	if Mat3Det(v) < 0 {
		for i := 0; i < 3; i++ {
			v[i][2] = -v[i][2]
		}
	}
	w := [3][3]float64{{0, -1, 0}, {1, 0, 0}, {0, 0, 1}}
	vt := Mat3Transpose(v)
	r1 = Mat3Mul(Mat3Mul(u, w), vt)
	r2 = Mat3Mul(Mat3Mul(u, Mat3Transpose(w)), vt)
	if Mat3Det(r1) < 0 {
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				r1[i][j] = -r1[i][j]
			}
		}
	}
	if Mat3Det(r2) < 0 {
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				r2[i][j] = -r2[i][j]
			}
		}
	}
	t = [3]float64{u[0][2], u[1][2], u[2][2]}
	return r1, r2, t
}

// ComputeCorrespondEpilines maps image points from one view to their epipolar
// lines in the other view via the fundamental matrix f. whichImage selects the
// source view: 1 means the points are in the first image and the lines lie in
// the second (l = F·p); 2 means the points are in the second image and the
// lines lie in the first (l = Fᵀ·p). Each returned line is the coefficient
// triple (a, b, c) of a·x + b·y + c = 0, normalized so that a² + b² = 1.
func ComputeCorrespondEpilines(points []cv.Point2f, whichImage int, f [3][3]float64) [][3]float64 {
	m := f
	if whichImage == 2 {
		m = Mat3Transpose(f)
	}
	out := make([][3]float64, len(points))
	for i, p := range points {
		l := Mat3VecMul(m, [3]float64{p.X, p.Y, 1})
		n := math.Hypot(l[0], l[1])
		if n > 1e-300 {
			l[0] /= n
			l[1] /= n
			l[2] /= n
		}
		out[i] = l
	}
	return out
}

// StereoRectification holds the result of [StereoRectify]: the rectifying
// rotations R1, R2 to apply to each camera (via [InitUndistortRectifyMap]), the
// rectified 3×4 projection matrices P1, P2, the 4×4 disparity-to-depth mapping
// Q and the stereo baseline (distance between the camera centers) in world
// units.
type StereoRectification struct {
	R1       [3][3]float64 // R1 is the rectifying rotation for the first camera.
	R2       [3][3]float64 // R2 is the rectifying rotation for the second camera.
	P1       [3][4]float64 // P1 is the rectified 3×4 projection matrix for the first camera.
	P2       [3][4]float64 // P2 is the rectified 3×4 projection matrix for the second camera.
	Q        [4][4]float64 // Q is the 4×4 disparity-to-depth mapping matrix.
	Baseline float64       // Baseline is the distance between the camera centers in world units.
}

// StereoRectify computes the rectification transforms for a calibrated stereo
// pair. pose is the relative pose (R, T) mapping first-camera coordinates to
// second-camera coordinates. The result aligns both image planes so that
// corresponding points share the same row, using a common new camera matrix
// taken from k1. The returned [StereoRectification] carries the per-camera
// rectifying rotations, rectified projection matrices and the disparity-to-depth
// matrix Q. It assumes a non-degenerate baseline; a zero baseline yields
// identity rotations.
func StereoRectify(k1, k2 CameraMatrix, pose Pose) StereoRectification {
	// Baseline vector in the first camera's frame: b = -Rᵀ·T (center of cam2).
	rt := Mat3Transpose(pose.R)
	b := Vec3Scale(Mat3VecMul(rt, pose.T), -1)
	baseline := Vec3Norm(b)
	knew := k1
	if baseline < 1e-12 {
		return StereoRectification{
			R1:       Mat3Identity(),
			R2:       pose.R,
			P1:       ProjectionMatrix(knew, Pose{R: Mat3Identity()}),
			P2:       ProjectionMatrix(knew, Pose{R: Mat3Identity(), T: [3]float64{}}),
			Q:        disparityToDepthMatrix(knew, 1),
			Baseline: 0,
		}
	}
	// New common orientation: x along the baseline, y perpendicular to it and
	// to the first camera's optical axis, z completing the right-handed frame.
	v1 := Vec3Normalize(b)
	oldZ := [3]float64{0, 0, 1}
	v2 := Vec3Normalize(Vec3Cross(oldZ, v1))
	v3 := Vec3Cross(v1, v2)
	rn := [3][3]float64{
		{v1[0], v1[1], v1[2]},
		{v2[0], v2[1], v2[2]},
		{v3[0], v3[1], v3[2]},
	}
	// Rectifying rotation per camera: R_rect_i = Rn · R_world_iᵀ.
	// Camera 1 has world rotation I, camera 2 has world rotation R.
	r1 := rn
	r2 := Mat3Mul(rn, Mat3Transpose(pose.R))
	// Rectified projection matrices: both use Rn; camera 2 shifted by baseline
	// along the new x-axis.
	p1 := ProjectionMatrix(knew, Pose{R: rn, T: [3]float64{}})
	p2 := ProjectionMatrix(knew, Pose{R: rn, T: [3]float64{-baseline, 0, 0}})
	return StereoRectification{
		R1:       r1,
		R2:       r2,
		P1:       p1,
		P2:       p2,
		Q:        disparityToDepthMatrix(knew, baseline),
		Baseline: baseline,
	}
}

// disparityToDepthMatrix builds the 4×4 matrix Q that maps a rectified pixel
// and its disparity to a homogeneous 3D point, assuming both rectified cameras
// share the intrinsic knew and are separated by the given baseline along x.
func disparityToDepthMatrix(knew CameraMatrix, baseline float64) [4][4]float64 {
	invB := 0.0
	if baseline != 0 {
		invB = 1 / baseline
	}
	return [4][4]float64{
		{1, 0, 0, -knew.Cx},
		{0, 1, 0, -knew.Cy},
		{0, 0, 0, knew.Fx},
		{0, 0, invB, 0},
	}
}

// DisparityTo3D reprojects a single rectified pixel (u, v) with its disparity to
// a 3D point using the disparity-to-depth matrix q from [StereoRectify]. The
// point is in the rectified first camera's frame; the z component is its depth.
func DisparityTo3D(u, v, disparity float64, q [4][4]float64) [3]float64 {
	in := [4]float64{u, v, disparity, 1}
	var out [4]float64
	for i := 0; i < 4; i++ {
		out[i] = q[i][0]*in[0] + q[i][1]*in[1] + q[i][2]*in[2] + q[i][3]*in[3]
	}
	w := out[3]
	if math.Abs(w) < 1e-300 {
		w = 1e-300
	}
	return [3]float64{out[0] / w, out[1] / w, out[2] / w}
}

// ReprojectImageTo3D reprojects an entire disparity map to 3D points using the
// disparity-to-depth matrix q. disparity[v][u] is the disparity at pixel
// (u, v); the result has the same shape, with result[v][u] the reconstructed
// point. Pixels with non-positive disparity are set to the zero point, matching
// the usual "invalid" convention.
func ReprojectImageTo3D(disparity [][]float64, q [4][4]float64) [][][3]float64 {
	out := make([][][3]float64, len(disparity))
	for v := range disparity {
		out[v] = make([][3]float64, len(disparity[v]))
		for u := range disparity[v] {
			d := disparity[v][u]
			if d <= 0 {
				continue
			}
			out[v][u] = DisparityTo3D(float64(u), float64(v), d, q)
		}
	}
	return out
}

// DepthFromDisparity returns the metric depth Z = focal·baseline/disparity for a
// rectified stereo pair. It returns +Inf for non-positive disparity, which
// corresponds to an infinitely distant (or invalid) point.
func DepthFromDisparity(disparity, focal, baseline float64) float64 {
	if disparity <= 0 {
		return math.Inf(1)
	}
	return focal * baseline / disparity
}
