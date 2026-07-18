package matching2

import (
	"math"

	"github.com/malcolmston/opencv/core"
)

// SolvePnPDLT estimates the camera pose (rotation R and translation t) that maps
// the 3-D world points objPts to the 2-D image points imgPts under the intrinsic
// matrix K, using the Direct Linear Transform. It needs at least six
// correspondences and assumes the points are not all coplanar. A world point X
// projects as K·(R·X + t). It reports false with too few points, a singular K or
// a degenerate configuration. R is orthonormalized to a proper rotation
// (determinant +1).
func SolvePnPDLT(objPts []core.Point3d, imgPts []core.Point2d, K [3][3]float64) (R [3][3]float64, t [3]float64, ok bool) {
	n := len(objPts)
	if n != len(imgPts) || n < 6 {
		return Mat3Identity(), [3]float64{}, false
	}
	Kinv, inv := Mat3Inverse(K)
	if !inv {
		return Mat3Identity(), [3]float64{}, false
	}
	norm := normalizeByKinv(imgPts, Kinv)
	a := make([][]float64, 0, 2*n)
	for i := 0; i < n; i++ {
		X, Y, Z := objPts[i].X, objPts[i].Y, objPts[i].Z
		x, y := norm[i].X, norm[i].Y
		a = append(a,
			[]float64{X, Y, Z, 1, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x},
			[]float64{0, 0, 0, 0, X, Y, Z, 1, -y * X, -y * Y, -y * Z, -y},
		)
	}
	p := matching2nullVector(a)
	P := [3][4]float64{
		{p[0], p[1], p[2], p[3]},
		{p[4], p[5], p[6], p[7]},
		{p[8], p[9], p[10], p[11]},
	}
	// Fix the overall sign so reconstructed depths are positive (points in front).
	var sw float64
	for i := 0; i < n; i++ {
		X := objPts[i]
		sw += P[2][0]*X.X + P[2][1]*X.Y + P[2][2]*X.Z + P[2][3]
	}
	if sw < 0 {
		for i := 0; i < 3; i++ {
			for j := 0; j < 4; j++ {
				P[i][j] = -P[i][j]
			}
		}
	}
	M := [3][3]float64{
		{P[0][0], P[0][1], P[0][2]},
		{P[1][0], P[1][1], P[1][2]},
		{P[2][0], P[2][1], P[2][2]},
	}
	u, s, v := matching2svd3(M)
	scale := (s[0] + s[1] + s[2]) / 3
	if scale < 1e-12 {
		return Mat3Identity(), [3]float64{}, false
	}
	uvt := Mat3Mul(u, Mat3Transpose(v))
	d := 1.0
	if Mat3Det(uvt) < 0 {
		d = -1.0
	}
	diag := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, d}}
	R = Mat3Mul(u, Mat3Mul(diag, Mat3Transpose(v)))
	t = [3]float64{P[0][3] / scale, P[1][3] / scale, P[2][3] / scale}
	if !matching2finite9(R) {
		return Mat3Identity(), [3]float64{}, false
	}
	return R, t, true
}

// ProjectPoints projects world points into the image under pose (R, t) and
// intrinsics K, returning one image point per world point. Point i is
// K·(R·objPtsᵢ + t), dehomogenized.
func ProjectPoints(objPts []core.Point3d, R [3][3]float64, t [3]float64, K [3][3]float64) []core.Point2d {
	out := make([]core.Point2d, len(objPts))
	for i, X := range objPts {
		cam := [3]float64{
			R[0][0]*X.X + R[0][1]*X.Y + R[0][2]*X.Z + t[0],
			R[1][0]*X.X + R[1][1]*X.Y + R[1][2]*X.Z + t[1],
			R[2][0]*X.X + R[2][1]*X.Y + R[2][2]*X.Z + t[2],
		}
		img := Mat3VecMul(K, cam)
		if img[2] == 0 {
			out[i] = core.Point2d{}
			continue
		}
		out[i] = core.Point2d{X: img[0] / img[2], Y: img[1] / img[2]}
	}
	return out
}

// ReprojectionErrors returns the per-point reprojection error, in pixels,
// between the observed image points and the projections of the world points
// under pose (R, t) and intrinsics K. objPts, imgPts must have equal length.
func ReprojectionErrors(objPts []core.Point3d, imgPts []core.Point2d, R [3][3]float64, t [3]float64, K [3][3]float64) []float64 {
	proj := ProjectPoints(objPts, R, t, K)
	out := make([]float64, len(imgPts))
	for i := range imgPts {
		out[i] = math.Hypot(proj[i].X-imgPts[i].X, proj[i].Y-imgPts[i].Y)
	}
	return out
}

// MeanReprojectionError returns the mean of [ReprojectionErrors], a single-number
// summary of pose quality. It returns 0 for empty input.
func MeanReprojectionError(objPts []core.Point3d, imgPts []core.Point2d, R [3][3]float64, t [3]float64, K [3][3]float64) float64 {
	errs := ReprojectionErrors(objPts, imgPts, R, t, K)
	if len(errs) == 0 {
		return 0
	}
	var s float64
	for _, e := range errs {
		s += e
	}
	return s / float64(len(errs))
}

// SolvePnPRansac robustly estimates the camera pose from correspondences that
// may contain outliers. A correspondence is an inlier when its reprojection
// error is at most threshold pixels. iters bounds the number of random six-point
// samples and seed makes the result deterministic. The final pose is refit over
// all inliers. Model is packed as [Pose]; Ok is false when no sample yields at
// least six inliers.
func SolvePnPRansac(objPts []core.Point3d, imgPts []core.Point2d, K [3][3]float64, threshold float64, iters int, seed int64) RANSACResult[Pose] {
	var empty RANSACResult[Pose]
	n := len(objPts)
	if n != len(imgPts) || n < 6 {
		return empty
	}
	fit := func(sample []int) (Pose, bool) {
		op := make([]core.Point3d, len(sample))
		ip := make([]core.Point2d, len(sample))
		for i, idx := range sample {
			op[i] = objPts[idx]
			ip[i] = imgPts[idx]
		}
		R, t, ok := SolvePnPDLT(op, ip, K)
		return Pose{R: R, T: t}, ok
	}
	inliers := func(pose Pose) []bool {
		errs := ReprojectionErrors(objPts, imgPts, pose.R, pose.T, K)
		mask := make([]bool, n)
		for i, e := range errs {
			mask[i] = e <= threshold
		}
		return mask
	}
	return RANSAC(n, 6, iters, 6, seed, fit, inliers, fit)
}

// Pose bundles a camera rotation and translation, the [SolvePnPRansac] model
// type. A world point X projects as K·(R·X + T).
type Pose struct {
	// R is the 3×3 rotation matrix (determinant +1).
	R [3][3]float64
	// T is the translation vector.
	T [3]float64
}

// RodriguesToMatrix converts an axis-angle rotation vector (whose direction is
// the rotation axis and whose magnitude is the rotation angle in radians) into a
// 3×3 rotation matrix, via the Rodrigues formula. A zero vector yields the
// identity.
func RodriguesToMatrix(rvec [3]float64) [3][3]float64 {
	theta := matching2norm3(rvec)
	if theta < 1e-15 {
		return Mat3Identity()
	}
	kx, ky, kz := rvec[0]/theta, rvec[1]/theta, rvec[2]/theta
	c, s := math.Cos(theta), math.Sin(theta)
	v := 1 - c
	return [3][3]float64{
		{c + kx*kx*v, kx*ky*v - kz*s, kx*kz*v + ky*s},
		{ky*kx*v + kz*s, c + ky*ky*v, ky*kz*v - kx*s},
		{kz*kx*v - ky*s, kz*ky*v + kx*s, c + kz*kz*v},
	}
}

// MatrixToRodrigues converts a 3×3 rotation matrix into its axis-angle rotation
// vector, the inverse of [RodriguesToMatrix]. The returned vector's magnitude is
// the rotation angle in radians and its direction is the axis.
func MatrixToRodrigues(R [3][3]float64) [3]float64 {
	tr := R[0][0] + R[1][1] + R[2][2]
	cosTheta := (tr - 1) / 2
	cosTheta = math.Max(-1, math.Min(1, cosTheta))
	theta := math.Acos(cosTheta)
	if theta < 1e-12 {
		return [3]float64{}
	}
	if math.Abs(theta-math.Pi) < 1e-6 {
		// Near 180°: extract axis from the diagonal of R + I.
		axis := [3]float64{
			math.Sqrt(math.Max(0, (R[0][0]+1)/2)),
			math.Sqrt(math.Max(0, (R[1][1]+1)/2)),
			math.Sqrt(math.Max(0, (R[2][2]+1)/2)),
		}
		if R[0][1]+R[1][0] < 0 {
			axis[1] = -axis[1]
		}
		if R[0][2]+R[2][0] < 0 {
			axis[2] = -axis[2]
		}
		nrm := matching2norm3(axis)
		if nrm > 0 {
			axis[0] *= theta / nrm
			axis[1] *= theta / nrm
			axis[2] *= theta / nrm
		}
		return axis
	}
	f := theta / (2 * math.Sin(theta))
	return [3]float64{
		(R[2][1] - R[1][2]) * f,
		(R[0][2] - R[2][0]) * f,
		(R[1][0] - R[0][1]) * f,
	}
}
