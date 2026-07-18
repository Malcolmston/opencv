package calib2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// CameraMatrix holds the intrinsic parameters of a pinhole camera: the focal
// lengths Fx, Fy in pixels, the principal point (Cx, Cy) in pixels and an
// optional axis Skew (usually zero). It corresponds to the 3×3 intrinsic matrix
//
//	[ Fx  Skew  Cx ]
//	[  0   Fy   Cy ]
//	[  0    0    1 ]
type CameraMatrix struct {
	Fx   float64 // Fx is the horizontal focal length in pixels.
	Fy   float64 // Fy is the vertical focal length in pixels.
	Cx   float64 // Cx is the x coordinate of the principal point in pixels.
	Cy   float64 // Cy is the y coordinate of the principal point in pixels.
	Skew float64 // Skew is the axis-skew coefficient (usually zero).
}

// Matrix returns the intrinsic parameters as a 3×3 row-major matrix.
func (c CameraMatrix) Matrix() [3][3]float64 {
	return [3][3]float64{
		{c.Fx, c.Skew, c.Cx},
		{0, c.Fy, c.Cy},
		{0, 0, 1},
	}
}

// Inverse returns the inverse of the intrinsic matrix. It panics if either
// focal length is zero, which would make the camera matrix singular.
func (c CameraMatrix) Inverse() [3][3]float64 {
	if c.Fx == 0 || c.Fy == 0 {
		panic("calib2: CameraMatrix.Inverse with zero focal length")
	}
	inv, ok := Mat3Inverse(c.Matrix())
	if !ok {
		panic("calib2: singular CameraMatrix")
	}
	return inv
}

// NewCameraMatrix builds a [CameraMatrix] from a 3×3 intrinsic matrix,
// extracting the focal lengths, skew and principal point.
func NewCameraMatrix(k [3][3]float64) CameraMatrix {
	return CameraMatrix{Fx: k[0][0], Fy: k[1][1], Cx: k[0][2], Cy: k[1][2], Skew: k[0][1]}
}

// DistortionCoeffs holds the Brown–Conrady lens-distortion coefficients: the
// radial terms K1, K2, K3 and the tangential terms P1, P2. The zero value
// describes a distortion-free lens.
type DistortionCoeffs struct {
	K1 float64 // K1 is the first radial-distortion coefficient.
	K2 float64 // K2 is the second radial-distortion coefficient.
	P1 float64 // P1 is the first tangential-distortion coefficient.
	P2 float64 // P2 is the second tangential-distortion coefficient.
	K3 float64 // K3 is the third radial-distortion coefficient.
}

// Slice returns the coefficients in OpenCV's canonical ordering
// [K1, K2, P1, P2, K3].
func (d DistortionCoeffs) Slice() []float64 {
	return []float64{d.K1, d.K2, d.P1, d.P2, d.K3}
}

// IsZero reports whether all coefficients are zero, i.e. the lens is modelled
// as distortion-free.
func (d DistortionCoeffs) IsZero() bool {
	return d.K1 == 0 && d.K2 == 0 && d.P1 == 0 && d.P2 == 0 && d.K3 == 0
}

// NewDistortionCoeffs builds a [DistortionCoeffs] from a coefficient slice in
// OpenCV's [K1, K2, P1, P2, K3] ordering. A nil or short slice leaves the
// remaining coefficients at zero.
func NewDistortionCoeffs(s []float64) DistortionCoeffs {
	get := func(i int) float64 {
		if i < len(s) {
			return s[i]
		}
		return 0
	}
	return DistortionCoeffs{K1: get(0), K2: get(1), P1: get(2), P2: get(3), K3: get(4)}
}

// Pose is a rigid-body transform from world (object) coordinates to camera
// coordinates, expressed as a 3×3 rotation R and a translation T such that a
// world point X maps to camera coordinates R·X + T.
type Pose struct {
	R [3][3]float64 // R is the 3×3 rotation matrix from world to camera coordinates.
	T [3]float64    // T is the translation vector from world to camera coordinates.
}

// NewPoseFromRvec builds a [Pose] from an axis-angle rotation vector and a
// translation vector.
func NewPoseFromRvec(rvec, tvec [3]float64) Pose {
	return Pose{R: Rodrigues(rvec), T: tvec}
}

// Rvec returns the rotation of the pose as an axis-angle rotation vector.
func (p Pose) Rvec() [3]float64 {
	return RodriguesInverse(p.R)
}

// Apply transforms a world point into camera coordinates: R·X + T.
func (p Pose) Apply(x [3]float64) [3]float64 {
	return Vec3Add(Mat3VecMul(p.R, x), p.T)
}

// Inverse returns the inverse rigid transform, mapping camera coordinates back
// to world coordinates.
func (p Pose) Inverse() Pose {
	rt := Mat3Transpose(p.R)
	return Pose{R: rt, T: Vec3Scale(Mat3VecMul(rt, p.T), -1)}
}

// Matrix returns the pose as a 3×4 [R|T] matrix in row-major order.
func (p Pose) Matrix() [3][4]float64 {
	var out [3][4]float64
	for i := 0; i < 3; i++ {
		out[i][0] = p.R[i][0]
		out[i][1] = p.R[i][1]
		out[i][2] = p.R[i][2]
		out[i][3] = p.T[i]
	}
	return out
}

// distortNormalized applies the Brown–Conrady model to a normalized image
// coordinate (x, y), returning the distorted normalized coordinate.
func distortNormalized(x, y float64, d DistortionCoeffs) (float64, float64) {
	r2 := x*x + y*y
	radial := 1 + d.K1*r2 + d.K2*r2*r2 + d.K3*r2*r2*r2
	xd := x*radial + 2*d.P1*x*y + d.P2*(r2+2*x*x)
	yd := y*radial + d.P1*(r2+2*y*y) + 2*d.P2*x*y
	return xd, yd
}

// ProjectPoint projects a single 3D world point into the image through the full
// pinhole model: it is transformed by the pose, distorted by the lens model and
// scaled by the intrinsics. Points at or behind the camera plane (z ≤ 0) are
// still projected mathematically; callers that need visibility must test z
// themselves.
func ProjectPoint(objectPoint [3]float64, pose Pose, k CameraMatrix, d DistortionCoeffs) cv.Point2f {
	cam := pose.Apply(objectPoint)
	z := cam[2]
	if z == 0 {
		z = 1e-12
	}
	x := cam[0] / z
	y := cam[1] / z
	xd, yd := distortNormalized(x, y, d)
	return cv.Point2f{
		X: k.Fx*xd + k.Skew*yd + k.Cx,
		Y: k.Fy*yd + k.Cy,
	}
}

// ProjectPoints projects a slice of 3D world points into the image through the
// full pinhole model, applying the pose, distortion and intrinsics to each
// point. The result has one image point per input point, in the same order.
func ProjectPoints(objectPoints [][3]float64, pose Pose, k CameraMatrix, d DistortionCoeffs) []cv.Point2f {
	out := make([]cv.Point2f, len(objectPoints))
	for i, p := range objectPoints {
		out[i] = ProjectPoint(p, pose, k, d)
	}
	return out
}

// ProjectionMatrix returns the 3×4 camera projection matrix P = K·[R|T] that
// maps homogeneous world points to homogeneous image points.
func ProjectionMatrix(k CameraMatrix, pose Pose) [3][4]float64 {
	km := k.Matrix()
	rt := pose.Matrix()
	var out [3][4]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			var s float64
			for l := 0; l < 3; l++ {
				s += km[i][l] * rt[l][j]
			}
			out[i][j] = s
		}
	}
	return out
}

// DecomposeProjectionMatrix factors a 3×4 projection matrix P into an intrinsic
// [CameraMatrix], a rotation R and a translation T such that P is proportional
// to K·[R|T]. The left 3×3 block is separated by RQ decomposition with sign
// normalization so that the focal lengths are positive and R is a proper
// rotation; the scale is fixed by normalizing the intrinsic (3,3) entry to 1.
func DecomposeProjectionMatrix(p [3][4]float64) (k CameraMatrix, r [3][3]float64, t [3]float64) {
	var m [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			m[i][j] = p[i][j]
		}
	}
	kk, rr := rq3(m)
	// Normalize so that kk[2][2] == 1, keeping the sign.
	scale := kk[2][2]
	if scale != 0 {
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				kk[i][j] /= scale
			}
		}
	}
	// Recover translation from the fourth column: t = K⁻¹ · (p[:,3] / scale).
	kinv, _ := Mat3Inverse(kk)
	col := [3]float64{p[0][3] / scale, p[1][3] / scale, p[2][3] / scale}
	t = Mat3VecMul(kinv, col)
	// If the recovered rotation is improper, flip its sign together with t.
	if Mat3Det(rr) < 0 {
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				rr[i][j] = -rr[i][j]
			}
		}
		t = Vec3Scale(t, -1)
	}
	return NewCameraMatrix(kk), rr, t
}

// rq3 performs an RQ decomposition of a 3×3 matrix, returning an upper-
// triangular R and an orthogonal Q with A = R·Q, normalized so the diagonal of
// R is positive.
func rq3(a [3][3]float64) (rMat, qMat [3][3]float64) {
	mul := func(x, y [3][3]float64) [3][3]float64 { return Mat3Mul(x, y) }
	// Zero a[2][1] with a rotation about x (post-multiplication).
	den := math.Hypot(a[2][2], a[2][1])
	c, s := 1.0, 0.0
	if den > 1e-300 {
		c = a[2][2] / den
		s = -a[2][1] / den
	}
	qx := [3][3]float64{{1, 0, 0}, {0, c, -s}, {0, s, c}}
	r := mul(a, qx)
	// Zero r[2][0] with a rotation about y.
	den = math.Hypot(r[2][2], r[2][0])
	c, s = 1.0, 0.0
	if den > 1e-300 {
		c = r[2][2] / den
		s = r[2][0] / den
	}
	qy := [3][3]float64{{c, 0, s}, {0, 1, 0}, {-s, 0, c}}
	r = mul(r, qy)
	// Zero r[1][0] with a rotation about z.
	den = math.Hypot(r[1][1], r[1][0])
	c, s = 1.0, 0.0
	if den > 1e-300 {
		c = r[1][1] / den
		s = -r[1][0] / den
	}
	qz := [3][3]float64{{c, -s, 0}, {s, c, 0}, {0, 0, 1}}
	r = mul(r, qz)
	q := mul(mul(Mat3Transpose(qz), Mat3Transpose(qy)), Mat3Transpose(qx))
	// Force positive diagonal of R by flipping column i of R and row i of Q.
	for i := 0; i < 3; i++ {
		if r[i][i] < 0 {
			for k := 0; k < 3; k++ {
				r[k][i] = -r[k][i]
				q[i][k] = -q[i][k]
			}
		}
	}
	return r, q
}
