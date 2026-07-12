package calib3d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// CameraMatrix holds the intrinsic parameters of a pinhole camera: the focal
// lengths Fx, Fy (in pixels) and the principal point (Cx, Cy). It corresponds to
// the 3×3 intrinsic matrix
//
//	[ Fx  0  Cx ]
//	[  0 Fy  Cy ]
//	[  0  0   1 ]
type CameraMatrix struct {
	Fx float64
	Fy float64
	Cx float64
	Cy float64
}

// Matrix returns the intrinsic parameters as a 3×3 matrix in row-major order.
func (c CameraMatrix) Matrix() [3][3]float64 {
	return [3][3]float64{
		{c.Fx, 0, c.Cx},
		{0, c.Fy, c.Cy},
		{0, 0, 1},
	}
}

// NewCameraMatrix builds a [CameraMatrix] from a 3×3 intrinsic matrix,
// extracting the focal lengths and principal point.
func NewCameraMatrix(k [3][3]float64) CameraMatrix {
	return CameraMatrix{Fx: k[0][0], Fy: k[1][1], Cx: k[0][2], Cy: k[1][2]}
}

// DistCoeffs holds the Brown–Conrady lens distortion coefficients: the radial
// terms K1, K2, K3 and the tangential terms P1, P2. The zero value describes a
// distortion-free lens.
type DistCoeffs struct {
	K1 float64
	K2 float64
	P1 float64
	P2 float64
	K3 float64
}

// Slice returns the coefficients in OpenCV's canonical ordering
// [K1, K2, P1, P2, K3], suitable for passing to [ProjectPoints] and [Undistort].
func (d DistCoeffs) Slice() []float64 {
	return []float64{d.K1, d.K2, d.P1, d.P2, d.K3}
}

// distParams unpacks a distortion coefficient slice into named radial and
// tangential terms, following OpenCV's [K1, K2, P1, P2, K3] ordering. A nil or
// short slice leaves the remaining coefficients at zero, so a nil slice means no
// distortion.
func distParams(dist []float64) (k1, k2, p1, p2, k3 float64) {
	get := func(i int) float64 {
		if i < len(dist) {
			return dist[i]
		}
		return 0
	}
	return get(0), get(1), get(2), get(3), get(4)
}

// distortNormalized applies the Brown–Conrady distortion model to a normalized
// image coordinate (x, y), returning the distorted normalized coordinate.
func distortNormalized(x, y, k1, k2, p1, p2, k3 float64) (xd, yd float64) {
	r2 := x*x + y*y
	radial := 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
	xd = x*radial + 2*p1*x*y + p2*(r2+2*x*x)
	yd = y*radial + p1*(r2+2*y*y) + 2*p2*x*y
	return xd, yd
}

// RodriguesToMatrix converts a rotation vector (axis-angle, whose direction is
// the rotation axis and whose magnitude is the rotation angle in radians) into
// the corresponding 3×3 rotation matrix, using Rodrigues' formula. A zero vector
// yields the identity.
func RodriguesToMatrix(rvec [3]float64) [3][3]float64 {
	theta := norm3(rvec)
	if theta < 1e-12 {
		return [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	}
	kx := rvec[0] / theta
	ky := rvec[1] / theta
	kz := rvec[2] / theta
	c := math.Cos(theta)
	s := math.Sin(theta)
	c1 := 1 - c
	// R = I·c + (1-c)·k·kᵀ + s·[k]ₓ
	return [3][3]float64{
		{c + kx*kx*c1, kx*ky*c1 - kz*s, kx*kz*c1 + ky*s},
		{ky*kx*c1 + kz*s, c + ky*ky*c1, ky*kz*c1 - kx*s},
		{kz*kx*c1 - ky*s, kz*ky*c1 + kx*s, c + kz*kz*c1},
	}
}

// RodriguesToVector converts a 3×3 rotation matrix into the equivalent rotation
// vector (axis-angle) via the inverse of Rodrigues' formula. It handles the
// small-angle and near-π degenerate cases. It is the exact inverse of
// [RodriguesToMatrix] up to the usual axis-angle ambiguity at θ = π.
func RodriguesToVector(r [3][3]float64) [3]float64 {
	trace := r[0][0] + r[1][1] + r[2][2]
	cosTheta := (trace - 1) / 2
	if cosTheta > 1 {
		cosTheta = 1
	} else if cosTheta < -1 {
		cosTheta = -1
	}
	theta := math.Acos(cosTheta)
	if theta < 1e-9 {
		return [3]float64{0, 0, 0}
	}
	if math.Pi-theta < 1e-6 {
		// Near θ = π: R ≈ I + 2·a·aᵀ, so recover the axis from the diagonal.
		xx := (r[0][0] + 1) / 2
		yy := (r[1][1] + 1) / 2
		zz := (r[2][2] + 1) / 2
		xy := (r[0][1] + r[1][0]) / 4
		xz := (r[0][2] + r[2][0]) / 4
		yz := (r[1][2] + r[2][1]) / 4
		var ax, ay, az float64
		switch {
		case xx >= yy && xx >= zz:
			ax = math.Sqrt(math.Max(xx, 0))
			ay = xy / ax
			az = xz / ax
		case yy >= zz:
			ay = math.Sqrt(math.Max(yy, 0))
			ax = xy / ay
			az = yz / ay
		default:
			az = math.Sqrt(math.Max(zz, 0))
			ax = xz / az
			ay = yz / az
		}
		n := math.Sqrt(ax*ax + ay*ay + az*az)
		if n < 1e-15 {
			return [3]float64{0, 0, 0}
		}
		return [3]float64{theta * ax / n, theta * ay / n, theta * az / n}
	}
	k := theta / (2 * math.Sin(theta))
	return [3]float64{
		k * (r[2][1] - r[1][2]),
		k * (r[0][2] - r[2][0]),
		k * (r[1][0] - r[0][1]),
	}
}

// ProjectPoints projects a set of 3D object points into the image plane of a
// camera. The object points objPts are given in world coordinates; rvec and
// tvec are the rotation vector (see [RodriguesToMatrix]) and translation that
// take world points into the camera frame; K is the 3×3 intrinsic matrix; and
// dist holds the distortion coefficients in [K1, K2, P1, P2, K3] order (a nil
// slice means no distortion).
//
// Each object point is rotated and translated into the camera frame, projected
// onto the normalized image plane, distorted with the Brown–Conrady model, and
// scaled by the intrinsics. The resulting pixel coordinates are rounded to the
// nearest integer [cv.Point]. Points at or behind the camera plane (Z ≤ 0) are
// projected using their homogeneous division as-is; callers that care should
// filter such points beforehand.
func ProjectPoints(objPts [][3]float64, rvec, tvec [3]float64, K [3][3]float64, dist []float64) []cv.Point {
	r := RodriguesToMatrix(rvec)
	k1, k2, p1, p2, k3 := distParams(dist)
	fx, fy, cx, cy := K[0][0], K[1][1], K[0][2], K[1][2]
	out := make([]cv.Point, len(objPts))
	for i, X := range objPts {
		// Transform into the camera frame.
		xc := r[0][0]*X[0] + r[0][1]*X[1] + r[0][2]*X[2] + tvec[0]
		yc := r[1][0]*X[0] + r[1][1]*X[1] + r[1][2]*X[2] + tvec[1]
		zc := r[2][0]*X[0] + r[2][1]*X[1] + r[2][2]*X[2] + tvec[2]
		if zc == 0 {
			zc = 1e-15
		}
		x := xc / zc
		y := yc / zc
		xd, yd := distortNormalized(x, y, k1, k2, p1, p2, k3)
		u := fx*xd + cx
		v := fy*yd + cy
		out[i] = cv.Point{X: int(math.Round(u)), Y: int(math.Round(v))}
	}
	return out
}
