package calib2

import "math"

// Rodrigues converts a rotation vector (axis-angle, whose direction is the
// rotation axis and whose magnitude is the rotation angle in radians) into the
// corresponding 3×3 rotation matrix, using Rodrigues' formula. A zero vector
// yields the identity matrix.
func Rodrigues(rvec [3]float64) [3][3]float64 {
	theta := Vec3Norm(rvec)
	if theta < 1e-15 {
		return Mat3Identity()
	}
	kx := rvec[0] / theta
	ky := rvec[1] / theta
	kz := rvec[2] / theta
	c := math.Cos(theta)
	s := math.Sin(theta)
	c1 := 1 - c
	return [3][3]float64{
		{c + kx*kx*c1, kx*ky*c1 - kz*s, kx*kz*c1 + ky*s},
		{ky*kx*c1 + kz*s, c + ky*ky*c1, ky*kz*c1 - kx*s},
		{kz*kx*c1 - ky*s, kz*ky*c1 + kx*s, c + kz*kz*c1},
	}
}

// RodriguesInverse converts a 3×3 rotation matrix into the equivalent rotation
// vector (axis-angle) via the inverse of Rodrigues' formula. It handles the
// small-angle and near-π degenerate cases and is the exact inverse of
// [Rodrigues] up to the usual axis-angle ambiguity at θ = π.
func RodriguesInverse(r [3][3]float64) [3]float64 {
	trace := r[0][0] + r[1][1] + r[2][2]
	cosTheta := (trace - 1) / 2
	if cosTheta > 1 {
		cosTheta = 1
	} else if cosTheta < -1 {
		cosTheta = -1
	}
	theta := math.Acos(cosTheta)
	if theta < 1e-9 {
		return [3]float64{}
	}
	if math.Pi-theta < 1e-6 {
		// Near θ = π: R ≈ I + 2·a·aᵀ; recover the axis from the diagonal.
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
		return [3]float64{ax * theta, ay * theta, az * theta}
	}
	s := 2 * math.Sin(theta)
	return [3]float64{
		(r[2][1] - r[1][2]) / s * theta,
		(r[0][2] - r[2][0]) / s * theta,
		(r[1][0] - r[0][1]) / s * theta,
	}
}

// IsRotationMatrix reports whether r is a proper rotation matrix to the given
// tolerance: its transpose is its inverse (rᵀr = I) and its determinant is +1.
func IsRotationMatrix(r [3][3]float64, tol float64) bool {
	rt := Mat3Transpose(r)
	p := Mat3Mul(rt, r)
	id := Mat3Identity()
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math.Abs(p[i][j]-id[i][j]) > tol {
				return false
			}
		}
	}
	return math.Abs(Mat3Det(r)-1) <= tol
}

// RotationX returns the 3×3 matrix of a rotation by theta radians about the
// x-axis.
func RotationX(theta float64) [3][3]float64 {
	c, s := math.Cos(theta), math.Sin(theta)
	return [3][3]float64{{1, 0, 0}, {0, c, -s}, {0, s, c}}
}

// RotationY returns the 3×3 matrix of a rotation by theta radians about the
// y-axis.
func RotationY(theta float64) [3][3]float64 {
	c, s := math.Cos(theta), math.Sin(theta)
	return [3][3]float64{{c, 0, s}, {0, 1, 0}, {-s, 0, c}}
}

// RotationZ returns the 3×3 matrix of a rotation by theta radians about the
// z-axis.
func RotationZ(theta float64) [3][3]float64 {
	c, s := math.Cos(theta), math.Sin(theta)
	return [3][3]float64{{c, -s, 0}, {s, c, 0}, {0, 0, 1}}
}

// EulerToRotation builds a rotation matrix from intrinsic Tait–Bryan angles in
// the Z-Y-X convention: R = Rz(yaw)·Ry(pitch)·Rx(roll). Angles are in radians.
func EulerToRotation(roll, pitch, yaw float64) [3][3]float64 {
	return Mat3Mul(Mat3Mul(RotationZ(yaw), RotationY(pitch)), RotationX(roll))
}

// RotationToEuler extracts intrinsic Tait–Bryan angles (roll about x, pitch
// about y, yaw about z) from a rotation matrix, inverting [EulerToRotation]. At
// the gimbal-lock singularity (|pitch| = π/2) roll is set to zero and yaw
// absorbs the remaining rotation. Angles are returned in radians.
func RotationToEuler(r [3][3]float64) (roll, pitch, yaw float64) {
	sy := -r[2][0]
	if sy > 1 {
		sy = 1
	} else if sy < -1 {
		sy = -1
	}
	pitch = math.Asin(sy)
	if math.Abs(r[2][0]) < 1-1e-9 {
		roll = math.Atan2(r[2][1], r[2][2])
		yaw = math.Atan2(r[1][0], r[0][0])
	} else {
		roll = 0
		yaw = math.Atan2(-r[0][1], r[1][1])
	}
	return roll, pitch, yaw
}

// RotationToQuaternion converts a rotation matrix into a unit quaternion in
// (w, x, y, z) order, using the numerically stable branch selection of Shepperd.
func RotationToQuaternion(r [3][3]float64) [4]float64 {
	trace := r[0][0] + r[1][1] + r[2][2]
	var q [4]float64
	switch {
	case trace > 0:
		s := math.Sqrt(trace+1) * 2
		q[0] = 0.25 * s
		q[1] = (r[2][1] - r[1][2]) / s
		q[2] = (r[0][2] - r[2][0]) / s
		q[3] = (r[1][0] - r[0][1]) / s
	case r[0][0] > r[1][1] && r[0][0] > r[2][2]:
		s := math.Sqrt(1+r[0][0]-r[1][1]-r[2][2]) * 2
		q[0] = (r[2][1] - r[1][2]) / s
		q[1] = 0.25 * s
		q[2] = (r[0][1] + r[1][0]) / s
		q[3] = (r[0][2] + r[2][0]) / s
	case r[1][1] > r[2][2]:
		s := math.Sqrt(1+r[1][1]-r[0][0]-r[2][2]) * 2
		q[0] = (r[0][2] - r[2][0]) / s
		q[1] = (r[0][1] + r[1][0]) / s
		q[2] = 0.25 * s
		q[3] = (r[1][2] + r[2][1]) / s
	default:
		s := math.Sqrt(1+r[2][2]-r[0][0]-r[1][1]) * 2
		q[0] = (r[1][0] - r[0][1]) / s
		q[1] = (r[0][2] + r[2][0]) / s
		q[2] = (r[1][2] + r[2][1]) / s
		q[3] = 0.25 * s
	}
	return NormalizeQuaternion(q)
}

// QuaternionToRotation converts a quaternion in (w, x, y, z) order into a 3×3
// rotation matrix. The quaternion is normalized internally, so non-unit input
// is accepted.
func QuaternionToRotation(q [4]float64) [3][3]float64 {
	n := NormalizeQuaternion(q)
	w, x, y, z := n[0], n[1], n[2], n[3]
	return [3][3]float64{
		{1 - 2*(y*y+z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)},
		{2 * (x*y + z*w), 1 - 2*(x*x+z*z), 2 * (y*z - x*w)},
		{2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2*(x*x+y*y)},
	}
}

// NormalizeQuaternion returns q scaled to unit length in (w, x, y, z) order. A
// zero quaternion is returned as the identity rotation (1, 0, 0, 0).
func NormalizeQuaternion(q [4]float64) [4]float64 {
	n := math.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
	if n < 1e-300 {
		return [4]float64{1, 0, 0, 0}
	}
	return [4]float64{q[0] / n, q[1] / n, q[2] / n, q[3] / n}
}

// ComposeRotations returns the rotation matrix equivalent to applying rotation
// b first and then a, i.e. the product a·b.
func ComposeRotations(a, b [3][3]float64) [3][3]float64 {
	return Mat3Mul(a, b)
}
