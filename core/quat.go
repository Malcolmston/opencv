package core

import (
	"fmt"
	"math"
)

// Quatd is a double-precision quaternion w + xi + yj + zk, mirroring cv::Quatd.
// It is the standard representation for 3D rotations.
type Quatd struct {
	W float64
	X float64
	Y float64
	Z float64
}

// NewQuatd builds a Quatd from its four components.
func NewQuatd(w, x, y, z float64) Quatd { return Quatd{w, x, y, z} }

// QuatdIdentity returns the identity rotation quaternion (1, 0, 0, 0).
func QuatdIdentity() Quatd { return Quatd{1, 0, 0, 0} }

// Add returns the component-wise sum q+o.
func (q Quatd) Add(o Quatd) Quatd { return Quatd{q.W + o.W, q.X + o.X, q.Y + o.Y, q.Z + o.Z} }

// Sub returns the component-wise difference q-o.
func (q Quatd) Sub(o Quatd) Quatd { return Quatd{q.W - o.W, q.X - o.X, q.Y - o.Y, q.Z - o.Z} }

// ScalarMul returns the quaternion scaled by s.
func (q Quatd) ScalarMul(s float64) Quatd { return Quatd{q.W * s, q.X * s, q.Y * s, q.Z * s} }

// Mul returns the Hamilton product q*o.
func (q Quatd) Mul(o Quatd) Quatd {
	return Quatd{
		q.W*o.W - q.X*o.X - q.Y*o.Y - q.Z*o.Z,
		q.W*o.X + q.X*o.W + q.Y*o.Z - q.Z*o.Y,
		q.W*o.Y - q.X*o.Z + q.Y*o.W + q.Z*o.X,
		q.W*o.Z + q.X*o.Y - q.Y*o.X + q.Z*o.W,
	}
}

// Conjugate returns the conjugate (w, -x, -y, -z).
func (q Quatd) Conjugate() Quatd { return Quatd{q.W, -q.X, -q.Y, -q.Z} }

// Dot returns the 4D dot product q·o.
func (q Quatd) Dot(o Quatd) float64 { return q.W*o.W + q.X*o.X + q.Y*o.Y + q.Z*o.Z }

// NormSq returns the squared norm |q|².
func (q Quatd) NormSq() float64 { return q.Dot(q) }

// Norm returns the quaternion norm |q|.
func (q Quatd) Norm() float64 { return math.Sqrt(q.NormSq()) }

// Normalize returns the unit quaternion; a zero quaternion is returned as-is.
func (q Quatd) Normalize() Quatd {
	n := q.Norm()
	if n == 0 {
		return q
	}
	return q.ScalarMul(1 / n)
}

// Inverse returns the multiplicative inverse conj(q)/|q|².
func (q Quatd) Inverse() Quatd {
	n := q.NormSq()
	if n == 0 {
		return q
	}
	return q.Conjugate().ScalarMul(1 / n)
}

// Equals reports whether q and o have identical components.
func (q Quatd) Equals(o Quatd) bool { return q.W == o.W && q.X == o.X && q.Y == o.Y && q.Z == o.Z }

// ToRotationMatrix returns the 3x3 rotation matrix of the unit quaternion. The
// quaternion is normalized first.
func (q Quatd) ToRotationMatrix() Matx33d {
	u := q.Normalize()
	w, x, y, z := u.W, u.X, u.Y, u.Z
	return Matx33d{
		1 - 2*(y*y+z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
		2 * (x*y + z*w), 1 - 2*(x*x+z*z), 2 * (y*z - x*w),
		2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2*(x*x+y*y),
	}
}

// RotateVector applies the rotation encoded by q to vector v.
func (q Quatd) RotateVector(v Vec3d) Vec3d { return q.ToRotationMatrix().MulVec(v) }

// String renders the quaternion.
func (q Quatd) String() string { return fmt.Sprintf("%g%+gi%+gj%+gk", q.W, q.X, q.Y, q.Z) }

// QuatdFromRotationMatrix builds a unit quaternion from a 3x3 rotation matrix
// using the numerically stable branch of Shepperd's method.
func QuatdFromRotationMatrix(m Matx33d) Quatd {
	tr := m[0] + m[4] + m[8]
	var q Quatd
	switch {
	case tr > 0:
		s := math.Sqrt(tr+1) * 2
		q = Quatd{0.25 * s, (m[7] - m[5]) / s, (m[2] - m[6]) / s, (m[3] - m[1]) / s}
	case m[0] > m[4] && m[0] > m[8]:
		s := math.Sqrt(1+m[0]-m[4]-m[8]) * 2
		q = Quatd{(m[7] - m[5]) / s, 0.25 * s, (m[1] + m[3]) / s, (m[2] + m[6]) / s}
	case m[4] > m[8]:
		s := math.Sqrt(1+m[4]-m[0]-m[8]) * 2
		q = Quatd{(m[2] - m[6]) / s, (m[1] + m[3]) / s, 0.25 * s, (m[5] + m[7]) / s}
	default:
		s := math.Sqrt(1+m[8]-m[0]-m[4]) * 2
		q = Quatd{(m[3] - m[1]) / s, (m[2] + m[6]) / s, (m[5] + m[7]) / s, 0.25 * s}
	}
	return q.Normalize()
}

// QuatdFromAxisAngle builds a unit quaternion rotating by angle radians about
// the given axis (which need not be normalized).
func QuatdFromAxisAngle(axis Vec3d, angle float64) Quatd {
	n := axis.Norm()
	if n == 0 {
		return QuatdIdentity()
	}
	half := angle / 2
	s := math.Sin(half) / n
	return Quatd{math.Cos(half), axis[0] * s, axis[1] * s, axis[2] * s}
}

// Slerpd performs spherical linear interpolation between unit quaternions a and
// b by parameter t in [0, 1].
func Slerpd(a, b Quatd, t float64) Quatd {
	a = a.Normalize()
	b = b.Normalize()
	dot := a.Dot(b)
	if dot < 0 {
		b = b.ScalarMul(-1)
		dot = -dot
	}
	if dot > 0.9995 {
		return a.Add(b.Sub(a).ScalarMul(t)).Normalize()
	}
	theta0 := math.Acos(dot)
	theta := theta0 * t
	sinTheta0 := math.Sin(theta0)
	s0 := math.Cos(theta) - dot*math.Sin(theta)/sinTheta0
	s1 := math.Sin(theta) / sinTheta0
	return a.ScalarMul(s0).Add(b.ScalarMul(s1))
}

// Quatf is the float32 analogue of Quatd, mirroring cv::Quatf.
type Quatf struct {
	W float32
	X float32
	Y float32
	Z float32
}

// NewQuatf builds a Quatf from its four components.
func NewQuatf(w, x, y, z float32) Quatf { return Quatf{w, x, y, z} }

// QuatfIdentity returns the identity rotation quaternion (1, 0, 0, 0).
func QuatfIdentity() Quatf { return Quatf{1, 0, 0, 0} }

// Add returns the component-wise sum q+o.
func (q Quatf) Add(o Quatf) Quatf { return Quatf{q.W + o.W, q.X + o.X, q.Y + o.Y, q.Z + o.Z} }

// Mul returns the Hamilton product q*o.
func (q Quatf) Mul(o Quatf) Quatf {
	return Quatf{
		q.W*o.W - q.X*o.X - q.Y*o.Y - q.Z*o.Z,
		q.W*o.X + q.X*o.W + q.Y*o.Z - q.Z*o.Y,
		q.W*o.Y - q.X*o.Z + q.Y*o.W + q.Z*o.X,
		q.W*o.Z + q.X*o.Y - q.Y*o.X + q.Z*o.W,
	}
}

// Conjugate returns the conjugate (w, -x, -y, -z).
func (q Quatf) Conjugate() Quatf { return Quatf{q.W, -q.X, -q.Y, -q.Z} }

// Norm returns the quaternion norm |q|.
func (q Quatf) Norm() float64 {
	return math.Sqrt(float64(q.W*q.W + q.X*q.X + q.Y*q.Y + q.Z*q.Z))
}

// Normalize returns the unit quaternion; a zero quaternion is returned as-is.
func (q Quatf) Normalize() Quatf {
	n := q.Norm()
	if n == 0 {
		return q
	}
	f := float32(1 / n)
	return Quatf{q.W * f, q.X * f, q.Y * f, q.Z * f}
}

// ToQuatd converts to a double-precision quaternion.
func (q Quatf) ToQuatd() Quatd { return Quatd{float64(q.W), float64(q.X), float64(q.Y), float64(q.Z)} }

// String renders the quaternion.
func (q Quatf) String() string { return fmt.Sprintf("%g%+gi%+gj%+gk", q.W, q.X, q.Y, q.Z) }
