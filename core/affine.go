package core

import (
	"fmt"
	"math"
)

// Affine3d is a 3D affine transform stored as a 3x3 rotation/scale matrix and a
// translation vector, mirroring cv::Affine3d. It maps a point p to R*p + T.
type Affine3d struct {
	R Matx33d
	T Vec3d
}

// NewAffine3d builds an Affine3d from a linear matrix and a translation.
func NewAffine3d(r Matx33d, t Vec3d) Affine3d { return Affine3d{r, t} }

// Affine3dIdentity returns the identity transform.
func Affine3dIdentity() Affine3d { return Affine3d{Matx33dIdentity(), Vec3d{}} }

// Rotation returns the linear (rotation/scale) part of the transform.
func (a Affine3d) Rotation() Matx33d { return a.R }

// Translation returns the translation part of the transform.
func (a Affine3d) Translation() Vec3d { return a.T }

// TransformPoint applies the transform to point p (R*p + T).
func (a Affine3d) TransformPoint(p Point3d) Point3d {
	v := a.R.MulVec(Vec3d{p.X, p.Y, p.Z})
	return Point3d{v[0] + a.T[0], v[1] + a.T[1], v[2] + a.T[2]}
}

// TransformVec applies only the linear part to v (no translation).
func (a Affine3d) TransformVec(v Vec3d) Vec3d { return a.R.MulVec(v) }

// Concatenate returns the composition a∘o, the transform that first applies o
// then a.
func (a Affine3d) Concatenate(o Affine3d) Affine3d {
	r := a.R.Mul(o.R)
	t := a.R.MulVec(o.T)
	return Affine3d{r, Vec3d{t[0] + a.T[0], t[1] + a.T[1], t[2] + a.T[2]}}
}

// Inv returns the inverse transform. When the linear part is singular the
// identity is returned.
func (a Affine3d) Inv() Affine3d {
	inv, ok := a.R.Inverse()
	if !ok {
		return Affine3dIdentity()
	}
	t := inv.MulVec(a.T)
	return Affine3d{inv, Vec3d{-t[0], -t[1], -t[2]}}
}

// Translate returns the transform with an extra translation applied after it.
func (a Affine3d) Translate(t Vec3d) Affine3d {
	return Affine3d{a.R, Vec3d{a.T[0] + t[0], a.T[1] + t[1], a.T[2] + t[2]}}
}

// Rotate returns the transform with an extra rotation r applied before it.
func (a Affine3d) Rotate(r Matx33d) Affine3d { return Affine3d{a.R.Mul(r), a.T} }

// ToMatx34d returns the transform as a 3x4 matrix [R | T].
func (a Affine3d) ToMatx34d() Matx34d {
	return Matx34d{
		a.R[0], a.R[1], a.R[2], a.T[0],
		a.R[3], a.R[4], a.R[5], a.T[1],
		a.R[6], a.R[7], a.R[8], a.T[2],
	}
}

// String renders the transform.
func (a Affine3d) String() string { return fmt.Sprintf("Affine3d{R:%v T:%v}", a.R, a.T) }

// RodriguesToMatrixd converts a rotation vector (axis scaled by angle in
// radians) to a 3x3 rotation matrix using Rodrigues' formula, mirroring
// cv::Rodrigues.
func RodriguesToMatrixd(rvec Vec3d) Matx33d {
	theta := rvec.Norm()
	if theta < 1e-15 {
		return Matx33dIdentity()
	}
	kx, ky, kz := rvec[0]/theta, rvec[1]/theta, rvec[2]/theta
	c, s := math.Cos(theta), math.Sin(theta)
	v := 1 - c
	return Matx33d{
		c + kx*kx*v, kx*ky*v - kz*s, kx*kz*v + ky*s,
		ky*kx*v + kz*s, c + ky*ky*v, ky*kz*v - kx*s,
		kz*kx*v - ky*s, kz*ky*v + kx*s, c + kz*kz*v,
	}
}

// MatrixToRodriguesd converts a 3x3 rotation matrix to a rotation vector (axis
// scaled by angle in radians), the inverse of RodriguesToMatrixd.
func MatrixToRodriguesd(r Matx33d) Vec3d {
	// Rotation angle from the trace.
	tr := (r[0] + r[4] + r[8] - 1) / 2
	if tr > 1 {
		tr = 1
	} else if tr < -1 {
		tr = -1
	}
	theta := math.Acos(tr)
	if theta < 1e-15 {
		return Vec3d{}
	}
	if math.Abs(theta-math.Pi) < 1e-6 {
		// Near 180°: extract axis from the symmetric part.
		ax := math.Sqrt(math.Max(0, (r[0]+1)/2))
		ay := math.Sqrt(math.Max(0, (r[4]+1)/2))
		az := math.Sqrt(math.Max(0, (r[8]+1)/2))
		if r[1]+r[3] < 0 {
			ay = -ay
		}
		if r[2]+r[6] < 0 {
			az = -az
		}
		return Vec3d{ax * theta, ay * theta, az * theta}
	}
	k := theta / (2 * math.Sin(theta))
	return Vec3d{(r[7] - r[5]) * k, (r[2] - r[6]) * k, (r[3] - r[1]) * k}
}

// Affine3f is the float32 analogue of Affine3d, mirroring cv::Affine3f.
type Affine3f struct {
	R Matx33f
	T Vec3f
}

// NewAffine3f builds an Affine3f from a linear matrix and a translation.
func NewAffine3f(r Matx33f, t Vec3f) Affine3f { return Affine3f{r, t} }

// Affine3fIdentity returns the identity transform.
func Affine3fIdentity() Affine3f { return Affine3f{Matx33fIdentity(), Vec3f{}} }

// Rotation returns the linear part of the transform.
func (a Affine3f) Rotation() Matx33f { return a.R }

// Translation returns the translation part of the transform.
func (a Affine3f) Translation() Vec3f { return a.T }

// TransformPoint applies the transform to point p (R*p + T).
func (a Affine3f) TransformPoint(p Point3f) Point3f {
	v := a.R.MulVec(Vec3f{p.X, p.Y, p.Z})
	return Point3f{v[0] + a.T[0], v[1] + a.T[1], v[2] + a.T[2]}
}

// TransformVec applies only the linear part to v.
func (a Affine3f) TransformVec(v Vec3f) Vec3f { return a.R.MulVec(v) }

// Concatenate returns the composition a∘o (apply o then a).
func (a Affine3f) Concatenate(o Affine3f) Affine3f {
	r := a.R.Mul(o.R)
	t := a.R.MulVec(o.T)
	return Affine3f{r, Vec3f{t[0] + a.T[0], t[1] + a.T[1], t[2] + a.T[2]}}
}

// Inv returns the inverse transform, or the identity when singular.
func (a Affine3f) Inv() Affine3f {
	inv, ok := a.R.Inverse()
	if !ok {
		return Affine3fIdentity()
	}
	t := inv.MulVec(a.T)
	return Affine3f{inv, Vec3f{-t[0], -t[1], -t[2]}}
}

// ToMatx34f returns the transform as a 3x4 matrix [R | T].
func (a Affine3f) ToMatx34f() Matx34f {
	return Matx34f{
		a.R[0], a.R[1], a.R[2], a.T[0],
		a.R[3], a.R[4], a.R[5], a.T[1],
		a.R[6], a.R[7], a.R[8], a.T[2],
	}
}

// String renders the transform.
func (a Affine3f) String() string { return fmt.Sprintf("Affine3f{R:%v T:%v}", a.R, a.T) }
