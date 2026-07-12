package rgbd

import "math"

// Pose is a rigid body transform in 3-D: a rotation R and a translation T that
// act on a point as p' = R·p + T. It is the transform type used by the
// odometry, warping and volumetric routines. The zero value is not a valid
// transform (its rotation is all zeros); use [IdentityPose] for the identity.
type Pose struct {
	R [3][3]float64
	T [3]float64
}

// IdentityPose returns the identity transform (R = I, T = 0), which maps every
// point to itself. It is the usual starting estimate for the odometry solvers.
func IdentityPose() Pose {
	return Pose{R: [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}
}

// PoseFromRt builds a [Pose] from an explicit rotation matrix and translation
// vector. It does not orthonormalise R; callers that assemble a rotation from
// noisy data should project it first (for example via [Rodrigues] of a rotation
// vector).
func PoseFromRt(r [3][3]float64, t [3]float64) Pose {
	return Pose{R: r, T: t}
}

// Apply transforms the point p by the pose, returning R·p + T.
func (p Pose) Apply(x [3]float64) [3]float64 {
	return add3(matVec3(p.R, x), p.T)
}

// Compose returns the composition p∘b, the transform that applies b first and
// then p: (p∘b)(x) = p.Apply(b.Apply(x)). Its rotation is p.R·b.R and its
// translation is p.R·b.T + p.T.
func (p Pose) Compose(b Pose) Pose {
	return Pose{
		R: mul3(p.R, b.R),
		T: add3(matVec3(p.R, b.T), p.T),
	}
}

// Inverse returns the inverse transform, which undoes p: its rotation is Rᵀ and
// its translation is −Rᵀ·T, so p.Inverse().Apply(p.Apply(x)) == x up to
// round-off.
func (p Pose) Inverse() Pose {
	rt := transpose3(p.R)
	return Pose{R: rt, T: scale3(matVec3(rt, p.T), -1)}
}

// Rodrigues converts a rotation vector (axis-angle, whose direction is the
// rotation axis and whose length is the rotation angle in radians) to a 3×3
// rotation matrix via the exponential map. A zero vector yields the identity;
// small angles use the first-order form to stay well conditioned. It is the
// inverse of [InverseRodrigues] and is used to turn the twist increments solved
// by the odometry into concrete rotations.
func Rodrigues(rvec [3]float64) [3][3]float64 {
	theta := norm3(rvec)
	ident := [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	if theta < 1e-12 {
		// First-order R ≈ I + [rvec]ₓ for a tiny rotation.
		return [3][3]float64{
			{1, -rvec[2], rvec[1]},
			{rvec[2], 1, -rvec[0]},
			{-rvec[1], rvec[0], 1},
		}
	}
	a := scale3(rvec, 1/theta)
	c := math.Cos(theta)
	s := math.Sin(theta)
	// K is the cross-product matrix of the unit axis a.
	k := [3][3]float64{
		{0, -a[2], a[1]},
		{a[2], 0, -a[0]},
		{-a[1], a[0], 0},
	}
	var r [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			r[i][j] = ident[i][j]*c + (1-c)*a[i]*a[j] + s*k[i][j]
		}
	}
	return r
}

// InverseRodrigues converts a 3×3 rotation matrix to its rotation vector
// (axis-angle) representation, the inverse of [Rodrigues]. The returned vector's
// length is the rotation angle in [0, π] and its direction is the axis. It is
// used to report or measure the rotation recovered by the odometry.
func InverseRodrigues(r [3][3]float64) [3]float64 {
	trace := r[0][0] + r[1][1] + r[2][2]
	cosT := (trace - 1) / 2
	if cosT > 1 {
		cosT = 1
	}
	if cosT < -1 {
		cosT = -1
	}
	theta := math.Acos(cosT)
	if theta < 1e-9 {
		// Near identity: the skew part is already the rotation vector.
		return [3]float64{
			0.5 * (r[2][1] - r[1][2]),
			0.5 * (r[0][2] - r[2][0]),
			0.5 * (r[1][0] - r[0][1]),
		}
	}
	if math.Abs(math.Pi-theta) < 1e-6 {
		// Angle near π: sin(θ) ≈ 0, recover the axis from the diagonal.
		axis := [3]float64{
			math.Sqrt(math.Max(0, (r[0][0]+1)/2)),
			math.Sqrt(math.Max(0, (r[1][1]+1)/2)),
			math.Sqrt(math.Max(0, (r[2][2]+1)/2)),
		}
		// Fix the relative signs from the off-diagonal entries.
		if r[2][1]-r[1][2] < 0 {
			axis[0] = -axis[0]
		}
		if r[0][2]-r[2][0] < 0 {
			axis[1] = -axis[1]
		}
		if r[1][0]-r[0][1] < 0 {
			axis[2] = -axis[2]
		}
		return scale3(normalize3(axis), theta)
	}
	f := theta / (2 * math.Sin(theta))
	return [3]float64{
		f * (r[2][1] - r[1][2]),
		f * (r[0][2] - r[2][0]),
		f * (r[1][0] - r[0][1]),
	}
}
