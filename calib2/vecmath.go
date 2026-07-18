package calib2

import "math"

// Vec3Add returns the element-wise sum a+b.
func Vec3Add(a, b [3]float64) [3]float64 {
	return [3]float64{a[0] + b[0], a[1] + b[1], a[2] + b[2]}
}

// Vec3Sub returns the element-wise difference a-b.
func Vec3Sub(a, b [3]float64) [3]float64 {
	return [3]float64{a[0] - b[0], a[1] - b[1], a[2] - b[2]}
}

// Vec3Scale returns the vector a scaled by the scalar s.
func Vec3Scale(a [3]float64, s float64) [3]float64 {
	return [3]float64{a[0] * s, a[1] * s, a[2] * s}
}

// Vec3Dot returns the dot product a·b.
func Vec3Dot(a, b [3]float64) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

// Vec3Cross returns the cross product a×b.
func Vec3Cross(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

// Vec3Norm returns the Euclidean length of a.
func Vec3Norm(a [3]float64) float64 {
	return math.Sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
}

// Vec3Normalize returns a unit vector in the direction of a. It returns the
// zero vector when a has zero length.
func Vec3Normalize(a [3]float64) [3]float64 {
	n := Vec3Norm(a)
	if n < 1e-300 {
		return [3]float64{}
	}
	return [3]float64{a[0] / n, a[1] / n, a[2] / n}
}

// Mat3Identity returns the 3×3 identity matrix.
func Mat3Identity() [3][3]float64 {
	return [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
}

// Mat3Mul returns the matrix product a·b of two 3×3 matrices.
func Mat3Mul(a, b [3][3]float64) [3][3]float64 {
	var out [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			out[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j]
		}
	}
	return out
}

// Mat3VecMul returns the matrix–vector product a·v for a 3×3 matrix and a
// 3-vector.
func Mat3VecMul(a [3][3]float64, v [3]float64) [3]float64 {
	return [3]float64{
		a[0][0]*v[0] + a[0][1]*v[1] + a[0][2]*v[2],
		a[1][0]*v[0] + a[1][1]*v[1] + a[1][2]*v[2],
		a[2][0]*v[0] + a[2][1]*v[1] + a[2][2]*v[2],
	}
}

// Mat3Transpose returns the transpose of a 3×3 matrix.
func Mat3Transpose(a [3][3]float64) [3][3]float64 {
	return [3][3]float64{
		{a[0][0], a[1][0], a[2][0]},
		{a[0][1], a[1][1], a[2][1]},
		{a[0][2], a[1][2], a[2][2]},
	}
}

// Mat3Det returns the determinant of a 3×3 matrix.
func Mat3Det(a [3][3]float64) float64 {
	return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1]) -
		a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0]) +
		a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
}

// Mat3Inverse returns the inverse of a 3×3 matrix and reports ok=false when the
// matrix is singular to working precision.
func Mat3Inverse(a [3][3]float64) (inv [3][3]float64, ok bool) {
	det := Mat3Det(a)
	if math.Abs(det) < 1e-300 {
		return inv, false
	}
	id := 1 / det
	inv[0][0] = (a[1][1]*a[2][2] - a[1][2]*a[2][1]) * id
	inv[0][1] = (a[0][2]*a[2][1] - a[0][1]*a[2][2]) * id
	inv[0][2] = (a[0][1]*a[1][2] - a[0][2]*a[1][1]) * id
	inv[1][0] = (a[1][2]*a[2][0] - a[1][0]*a[2][2]) * id
	inv[1][1] = (a[0][0]*a[2][2] - a[0][2]*a[2][0]) * id
	inv[1][2] = (a[0][2]*a[1][0] - a[0][0]*a[1][2]) * id
	inv[2][0] = (a[1][0]*a[2][1] - a[1][1]*a[2][0]) * id
	inv[2][1] = (a[0][1]*a[2][0] - a[0][0]*a[2][1]) * id
	inv[2][2] = (a[0][0]*a[1][1] - a[0][1]*a[1][0]) * id
	return inv, true
}

// skew3 returns the 3×3 skew-symmetric "cross-product matrix" [v]ₓ such that
// [v]ₓ·w = v×w for any w.
func skew3(v [3]float64) [3][3]float64 {
	return [3][3]float64{
		{0, -v[2], v[1]},
		{v[2], 0, -v[0]},
		{-v[1], v[0], 0},
	}
}
