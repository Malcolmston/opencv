package surface_matching

import "math"

// This file provides the small, self-contained linear-algebra kernels the
// surface_matching routines need: 3-vector and 3×3 matrix helpers, a symmetric
// Jacobi eigensolver, a 3×3 singular value decomposition, and quaternion
// conversions for rotation averaging. Everything here is implemented locally so
// the package depends only on the Go standard library (plus the root cv package
// elsewhere); it pulls in no sibling cv/* subpackage.

// Vec3 is a point or vector in right-handed 3-D space.
type Vec3 = [3]float64

// Mat3 is a row-major 3×3 matrix. m[r][c] is the entry in row r, column c.
type Mat3 = [3][3]float64

// identity3 returns the 3×3 identity matrix.
func identity3() Mat3 {
	return Mat3{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
}

// mul3 returns the matrix product a·b of two 3×3 matrices.
func mul3(a, b Mat3) Mat3 {
	var out Mat3
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			out[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j]
		}
	}
	return out
}

// transpose3 returns the transpose of a 3×3 matrix.
func transpose3(a Mat3) Mat3 {
	var out Mat3
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			out[i][j] = a[j][i]
		}
	}
	return out
}

// matVec3 returns the matrix-vector product a·v.
func matVec3(a Mat3, v Vec3) Vec3 {
	return Vec3{
		a[0][0]*v[0] + a[0][1]*v[1] + a[0][2]*v[2],
		a[1][0]*v[0] + a[1][1]*v[1] + a[1][2]*v[2],
		a[2][0]*v[0] + a[2][1]*v[1] + a[2][2]*v[2],
	}
}

// det3 returns the determinant of a 3×3 matrix.
func det3(m Mat3) float64 {
	return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) -
		m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) +
		m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
}

// add3 returns a+b for 3-vectors.
func add3(a, b Vec3) Vec3 { return Vec3{a[0] + b[0], a[1] + b[1], a[2] + b[2]} }

// sub3 returns a-b for 3-vectors.
func sub3(a, b Vec3) Vec3 { return Vec3{a[0] - b[0], a[1] - b[1], a[2] - b[2]} }

// scale3 returns s·v for a 3-vector.
func scale3(v Vec3, s float64) Vec3 { return Vec3{v[0] * s, v[1] * s, v[2] * s} }

// dot3 returns the dot product of two 3-vectors.
func dot3(a, b Vec3) float64 { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }

// cross3 returns the cross product a×b of two 3-vectors.
func cross3(a, b Vec3) Vec3 {
	return Vec3{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

// norm3 returns the Euclidean length of a 3-vector.
func norm3(v Vec3) float64 { return math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) }

// normalize3 returns v scaled to unit length. A zero (or near-zero) vector is
// returned unchanged so callers can detect the degenerate case.
func normalize3(v Vec3) Vec3 {
	n := norm3(v)
	if n < 1e-12 {
		return v
	}
	return Vec3{v[0] / n, v[1] / n, v[2] / n}
}

// signf returns -1, 0 or +1 following the sign of x.
func signf(x float64) float64 {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	default:
		return 0
	}
}

// rotationX returns the rotation matrix for a rotation of angle radians about
// the +x axis (right-handed).
func rotationX(angle float64) Mat3 {
	c := math.Cos(angle)
	s := math.Sin(angle)
	return Mat3{
		{1, 0, 0},
		{0, c, -s},
		{0, s, c},
	}
}

// rotationAxisAngle returns the rotation matrix that rotates by angle radians
// about the given (not necessarily unit) axis, via Rodrigues' formula.
func rotationAxisAngle(axis Vec3, angle float64) Mat3 {
	u := normalize3(axis)
	c := math.Cos(angle)
	s := math.Sin(angle)
	t := 1 - c
	x, y, z := u[0], u[1], u[2]
	return Mat3{
		{c + x*x*t, x*y*t - z*s, x*z*t + y*s},
		{y*x*t + z*s, c + y*y*t, y*z*t - x*s},
		{z*x*t - y*s, z*y*t + x*s, c + z*z*t},
	}
}

// alignToXAxis returns a proper rotation R such that R·n is the +x axis, for a
// unit vector n. The rows of R are (n, b, n×b) for a unit vector b
// perpendicular to n, which makes R·n = (n·n, b·n, (n×b)·n) = (1, 0, 0). The
// determinant is n·n = 1, so R is a right-handed rotation.
func alignToXAxis(n Vec3) Mat3 {
	n = normalize3(n)
	// Choose the world axis least parallel to n to build a stable perpendicular.
	var axis Vec3
	ax, ay, az := math.Abs(n[0]), math.Abs(n[1]), math.Abs(n[2])
	switch {
	case ax <= ay && ax <= az:
		axis = Vec3{1, 0, 0}
	case ay <= ax && ay <= az:
		axis = Vec3{0, 1, 0}
	default:
		axis = Vec3{0, 0, 1}
	}
	b := normalize3(cross3(n, axis))
	c := cross3(n, b) // already unit length
	return Mat3{
		{n[0], n[1], n[2]},
		{b[0], b[1], b[2]},
		{c[0], c[1], c[2]},
	}
}

// rotationAngle returns the geodesic angle in radians between two rotation
// matrices, i.e. the magnitude of the relative rotation aᵀ·b.
func rotationAngle(a, b Mat3) float64 {
	rel := mul3(transpose3(a), b)
	tr := rel[0][0] + rel[1][1] + rel[2][2]
	cosA := (tr - 1) / 2
	if cosA > 1 {
		cosA = 1
	}
	if cosA < -1 {
		cosA = -1
	}
	return math.Acos(cosA)
}

// jacobiEigenSym computes the eigenvalues and eigenvectors of a symmetric 3×3
// matrix by cyclic Jacobi rotation. The returned vecs matrix holds the
// eigenvectors as columns: vecs[r][c] is component r of the eigenvector whose
// eigenvalue is vals[c].
func jacobiEigenSym(a Mat3) (vals Vec3, vecs Mat3) {
	m := a
	var v Mat3
	v[0][0], v[1][1], v[2][2] = 1, 1, 1
	for sweep := 0; sweep < 100; sweep++ {
		off := 0.0
		for p := 0; p < 3; p++ {
			for q := p + 1; q < 3; q++ {
				off += m[p][q] * m[p][q]
			}
		}
		if off < 1e-30 {
			break
		}
		for p := 0; p < 3; p++ {
			for q := p + 1; q < 3; q++ {
				apq := m[p][q]
				if math.Abs(apq) < 1e-300 {
					continue
				}
				theta := (m[q][q] - m[p][p]) / (2 * apq)
				t := 1.0
				if theta != 0 {
					t = signf(theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				}
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				for k := 0; k < 3; k++ {
					mkp := m[k][p]
					mkq := m[k][q]
					m[k][p] = c*mkp - s*mkq
					m[k][q] = s*mkp + c*mkq
				}
				for k := 0; k < 3; k++ {
					mpk := m[p][k]
					mqk := m[q][k]
					m[p][k] = c*mpk - s*mqk
					m[q][k] = s*mpk + c*mqk
				}
				for k := 0; k < 3; k++ {
					vkp := v[k][p]
					vkq := v[k][q]
					v[k][p] = c*vkp - s*vkq
					v[k][q] = s*vkp + c*vkq
				}
			}
		}
	}
	vals = Vec3{m[0][0], m[1][1], m[2][2]}
	return vals, v
}

// svd3 computes the singular value decomposition A = U·diag(S)·Vᵀ of a 3×3
// matrix. The singular values S are returned in non-increasing order with the
// columns of U and V ordered to match. It underpins the rigid alignment used by
// [ICP] and by pose recovery.
func svd3(a Mat3) (u Mat3, s Vec3, v Mat3) {
	ata := mul3(transpose3(a), a)
	vals, vecs := jacobiEigenSym(ata)
	idx := [3]int{0, 1, 2}
	for i := 0; i < 3; i++ {
		for j := i + 1; j < 3; j++ {
			if vals[idx[j]] > vals[idx[i]] {
				idx[i], idx[j] = idx[j], idx[i]
			}
		}
	}
	for c := 0; c < 3; c++ {
		col := idx[c]
		ev := vals[col]
		if ev < 0 {
			ev = 0
		}
		s[c] = math.Sqrt(ev)
		for r := 0; r < 3; r++ {
			v[r][c] = vecs[r][col]
		}
	}
	for c := 0; c < 3; c++ {
		vc := Vec3{v[0][c], v[1][c], v[2][c]}
		av := matVec3(a, vc)
		if s[c] > 1e-12 {
			for r := 0; r < 3; r++ {
				u[r][c] = av[r] / s[c]
			}
		} else {
			u[0][c] = math.NaN() // marker; filled below
		}
	}
	for c := 0; c < 3; c++ {
		if !math.IsNaN(u[0][c]) {
			continue
		}
		a0 := Vec3{u[0][(c+1)%3], u[1][(c+1)%3], u[2][(c+1)%3]}
		a1 := Vec3{u[0][(c+2)%3], u[1][(c+2)%3], u[2][(c+2)%3]}
		cr := cross3(a1, a0)
		nn := norm3(cr)
		if nn < 1e-12 {
			cr = Vec3{0, 0, 0}
			cr[c] = 1
			nn = 1
		}
		for r := 0; r < 3; r++ {
			u[r][c] = cr[r] / nn
		}
	}
	return u, s, v
}

// quat is a unit quaternion (w, x, y, z) representing a rotation.
type quat struct{ w, x, y, z float64 }

// matToQuat converts a rotation matrix to a unit quaternion. It uses the
// numerically stable branch based on the largest diagonal term.
func matToQuat(m Mat3) quat {
	tr := m[0][0] + m[1][1] + m[2][2]
	var q quat
	switch {
	case tr > 0:
		s := math.Sqrt(tr+1) * 2 // s = 4w
		q.w = 0.25 * s
		q.x = (m[2][1] - m[1][2]) / s
		q.y = (m[0][2] - m[2][0]) / s
		q.z = (m[1][0] - m[0][1]) / s
	case m[0][0] > m[1][1] && m[0][0] > m[2][2]:
		s := math.Sqrt(1+m[0][0]-m[1][1]-m[2][2]) * 2 // s = 4x
		q.w = (m[2][1] - m[1][2]) / s
		q.x = 0.25 * s
		q.y = (m[0][1] + m[1][0]) / s
		q.z = (m[0][2] + m[2][0]) / s
	case m[1][1] > m[2][2]:
		s := math.Sqrt(1+m[1][1]-m[0][0]-m[2][2]) * 2 // s = 4y
		q.w = (m[0][2] - m[2][0]) / s
		q.x = (m[0][1] + m[1][0]) / s
		q.y = 0.25 * s
		q.z = (m[1][2] + m[2][1]) / s
	default:
		s := math.Sqrt(1+m[2][2]-m[0][0]-m[1][1]) * 2 // s = 4z
		q.w = (m[1][0] - m[0][1]) / s
		q.x = (m[0][2] + m[2][0]) / s
		q.y = (m[1][2] + m[2][1]) / s
		q.z = 0.25 * s
	}
	return q.normalized()
}

// quatToMat converts a unit quaternion to its rotation matrix.
func quatToMat(q quat) Mat3 {
	q = q.normalized()
	w, x, y, z := q.w, q.x, q.y, q.z
	return Mat3{
		{1 - 2*(y*y+z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)},
		{2 * (x*y + z*w), 1 - 2*(x*x+z*z), 2 * (y*z - x*w)},
		{2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2*(x*x+y*y)},
	}
}

// dot returns the 4-D dot product of two quaternions.
func (q quat) dot(o quat) float64 { return q.w*o.w + q.x*o.x + q.y*o.y + q.z*o.z }

// normalized returns the quaternion scaled to unit length; a degenerate zero
// quaternion is mapped to the identity rotation.
func (q quat) normalized() quat {
	n := math.Sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z)
	if n < 1e-12 {
		return quat{1, 0, 0, 0}
	}
	return quat{q.w / n, q.x / n, q.y / n, q.z / n}
}
