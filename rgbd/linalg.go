package rgbd

import "math"

// This file provides the small, self-contained linear-algebra kernels the rgbd
// routines need: 3×3 matrix and 3-vector helpers, a symmetric Jacobi
// eigensolver and a 3×3 singular value decomposition. They are implemented
// locally so the package depends only on the Go standard library and the root
// cv package.

// mul3 returns the matrix product a·b of two 3×3 matrices.
func mul3(a, b [3][3]float64) [3][3]float64 {
	var out [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			out[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j]
		}
	}
	return out
}

// transpose3 returns the transpose of a 3×3 matrix.
func transpose3(a [3][3]float64) [3][3]float64 {
	var out [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			out[i][j] = a[j][i]
		}
	}
	return out
}

// matVec3 returns the matrix-vector product a·v.
func matVec3(a [3][3]float64, v [3]float64) [3]float64 {
	return [3]float64{
		a[0][0]*v[0] + a[0][1]*v[1] + a[0][2]*v[2],
		a[1][0]*v[0] + a[1][1]*v[1] + a[1][2]*v[2],
		a[2][0]*v[0] + a[2][1]*v[1] + a[2][2]*v[2],
	}
}

// det3 returns the determinant of a 3×3 matrix.
func det3(m [3][3]float64) float64 {
	return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) -
		m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) +
		m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
}

// add3 returns a+b for 3-vectors.
func add3(a, b [3]float64) [3]float64 {
	return [3]float64{a[0] + b[0], a[1] + b[1], a[2] + b[2]}
}

// sub3 returns a-b for 3-vectors.
func sub3(a, b [3]float64) [3]float64 {
	return [3]float64{a[0] - b[0], a[1] - b[1], a[2] - b[2]}
}

// scale3 returns s·v for a 3-vector.
func scale3(v [3]float64, s float64) [3]float64 {
	return [3]float64{v[0] * s, v[1] * s, v[2] * s}
}

// dot3 returns the dot product of two 3-vectors.
func dot3(a, b [3]float64) float64 {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
}

// cross3 returns the cross product a×b of two 3-vectors.
func cross3(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

// norm3 returns the Euclidean length of a 3-vector.
func norm3(v [3]float64) float64 {
	return math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
}

// normalize3 returns v scaled to unit length. A zero (or near-zero) vector is
// returned unchanged so callers can detect the degenerate case.
func normalize3(v [3]float64) [3]float64 {
	n := norm3(v)
	if n < 1e-12 {
		return v
	}
	return [3]float64{v[0] / n, v[1] / n, v[2] / n}
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

// jacobiEigenSym computes the eigenvalues and eigenvectors of a symmetric 3×3
// matrix by cyclic Jacobi rotation. The returned vecs matrix holds the
// eigenvectors as columns: vecs[r][c] is component r of the eigenvector whose
// eigenvalue is vals[c].
func jacobiEigenSym(a [3][3]float64) (vals [3]float64, vecs [3][3]float64) {
	m := a
	var v [3][3]float64
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
	vals = [3]float64{m[0][0], m[1][1], m[2][2]}
	return vals, v
}

// svd3 computes the singular value decomposition A = U·diag(S)·Vᵀ of a 3×3
// matrix. The singular values S are returned in non-increasing order with the
// columns of U and V ordered to match. It underpins the rigid alignment in
// [ICP].
func svd3(a [3][3]float64) (u [3][3]float64, s [3]float64, v [3][3]float64) {
	// Eigen-decompose AᵀA (symmetric, positive semi-definite): its eigenvectors
	// are the right singular vectors and its eigenvalues the squared singular
	// values.
	ata := mul3(transpose3(a), a)
	vals, vecs := jacobiEigenSym(ata)
	// Sort eigenvalues (and their vectors) in descending order.
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
	// Recover the U columns as u_i = A·v_i / s_i. Where s_i is tiny, mark the
	// column for an orthonormal completion below.
	for c := 0; c < 3; c++ {
		vc := [3]float64{v[0][c], v[1][c], v[2][c]}
		av := matVec3(a, vc)
		if s[c] > 1e-12 {
			for r := 0; r < 3; r++ {
				u[r][c] = av[r] / s[c]
			}
		} else {
			u[0][c] = math.NaN() // marker; filled below
		}
	}
	// Complete degenerate U columns via cross products of the valid ones so U
	// stays orthonormal.
	for c := 0; c < 3; c++ {
		if !math.IsNaN(u[0][c]) {
			continue
		}
		a0 := [3]float64{u[0][(c+1)%3], u[1][(c+1)%3], u[2][(c+1)%3]}
		a1 := [3]float64{u[0][(c+2)%3], u[1][(c+2)%3], u[2][(c+2)%3]}
		cr := cross3(a1, a0)
		nn := norm3(cr)
		if nn < 1e-12 {
			cr = [3]float64{0, 0, 0}
			cr[c] = 1
			nn = 1
		}
		for r := 0; r < 3; r++ {
			u[r][c] = cr[r] / nn
		}
	}
	return u, s, v
}
