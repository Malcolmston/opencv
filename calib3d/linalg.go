package calib3d

import "math"

// This file contains the small, self-contained linear-algebra kernels the
// package relies on. Everything here is implemented from scratch against the
// standard library so that calib3d depends only on the root cv package and Go's
// stdlib. None of these helpers are exported; they exist purely to support the
// public camera-geometry API.

// mat3 is a 3×3 matrix in row-major order. It mirrors the [3][3]float64 shape
// used across the public API but is handled through the helpers below.

// mul3 returns the matrix product a·b of two 3×3 matrices.
func mul3(a, b [3][3]float64) [3][3]float64 {
	var r [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			var s float64
			for k := 0; k < 3; k++ {
				s += a[i][k] * b[k][j]
			}
			r[i][j] = s
		}
	}
	return r
}

// transpose3 returns the transpose of a 3×3 matrix.
func transpose3(a [3][3]float64) [3][3]float64 {
	var r [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			r[j][i] = a[i][j]
		}
	}
	return r
}

// matVec3 returns the matrix–vector product a·v.
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

// inv3 returns the inverse of a 3×3 matrix and reports whether it is
// invertible (a determinant of near-zero magnitude yields ok == false).
func inv3(m [3][3]float64) ([3][3]float64, bool) {
	det := det3(m)
	if math.Abs(det) < 1e-18 {
		return [3][3]float64{}, false
	}
	id := 1 / det
	var r [3][3]float64
	r[0][0] = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * id
	r[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * id
	r[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * id
	r[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * id
	r[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * id
	r[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * id
	r[2][0] = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * id
	r[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * id
	r[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * id
	return r, true
}

// cross3 returns the vector cross product a×b.
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

// jacobiEigenSym computes the eigenvalues and eigenvectors of a real symmetric
// matrix using the cyclic Jacobi rotation method. The input a is treated as
// symmetric and is not modified. It returns the eigenvalues and a matrix whose
// columns are the corresponding (orthonormal) eigenvectors: vecs[i][j] is the
// i-th component of the eigenvector for vals[j]. The routine converges
// quadratically and is numerically robust for the small matrices (3×3 up to
// 9×9) used throughout this package.
func jacobiEigenSym(a [][]float64) (vals []float64, vecs [][]float64) {
	n := len(a)
	m := make([][]float64, n)
	for i := 0; i < n; i++ {
		m[i] = make([]float64, n)
		copy(m[i], a[i])
	}
	v := make([][]float64, n)
	for i := 0; i < n; i++ {
		v[i] = make([]float64, n)
		v[i][i] = 1
	}
	for sweep := 0; sweep < 100; sweep++ {
		off := 0.0
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				off += m[p][q] * m[p][q]
			}
		}
		if off < 1e-30 {
			break
		}
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
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
				// Apply the Givens rotation to rows/cols p and q of m.
				for k := 0; k < n; k++ {
					mkp := m[k][p]
					mkq := m[k][q]
					m[k][p] = c*mkp - s*mkq
					m[k][q] = s*mkp + c*mkq
				}
				for k := 0; k < n; k++ {
					mpk := m[p][k]
					mqk := m[q][k]
					m[p][k] = c*mpk - s*mqk
					m[q][k] = s*mpk + c*mqk
				}
				// Accumulate the rotation into the eigenvector matrix.
				for k := 0; k < n; k++ {
					vkp := v[k][p]
					vkq := v[k][q]
					v[k][p] = c*vkp - s*vkq
					v[k][q] = s*vkp + c*vkq
				}
			}
		}
	}
	vals = make([]float64, n)
	for i := 0; i < n; i++ {
		vals[i] = m[i][i]
	}
	return vals, v
}

// smallestEigvec returns the eigenvector of the symmetric matrix a associated
// with its smallest eigenvalue, as a freshly allocated slice. It is the null
// space of a rank-deficient normal-equation matrix A·Aᵀ and underpins the
// homography, fundamental-matrix and triangulation solvers.
func smallestEigvec(a [][]float64) []float64 {
	vals, vecs := jacobiEigenSym(a)
	best := 0
	for i := 1; i < len(vals); i++ {
		if vals[i] < vals[best] {
			best = i
		}
	}
	n := len(a)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = vecs[i][best]
	}
	return out
}

// svd3 computes the singular value decomposition A = U·diag(S)·Vᵀ of a 3×3
// matrix. The singular values S are returned in non-increasing order with the
// columns of U and V ordered to match. It is used to enforce the rank-2
// constraint on fundamental matrices and to orthonormalise rotation matrices.
func svd3(a [3][3]float64) (u [3][3]float64, s [3]float64, v [3][3]float64) {
	// Eigen-decompose AᵀA (symmetric, positive semi-definite). Its eigenvectors
	// are the right singular vectors and its eigenvalues the squared singular
	// values.
	at := transpose3(a)
	ata := mul3(at, a)
	dyn := [][]float64{
		{ata[0][0], ata[0][1], ata[0][2]},
		{ata[1][0], ata[1][1], ata[1][2]},
		{ata[2][0], ata[2][1], ata[2][2]},
	}
	vals, vecs := jacobiEigenSym(dyn)
	// Sort eigenvalues (and their vectors) in descending order.
	idx := []int{0, 1, 2}
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
	// Recover U columns as u_i = A·v_i / s_i. Where s_i is tiny, fill in an
	// orthonormal completion so U stays a proper rotation/reflection.
	for c := 0; c < 3; c++ {
		vc := [3]float64{v[0][c], v[1][c], v[2][c]}
		av := matVec3(a, vc)
		// Use a threshold relative to the largest singular value so that a
		// numerically-tiny singular value (e.g. the structural zero of an
		// essential matrix, which survives as ~1e-9 after normalization) is
		// treated as degenerate and its U column is completed rather than being
		// recovered as A·v/s ≈ 0/0.
		if s[c] > 1e-12 && s[c] > 1e-6*s[0] {
			for r := 0; r < 3; r++ {
				u[r][c] = av[r] / s[c]
			}
		} else {
			u[0][c] = math.NaN() // marker; filled below
		}
	}
	// Complete any degenerate U columns via cross products of the valid ones.
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
