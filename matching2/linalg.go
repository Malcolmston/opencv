package matching2

import "math"

// Mat3Identity returns the 3×3 identity matrix.
func Mat3Identity() [3][3]float64 {
	return [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
}

// Mat3Mul returns the matrix product a·b of two 3×3 matrices.
func Mat3Mul(a, b [3][3]float64) [3][3]float64 {
	var m [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			m[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j]
		}
	}
	return m
}

// Mat3Transpose returns the transpose aᵀ of a 3×3 matrix.
func Mat3Transpose(a [3][3]float64) [3][3]float64 {
	return [3][3]float64{
		{a[0][0], a[1][0], a[2][0]},
		{a[0][1], a[1][1], a[2][1]},
		{a[0][2], a[1][2], a[2][2]},
	}
}

// Mat3VecMul returns the matrix–vector product a·v, where v is a 3-vector.
func Mat3VecMul(a [3][3]float64, v [3]float64) [3]float64 {
	return [3]float64{
		a[0][0]*v[0] + a[0][1]*v[1] + a[0][2]*v[2],
		a[1][0]*v[0] + a[1][1]*v[1] + a[1][2]*v[2],
		a[2][0]*v[0] + a[2][1]*v[1] + a[2][2]*v[2],
	}
}

// Mat3Scale returns the 3×3 matrix a with every entry multiplied by s.
func Mat3Scale(a [3][3]float64, s float64) [3][3]float64 {
	var m [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			m[i][j] = a[i][j] * s
		}
	}
	return m
}

// Mat3Det returns the determinant of a 3×3 matrix.
func Mat3Det(m [3][3]float64) float64 {
	return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) -
		m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) +
		m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
}

// Mat3Inverse returns the inverse of a 3×3 matrix and reports whether it is
// invertible. When the determinant is numerically zero it returns the zero
// matrix and false.
func Mat3Inverse(m [3][3]float64) ([3][3]float64, bool) {
	det := Mat3Det(m)
	if math.Abs(det) < 1e-300 {
		return [3][3]float64{}, false
	}
	inv := 1.0 / det
	var r [3][3]float64
	r[0][0] = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * inv
	r[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * inv
	r[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * inv
	r[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * inv
	r[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * inv
	r[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * inv
	r[2][0] = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * inv
	r[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * inv
	r[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * inv
	return r, true
}

// matching2signf returns -1, 0 or +1 following the sign of x.
func matching2signf(x float64) float64 {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	default:
		return 0
	}
}

// matching2symEig computes the eigenvalues and eigenvectors of a symmetric n×n
// matrix using the cyclic Jacobi method. It returns the eigenvalues in
// ascending order and vecs, where vecs[k] is the unit eigenvector belonging to
// vals[k]. The input is not modified.
func matching2symEig(a [][]float64) (vals []float64, vecs [][]float64) {
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
		if off < 1e-32 {
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
					t = matching2signf(theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				}
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
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
				for k := 0; k < n; k++ {
					vkp := v[k][p]
					vkq := v[k][q]
					v[k][p] = c*vkp - s*vkq
					v[k][q] = s*vkp + c*vkq
				}
			}
		}
	}
	// Collect eigenvalues (diagonal) and eigenvectors (columns of v), then sort
	// ascending by eigenvalue for a stable, deterministic ordering.
	idx := make([]int, n)
	diag := make([]float64, n)
	for i := 0; i < n; i++ {
		idx[i] = i
		diag[i] = m[i][i]
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if diag[idx[j]] < diag[idx[i]] {
				idx[i], idx[j] = idx[j], idx[i]
			}
		}
	}
	vals = make([]float64, n)
	vecs = make([][]float64, n)
	for k := 0; k < n; k++ {
		col := idx[k]
		vals[k] = diag[col]
		ev := make([]float64, n)
		for r := 0; r < n; r++ {
			ev[r] = v[r][col]
		}
		vecs[k] = ev
	}
	return vals, vecs
}

// matching2nullVector returns the unit right-singular vector of the m×n matrix A
// belonging to its smallest singular value: the least-squares solution of
// A·x = 0 subject to ‖x‖ = 1. It is obtained as the smallest eigenvector of the
// normal-equation matrix AᵀA, which underpins every direct-linear-transform
// solver in this package.
func matching2nullVector(a [][]float64) []float64 {
	rows := len(a)
	if rows == 0 {
		return nil
	}
	n := len(a[0])
	ata := make([][]float64, n)
	for i := 0; i < n; i++ {
		ata[i] = make([]float64, n)
	}
	for r := 0; r < rows; r++ {
		row := a[r]
		for i := 0; i < n; i++ {
			ri := row[i]
			if ri == 0 {
				continue
			}
			for j := i; j < n; j++ {
				ata[i][j] += ri * row[j]
			}
		}
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			ata[j][i] = ata[i][j]
		}
	}
	_, vecs := matching2symEig(ata)
	return vecs[0]
}

// matching2svd3 computes the singular value decomposition A = U·diag(S)·Vᵀ of a
// 3×3 matrix. Singular values S are returned in non-increasing order with the
// columns of U and V ordered to match. Degenerate U columns (near-zero singular
// value) are completed to keep U orthonormal.
func matching2svd3(a [3][3]float64) (u [3][3]float64, s [3]float64, v [3][3]float64) {
	at := Mat3Transpose(a)
	ata := Mat3Mul(at, a)
	dyn := [][]float64{
		{ata[0][0], ata[0][1], ata[0][2]},
		{ata[1][0], ata[1][1], ata[1][2]},
		{ata[2][0], ata[2][1], ata[2][2]},
	}
	vals, vecs := matching2symEig(dyn) // ascending
	// Reverse to descending order.
	order := [3]int{2, 1, 0}
	for c := 0; c < 3; c++ {
		src := order[c]
		ev := vals[src]
		if ev < 0 {
			ev = 0
		}
		s[c] = math.Sqrt(ev)
		for r := 0; r < 3; r++ {
			v[r][c] = vecs[src][r]
		}
	}
	for c := 0; c < 3; c++ {
		vc := [3]float64{v[0][c], v[1][c], v[2][c]}
		av := Mat3VecMul(a, vc)
		if s[c] > 1e-12 && s[c] > 1e-9*s[0] {
			for r := 0; r < 3; r++ {
				u[r][c] = av[r] / s[c]
			}
		} else {
			u[0][c] = math.NaN()
		}
	}
	for c := 0; c < 3; c++ {
		if !math.IsNaN(u[0][c]) {
			continue
		}
		a0 := [3]float64{u[0][(c+1)%3], u[1][(c+1)%3], u[2][(c+1)%3]}
		a1 := [3]float64{u[0][(c+2)%3], u[1][(c+2)%3], u[2][(c+2)%3]}
		cr := matching2cross(a1, a0)
		nn := matching2norm3(cr)
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

// matching2cross returns the 3-vector cross product a×b.
func matching2cross(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

// matching2norm3 returns the Euclidean length of a 3-vector.
func matching2norm3(v [3]float64) float64 {
	return math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
}
