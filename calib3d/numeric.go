package calib3d

import "math"

// This file adds the general-purpose (arbitrary-size) linear-algebra kernels the
// higher-level calibration and multi-view routines need on top of the fixed 3×3
// helpers in linalg.go. Everything here is standard-library-only and unexported.

// solveLinear solves the square linear system A·x = b by Gaussian elimination
// with partial pivoting. A is an n×n matrix given row-major; it is copied and
// left unmodified. ok is false when the system is singular to working precision.
func solveLinear(A [][]float64, b []float64) (x []float64, ok bool) {
	n := len(A)
	if n == 0 || len(b) != n {
		return nil, false
	}
	m := make([][]float64, n)
	for i := 0; i < n; i++ {
		m[i] = make([]float64, n+1)
		copy(m[i], A[i])
		m[i][n] = b[i]
	}
	for col := 0; col < n; col++ {
		// Partial pivot: pick the row with the largest magnitude in this column.
		piv := col
		best := math.Abs(m[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(m[r][col]); v > best {
				best = v
				piv = r
			}
		}
		if best < 1e-15 {
			return nil, false
		}
		m[col], m[piv] = m[piv], m[col]
		// Eliminate below.
		for r := col + 1; r < n; r++ {
			f := m[r][col] / m[col][col]
			if f == 0 {
				continue
			}
			for c := col; c <= n; c++ {
				m[r][c] -= f * m[col][c]
			}
		}
	}
	// Back-substitution.
	x = make([]float64, n)
	for r := n - 1; r >= 0; r-- {
		s := m[r][n]
		for c := r + 1; c < n; c++ {
			s -= m[r][c] * x[c]
		}
		x[r] = s / m[r][r]
	}
	return x, true
}

// leastSquares returns the minimum-norm least-squares solution of the
// (possibly overdetermined) system A·x ≈ b via the normal equations
// AᵀA·x = Aᵀb. A has m rows of length n with m ≥ n. ok is false when AᵀA is
// singular.
func leastSquares(A [][]float64, b []float64) (x []float64, ok bool) {
	m := len(A)
	if m == 0 {
		return nil, false
	}
	n := len(A[0])
	ata := make([][]float64, n)
	atb := make([]float64, n)
	for i := 0; i < n; i++ {
		ata[i] = make([]float64, n)
	}
	for r := 0; r < m; r++ {
		for i := 0; i < n; i++ {
			atb[i] += A[r][i] * b[r]
			for j := 0; j < n; j++ {
				ata[i][j] += A[r][i] * A[r][j]
			}
		}
	}
	return solveLinear(ata, atb)
}

// nullspaceVec returns the unit vector x minimising ‖A·x‖ subject to ‖x‖ = 1,
// i.e. the right singular vector of A for its smallest singular value. It is
// obtained as the smallest-eigenvalue eigenvector of AᵀA. rows is the list of
// design-matrix rows (each of length n).
func nullspaceVec(rows [][]float64, n int) []float64 {
	ata := make([][]float64, n)
	for i := 0; i < n; i++ {
		ata[i] = make([]float64, n)
	}
	for _, r := range rows {
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				ata[i][j] += r[i] * r[j]
			}
		}
	}
	return smallestEigvec(ata)
}

// orthonormalize returns the closest proper rotation matrix (in the Frobenius
// sense) to m, computed as U·Vᵀ from the SVD of m with the sign of the last
// column of U adjusted to force determinant +1.
func orthonormalize(m [3][3]float64) [3][3]float64 {
	u, _, v := svd3(m)
	r := mul3(u, transpose3(v))
	if det3(r) < 0 {
		u[0][2] = -u[0][2]
		u[1][2] = -u[1][2]
		u[2][2] = -u[2][2]
		r = mul3(u, transpose3(v))
	}
	return r
}

// dot3 returns the dot product of two 3-vectors.
func dot3(a, b [3]float64) float64 { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }

// sub3 returns a − b.
func sub3(a, b [3]float64) [3]float64 { return [3]float64{a[0] - b[0], a[1] - b[1], a[2] - b[2]} }

// add3 returns a + b.
func add3(a, b [3]float64) [3]float64 { return [3]float64{a[0] + b[0], a[1] + b[1], a[2] + b[2]} }

// normalize3 returns the unit vector along v and its original length. A
// near-zero vector is returned unchanged with length 0.
func normalize3(v [3]float64) (unit [3]float64, length float64) {
	length = norm3(v)
	if length < 1e-18 {
		return v, 0
	}
	return [3]float64{v[0] / length, v[1] / length, v[2] / length}, length
}

// colsToMat builds a 3×3 matrix from three column vectors.
func colsToMat(c0, c1, c2 [3]float64) [3][3]float64 {
	return [3][3]float64{
		{c0[0], c1[0], c2[0]},
		{c0[1], c1[1], c2[1]},
		{c0[2], c1[2], c2[2]},
	}
}

// col3 extracts column j of a 3×3 matrix.
func col3(m [3][3]float64, j int) [3]float64 {
	return [3]float64{m[0][j], m[1][j], m[2][j]}
}
