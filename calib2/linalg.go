package calib2

import (
	"math"
)

// Matrix is a dense, row-major matrix of float64 values, used throughout the
// package for the linear-algebra steps of calibration and triangulation. The
// zero value is not usable; construct instances with [NewMatrix],
// [ZerosMatrix] or [IdentityMatrix].
type Matrix struct {
	// Rows is the number of rows.
	Rows int
	// Cols is the number of columns.
	Cols int
	// Data holds the elements in row-major order, length Rows*Cols; the value
	// at row i, column j is Data[i*Cols+j].
	Data []float64
}

// NewMatrix allocates a zero-filled Matrix with the given dimensions. It panics
// if either dimension is not positive.
func NewMatrix(rows, cols int) *Matrix {
	if rows <= 0 || cols <= 0 {
		panic("calib2: NewMatrix requires positive dimensions")
	}
	return &Matrix{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

// ZerosMatrix returns a rows×cols matrix filled with zeros. It is an alias for
// [NewMatrix] provided for readability at call sites.
func ZerosMatrix(rows, cols int) *Matrix {
	return NewMatrix(rows, cols)
}

// IdentityMatrix returns an n×n identity matrix. It panics if n is not
// positive.
func IdentityMatrix(n int) *Matrix {
	m := NewMatrix(n, n)
	for i := 0; i < n; i++ {
		m.Data[i*n+i] = 1
	}
	return m
}

// At returns the element at row i, column j. It panics if the indices are out
// of range.
func (m *Matrix) At(i, j int) float64 {
	if i < 0 || i >= m.Rows || j < 0 || j >= m.Cols {
		panic("calib2: Matrix.At index out of range")
	}
	return m.Data[i*m.Cols+j]
}

// Set stores v at row i, column j. It panics if the indices are out of range.
func (m *Matrix) Set(i, j int, v float64) {
	if i < 0 || i >= m.Rows || j < 0 || j >= m.Cols {
		panic("calib2: Matrix.Set index out of range")
	}
	m.Data[i*m.Cols+j] = v
}

// Clone returns an independent deep copy of the matrix.
func (m *Matrix) Clone() *Matrix {
	out := &Matrix{Rows: m.Rows, Cols: m.Cols, Data: make([]float64, len(m.Data))}
	copy(out.Data, m.Data)
	return out
}

// Transpose returns a new matrix that is the transpose of the receiver.
func (m *Matrix) Transpose() *Matrix {
	out := NewMatrix(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			out.Data[j*m.Rows+i] = m.Data[i*m.Cols+j]
		}
	}
	return out
}

// Mul returns the matrix product m·other. It panics if the inner dimensions do
// not agree.
func (m *Matrix) Mul(other *Matrix) *Matrix {
	if m.Cols != other.Rows {
		panic("calib2: Matrix.Mul dimension mismatch")
	}
	out := NewMatrix(m.Rows, other.Cols)
	for i := 0; i < m.Rows; i++ {
		for k := 0; k < m.Cols; k++ {
			a := m.Data[i*m.Cols+k]
			if a == 0 {
				continue
			}
			for j := 0; j < other.Cols; j++ {
				out.Data[i*out.Cols+j] += a * other.Data[k*other.Cols+j]
			}
		}
	}
	return out
}

// MulVec returns the matrix–vector product m·v as a new slice. It panics if the
// length of v does not equal the number of columns.
func (m *Matrix) MulVec(v []float64) []float64 {
	if len(v) != m.Cols {
		panic("calib2: Matrix.MulVec dimension mismatch")
	}
	out := make([]float64, m.Rows)
	for i := 0; i < m.Rows; i++ {
		var s float64
		for j := 0; j < m.Cols; j++ {
			s += m.Data[i*m.Cols+j] * v[j]
		}
		out[i] = s
	}
	return out
}

// FrobeniusNorm returns the Frobenius norm (square root of the sum of squared
// elements) of the matrix.
func (m *Matrix) FrobeniusNorm() float64 {
	var s float64
	for _, v := range m.Data {
		s += v * v
	}
	return math.Sqrt(s)
}

// gram returns the symmetric Gram matrix mᵀ·m as an n×n [][]float64 where
// n = m.Cols. It is the normal-equations matrix used by the null-space solvers.
func (m *Matrix) gram() [][]float64 {
	n := m.Cols
	g := make([][]float64, n)
	for i := range g {
		g[i] = make([]float64, n)
	}
	for r := 0; r < m.Rows; r++ {
		row := m.Data[r*m.Cols : r*m.Cols+m.Cols]
		for i := 0; i < n; i++ {
			ai := row[i]
			if ai == 0 {
				continue
			}
			for j := i; j < n; j++ {
				g[i][j] += ai * row[j]
			}
		}
	}
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			g[j][i] = g[i][j]
		}
	}
	return g
}

// jacobiEigen computes the eigenvalues and eigenvectors of a real symmetric
// n×n matrix by the cyclic Jacobi rotation method. It returns the eigenvalues
// and a matrix whose columns are the corresponding orthonormal eigenvectors:
// vecs[i][k] is component i of the k-th eigenvector. The routine is
// deterministic and converges quadratically for symmetric input.
func jacobiEigen(a [][]float64) (vals []float64, vecs [][]float64) {
	n := len(a)
	// Work on a copy so the input is not mutated.
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, n)
		copy(m[i], a[i])
	}
	v := make([][]float64, n)
	for i := range v {
		v[i] = make([]float64, n)
		v[i][i] = 1
	}
	for sweep := 0; sweep < 100; sweep++ {
		// Sum of off-diagonal magnitudes; stop when negligible.
		var off float64
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				off += math.Abs(m[p][q])
			}
		}
		if off < 1e-300 || off < 1e-18 {
			break
		}
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				if math.Abs(m[p][q]) < 1e-300 {
					continue
				}
				app := m[p][p]
				aqq := m[q][q]
				apq := m[p][q]
				phi := 0.5 * math.Atan2(2*apq, aqq-app)
				c := math.Cos(phi)
				s := math.Sin(phi)
				// Apply the rotation J to M on both sides: M = Jᵀ M J.
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
				// Accumulate the eigenvectors.
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

// smallestEigenvector returns the unit eigenvector of the symmetric matrix a
// associated with its smallest eigenvalue. It is the null-space estimate used
// by the DLT homography solver, Zhang's intrinsic solve and triangulation.
func smallestEigenvector(a [][]float64) []float64 {
	vals, vecs := jacobiEigen(a)
	n := len(vals)
	best := 0
	for i := 1; i < n; i++ {
		if vals[i] < vals[best] {
			best = i
		}
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = vecs[i][best]
	}
	return out
}

// svd3 computes a singular value decomposition of a 3×3 matrix, returning
// orthonormal u and v and descending singular values s such that
// a = u·diag(s)·vᵀ. It is built on the symmetric eigensolver applied to aᵀa and
// is used by the essential-matrix decomposition.
func svd3(a [3][3]float64) (u [3][3]float64, s [3]float64, v [3][3]float64) {
	ata := Mat3Mul(Mat3Transpose(a), a)
	g := [][]float64{
		{ata[0][0], ata[0][1], ata[0][2]},
		{ata[1][0], ata[1][1], ata[1][2]},
		{ata[2][0], ata[2][1], ata[2][2]},
	}
	vals, vecs := jacobiEigen(g)
	// Sort eigenpairs by descending eigenvalue.
	idx := []int{0, 1, 2}
	for i := 0; i < 3; i++ {
		for j := i + 1; j < 3; j++ {
			if vals[idx[j]] > vals[idx[i]] {
				idx[i], idx[j] = idx[j], idx[i]
			}
		}
	}
	for c := 0; c < 3; c++ {
		k := idx[c]
		val := vals[k]
		if val < 0 {
			val = 0
		}
		s[c] = math.Sqrt(val)
		for r := 0; r < 3; r++ {
			v[r][c] = vecs[r][k]
		}
	}
	// u_c = a·v_c / s_c for non-zero singular values. The threshold is relative
	// to the largest singular value so a numerically tiny (rank-deficient)
	// singular value — e.g. the zero of an essential matrix — is treated as
	// zero and its u column is filled orthonormally below rather than divided by
	// near-zero noise.
	eps := 1e-6 * s[0]
	if eps < 1e-300 {
		eps = 1e-300
	}
	var known [3]bool
	for c := 0; c < 3; c++ {
		vc := [3]float64{v[0][c], v[1][c], v[2][c]}
		av := Mat3VecMul(a, vc)
		if s[c] > eps {
			for r := 0; r < 3; r++ {
				u[r][c] = av[r] / s[c]
			}
			known[c] = true
		}
	}
	// Fill any missing u column as the cross product of the known ones to keep
	// u orthonormal (handles zero singular values, e.g. essential matrices).
	fillMissing := func() {
		for c := 0; c < 3; c++ {
			if known[c] {
				continue
			}
			a1 := (c + 1) % 3
			a2 := (c + 2) % 3
			c1 := [3]float64{u[0][a1], u[1][a1], u[2][a1]}
			c2 := [3]float64{u[0][a2], u[1][a2], u[2][a2]}
			cr := Vec3Normalize(Vec3Cross(c1, c2))
			for r := 0; r < 3; r++ {
				u[r][c] = cr[r]
			}
			known[c] = true
		}
	}
	fillMissing()
	return u, s, v
}

// solveLinear solves the square linear system A·x = b by Gaussian elimination
// with partial pivoting. It reports ok=false when the matrix is singular to
// working precision. A and b are not modified.
func solveLinear(a [][]float64, b []float64) (x []float64, ok bool) {
	n := len(a)
	// Build an augmented working copy.
	m := make([][]float64, n)
	for i := 0; i < n; i++ {
		m[i] = make([]float64, n+1)
		copy(m[i], a[i])
		m[i][n] = b[i]
	}
	for col := 0; col < n; col++ {
		// Partial pivot.
		piv := col
		max := math.Abs(m[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(m[r][col]); v > max {
				max = v
				piv = r
			}
		}
		if max < 1e-15 {
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
	for i := n - 1; i >= 0; i-- {
		s := m[i][n]
		for j := i + 1; j < n; j++ {
			s -= m[i][j] * x[j]
		}
		x[i] = s / m[i][i]
	}
	return x, true
}
