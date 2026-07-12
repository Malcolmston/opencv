package face

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// This file collects the small, general dense-matrix kernels used by the newer
// capabilities in this package (landmark regression, the correlation filter and
// the biologically-inspired features). They complement the symmetric Jacobi
// eigensolver in linalg.go with a general linear-system solver and a matrix
// inverse, and provide a handful of numeric conveniences. Everything is
// standard-library only and operates on dense row-major [][]float64 matrices.

// replicateAt reads the single-channel Mat g at (y,x), clamping out-of-range
// coordinates to the nearest edge (the BORDER_REPLICATE convention). It is the
// package-local analogue of the root Mat's unexported edge-clamped accessor and
// assumes g has one channel.
func replicateAt(g *cv.Mat, y, x int) uint8 {
	if y < 0 {
		y = 0
	} else if y >= g.Rows {
		y = g.Rows - 1
	}
	if x < 0 {
		x = 0
	} else if x >= g.Cols {
		x = g.Cols - 1
	}
	return g.Data[y*g.Cols+x]
}

// clampByte rounds v to the nearest integer and saturates it into the 8-bit
// range [0,255], matching the root package's pixel arithmetic.
func clampByte(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v + 0.5)
}

// solveLinearSystem solves the square system a·x = b for x using Gaussian
// elimination with partial pivoting. a (n×n) and b (length n) are copied, so the
// inputs are left unmodified. It reports ok=false when the system is singular
// (a zero pivot), in which case x is nil.
func solveLinearSystem(a [][]float64, b []float64) (x []float64, ok bool) {
	n := len(a)
	if n == 0 || len(b) != n {
		return nil, false
	}
	// Augmented working copy.
	m := make([][]float64, n)
	for i := 0; i < n; i++ {
		m[i] = make([]float64, n+1)
		copy(m[i], a[i])
		m[i][n] = b[i]
	}
	for col := 0; col < n; col++ {
		// Partial pivot: pick the row with the largest magnitude in this column.
		pivot := col
		best := math.Abs(m[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(m[r][col]); v > best {
				best = v
				pivot = r
			}
		}
		if best < 1e-15 {
			return nil, false
		}
		m[col], m[pivot] = m[pivot], m[col]
		// Eliminate below.
		pv := m[col][col]
		for r := col + 1; r < n; r++ {
			f := m[r][col] / pv
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

// ridgeSolve fits a multi-output linear model Y ≈ X·W by ridge regression,
// returning the (f×o) coefficient matrix W that minimises ‖X·W − Y‖² + λ‖W‖².
// X is (n×f) samples-by-features and Y is (n×o) samples-by-outputs. The normal
// equations (XᵀX + λI)·W = XᵀY are solved column by column. λ must be positive
// for a well-posed problem; it also guarantees the system is non-singular.
func ridgeSolve(X, Y [][]float64, lambda float64) [][]float64 {
	n := len(X)
	if n == 0 {
		return nil
	}
	f := len(X[0])
	o := len(Y[0])

	// A = XᵀX + λI  (f×f, symmetric positive definite).
	A := make([][]float64, f)
	for i := 0; i < f; i++ {
		A[i] = make([]float64, f)
	}
	for _, row := range X {
		for i := 0; i < f; i++ {
			xi := row[i]
			if xi == 0 {
				continue
			}
			ai := A[i]
			for j := 0; j < f; j++ {
				ai[j] += xi * row[j]
			}
		}
	}
	for i := 0; i < f; i++ {
		A[i][i] += lambda
	}

	// B = XᵀY  (f×o).
	B := make([][]float64, f)
	for i := 0; i < f; i++ {
		B[i] = make([]float64, o)
	}
	for s := 0; s < n; s++ {
		xr := X[s]
		yr := Y[s]
		for i := 0; i < f; i++ {
			xi := xr[i]
			if xi == 0 {
				continue
			}
			bi := B[i]
			for k := 0; k < o; k++ {
				bi[k] += xi * yr[k]
			}
		}
	}

	// Solve A·W = B for each output column.
	W := make([][]float64, f)
	for i := 0; i < f; i++ {
		W[i] = make([]float64, o)
	}
	for k := 0; k < o; k++ {
		bk := make([]float64, f)
		for i := 0; i < f; i++ {
			bk[i] = B[i][k]
		}
		xk, ok := solveLinearSystem(A, bk)
		if !ok {
			continue
		}
		for i := 0; i < f; i++ {
			W[i][k] = xk[i]
		}
	}
	return W
}
