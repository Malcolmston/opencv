package cv

import (
	"fmt"
	"math"
	"sort"
)

// requireSquareFloat panics unless m is a non-empty square FloatMat.
func requireSquareFloat(m *FloatMat, name string) {
	if m == nil || m.Rows == 0 || m.Rows != m.Cols {
		r, c := 0, 0
		if m != nil {
			r, c = m.Rows, m.Cols
		}
		panic(fmt.Sprintf("cv: %s requires a square matrix, got %dx%d", name, r, c))
	}
}

// fclone returns a deep copy of a FloatMat with its own backing storage.
func fclone(m *FloatMat) *FloatMat {
	out := NewFloatMat(m.Rows, m.Cols)
	copy(out.Data, m.Data)
	return out
}

// Trace returns the sum of the diagonal elements of a matrix. For a
// non-square matrix the shorter of the two dimensions bounds the diagonal,
// matching OpenCV's cv::trace (which reports the first component of the
// scalar).
func Trace(m *FloatMat) float64 {
	n := m.Rows
	if m.Cols < n {
		n = m.Cols
	}
	var s float64
	for i := 0; i < n; i++ {
		s += m.Data[i*m.Cols+i]
	}
	return s
}

// Determinant returns the determinant of a square matrix, computed by Gaussian
// elimination with partial pivoting. It panics if the matrix is not square.
func Determinant(m *FloatMat) float64 {
	requireSquareFloat(m, "Determinant")
	n := m.Rows
	a := make([]float64, len(m.Data))
	copy(a, m.Data)
	det := 1.0
	for col := 0; col < n; col++ {
		piv := col
		best := math.Abs(a[col*n+col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(a[r*n+col]); v > best {
				best = v
				piv = r
			}
		}
		if best == 0 {
			return 0
		}
		if piv != col {
			for c := 0; c < n; c++ {
				a[col*n+c], a[piv*n+c] = a[piv*n+c], a[col*n+c]
			}
			det = -det
		}
		p := a[col*n+col]
		det *= p
		for r := col + 1; r < n; r++ {
			f := a[r*n+col] / p
			if f == 0 {
				continue
			}
			for c := col; c < n; c++ {
				a[r*n+c] -= f * a[col*n+c]
			}
		}
	}
	return det
}

// Invert computes the inverse of a square matrix using Gauss-Jordan
// elimination with partial pivoting. The boolean result reports whether the
// matrix was non-singular; when false the returned matrix is all zeros. It
// panics if the matrix is not square.
func Invert(m *FloatMat) (*FloatMat, bool) {
	requireSquareFloat(m, "Invert")
	n := m.Rows
	// Augmented [A | I].
	a := make([]float64, n*n)
	copy(a, m.Data)
	inv := NewFloatMat(n, n)
	for i := 0; i < n; i++ {
		inv.Data[i*n+i] = 1
	}
	for col := 0; col < n; col++ {
		piv := col
		best := math.Abs(a[col*n+col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(a[r*n+col]); v > best {
				best = v
				piv = r
			}
		}
		if best < 1e-15 {
			return NewFloatMat(n, n), false
		}
		if piv != col {
			for c := 0; c < n; c++ {
				a[col*n+c], a[piv*n+c] = a[piv*n+c], a[col*n+c]
				inv.Data[col*n+c], inv.Data[piv*n+c] = inv.Data[piv*n+c], inv.Data[col*n+c]
			}
		}
		p := a[col*n+col]
		for c := 0; c < n; c++ {
			a[col*n+c] /= p
			inv.Data[col*n+c] /= p
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := a[r*n+col]
			if f == 0 {
				continue
			}
			for c := 0; c < n; c++ {
				a[r*n+c] -= f * a[col*n+c]
				inv.Data[r*n+c] -= f * inv.Data[col*n+c]
			}
		}
	}
	return inv, true
}

// Solve solves the linear system a*x = b for x, where a is an N×N matrix and b
// is an N×M right-hand side (M may be 1). It uses Gauss-Jordan elimination with
// partial pivoting. The boolean reports whether a was non-singular. It panics
// if the shapes are incompatible.
func Solve(a, b *FloatMat) (*FloatMat, bool) {
	requireSquareFloat(a, "Solve")
	n := a.Rows
	if b.Rows != n {
		panic(fmt.Sprintf("cv: Solve rhs has %d rows, want %d", b.Rows, n))
	}
	m := b.Cols
	work := make([]float64, n*n)
	copy(work, a.Data)
	x := fclone(b)
	for col := 0; col < n; col++ {
		piv := col
		best := math.Abs(work[col*n+col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(work[r*n+col]); v > best {
				best = v
				piv = r
			}
		}
		if best < 1e-15 {
			return NewFloatMat(n, m), false
		}
		if piv != col {
			for c := 0; c < n; c++ {
				work[col*n+c], work[piv*n+c] = work[piv*n+c], work[col*n+c]
			}
			for c := 0; c < m; c++ {
				x.Data[col*m+c], x.Data[piv*m+c] = x.Data[piv*m+c], x.Data[col*m+c]
			}
		}
		p := work[col*n+col]
		for c := 0; c < n; c++ {
			work[col*n+c] /= p
		}
		for c := 0; c < m; c++ {
			x.Data[col*m+c] /= p
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := work[r*n+col]
			if f == 0 {
				continue
			}
			for c := 0; c < n; c++ {
				work[r*n+c] -= f * work[col*n+c]
			}
			for c := 0; c < m; c++ {
				x.Data[r*m+c] -= f * x.Data[col*m+c]
			}
		}
	}
	return x, true
}

// SetIdentity sets a matrix to a scaled identity: the main diagonal is filled
// with value and every other element is set to zero. It works on rectangular
// matrices, matching OpenCV's cv::setIdentity.
func SetIdentity(m *FloatMat, value float64) {
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			if x == y {
				m.Data[y*m.Cols+x] = value
			} else {
				m.Data[y*m.Cols+x] = 0
			}
		}
	}
}

// CompleteSymm makes a square matrix symmetric by copying one triangle onto the
// other. When lowerToUpper is true the lower triangle is mirrored into the
// upper triangle; otherwise the upper triangle is mirrored into the lower one.
// It panics if the matrix is not square.
func CompleteSymm(m *FloatMat, lowerToUpper bool) {
	requireSquareFloat(m, "CompleteSymm")
	n := m.Rows
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if lowerToUpper {
				m.Data[i*n+j] = m.Data[j*n+i]
			} else {
				m.Data[j*n+i] = m.Data[i*n+j]
			}
		}
	}
}

// matMul multiplies two row-major matrices (a: p×q, b: q×r) and returns p×r.
func matMul(a []float64, ar, ac int, b []float64, br, bc int) []float64 {
	if ac != br {
		panic(fmt.Sprintf("cv: matMul inner dimension mismatch %d vs %d", ac, br))
	}
	out := make([]float64, ar*bc)
	for i := 0; i < ar; i++ {
		for k := 0; k < ac; k++ {
			aik := a[i*ac+k]
			if aik == 0 {
				continue
			}
			for j := 0; j < bc; j++ {
				out[i*bc+j] += aik * b[k*bc+j]
			}
		}
	}
	return out
}

// Gemm computes the generalised matrix product alpha*op(a)*op(b) + beta*c,
// where op(x) is x transposed when the corresponding flag is set. If c is nil
// the beta term is omitted. This mirrors OpenCV's cv::gemm. It panics on a
// dimension mismatch.
func Gemm(a, b *FloatMat, alpha float64, c *FloatMat, beta float64, transA, transB bool) *FloatMat {
	ar, ac := a.Rows, a.Cols
	ad := a.Data
	if transA {
		ad = transposeFlat(a.Data, a.Rows, a.Cols)
		ar, ac = a.Cols, a.Rows
	}
	br, bc := b.Rows, b.Cols
	bd := b.Data
	if transB {
		bd = transposeFlat(b.Data, b.Rows, b.Cols)
		br, bc = b.Cols, b.Rows
	}
	prod := matMul(ad, ar, ac, bd, br, bc)
	out := NewFloatMat(ar, bc)
	for i := range prod {
		out.Data[i] = alpha * prod[i]
	}
	if c != nil {
		if c.Rows != ar || c.Cols != bc {
			panic(fmt.Sprintf("cv: Gemm c is %dx%d, want %dx%d", c.Rows, c.Cols, ar, bc))
		}
		for i := range out.Data {
			out.Data[i] += beta * c.Data[i]
		}
	}
	return out
}

// transposeFlat returns the transpose of a row-major r×c matrix as c×r.
func transposeFlat(d []float64, r, c int) []float64 {
	out := make([]float64, len(d))
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			out[j*r+i] = d[i*c+j]
		}
	}
	return out
}

// Eigen computes the eigenvalues and eigenvectors of a real symmetric matrix
// using the cyclic Jacobi method. The eigenvalues are returned in descending
// order and the returned matrix holds the corresponding unit eigenvectors as
// its rows, matching OpenCV's cv::eigen layout. It panics if the matrix is not
// square.
func Eigen(src *FloatMat) (eigenvalues []float64, eigenvectors *FloatMat) {
	requireSquareFloat(src, "Eigen")
	n := src.Rows
	vals, vecsCols := symmetricJacobi(src.Data, n)
	// vecsCols holds eigenvectors as columns; sort descending by value and
	// emit them as rows.
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool { return vals[idx[i]] > vals[idx[j]] })
	eigenvalues = make([]float64, n)
	eigenvectors = NewFloatMat(n, n)
	for r, id := range idx {
		eigenvalues[r] = vals[id]
		for row := 0; row < n; row++ {
			eigenvectors.Data[r*n+row] = vecsCols[row*n+id]
		}
	}
	return eigenvalues, eigenvectors
}

// symmetricJacobi diagonalises a symmetric n×n matrix (row-major) and returns
// its eigenvalues and eigenvectors (stored as columns of the returned slice).
func symmetricJacobi(src []float64, n int) (vals []float64, vecs []float64) {
	a := make([]float64, n*n)
	copy(a, src)
	v := make([]float64, n*n)
	for i := 0; i < n; i++ {
		v[i*n+i] = 1
	}
	if n == 1 {
		return []float64{a[0]}, v
	}
	for sweep := 0; sweep < 100; sweep++ {
		off := 0.0
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				off += a[p*n+q] * a[p*n+q]
			}
		}
		if off < 1e-30 {
			break
		}
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				apq := a[p*n+q]
				if math.Abs(apq) < 1e-300 {
					continue
				}
				app := a[p*n+p]
				aqq := a[q*n+q]
				theta := (aqq - app) / (2 * apq)
				t := sign1(theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				if theta == 0 {
					t = 1
				}
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				// Rotate rows/cols p and q.
				for k := 0; k < n; k++ {
					akp := a[k*n+p]
					akq := a[k*n+q]
					a[k*n+p] = c*akp - s*akq
					a[k*n+q] = s*akp + c*akq
				}
				for k := 0; k < n; k++ {
					apk := a[p*n+k]
					aqk := a[q*n+k]
					a[p*n+k] = c*apk - s*aqk
					a[q*n+k] = s*apk + c*aqk
				}
				for k := 0; k < n; k++ {
					vkp := v[k*n+p]
					vkq := v[k*n+q]
					v[k*n+p] = c*vkp - s*vkq
					v[k*n+q] = s*vkp + c*vkq
				}
			}
		}
	}
	vals = make([]float64, n)
	for i := 0; i < n; i++ {
		vals[i] = a[i*n+i]
	}
	return vals, v
}

// sign1 returns -1 for negative input and +1 otherwise (including zero).
func sign1(v float64) float64 {
	if v < 0 {
		return -1
	}
	return 1
}

// SVDecomp computes the singular value decomposition of a matrix A so that
// A = u * diag(w) * vt. It uses the one-sided Jacobi method and returns the
// singular values w in descending order, the left singular vectors as the
// columns of u (rows×n), and the transposed right singular vectors vt (n×n),
// where n is the number of columns of the input.
func SVDecomp(src *FloatMat) (w []float64, u, vt *FloatMat) {
	m, n := src.Rows, src.Cols
	// Work on a copy of the columns of A.
	a := make([]float64, m*n)
	copy(a, src.Data)
	v := make([]float64, n*n)
	for i := 0; i < n; i++ {
		v[i*n+i] = 1
	}
	for sweep := 0; sweep < 60; sweep++ {
		converged := true
		for i := 0; i < n; i++ {
			for j := i + 1; j < n; j++ {
				var alpha, beta, gamma float64
				for k := 0; k < m; k++ {
					ai := a[k*n+i]
					aj := a[k*n+j]
					alpha += ai * ai
					beta += aj * aj
					gamma += ai * aj
				}
				if math.Abs(gamma) < 1e-300 {
					continue
				}
				denom := math.Sqrt(alpha * beta)
				if denom > 0 && math.Abs(gamma)/denom > 1e-14 {
					converged = false
				}
				zeta := (beta - alpha) / (2 * gamma)
				t := sign1(zeta) / (math.Abs(zeta) + math.Sqrt(1+zeta*zeta))
				c := 1 / math.Sqrt(1+t*t)
				s := c * t
				for k := 0; k < m; k++ {
					ai := a[k*n+i]
					aj := a[k*n+j]
					a[k*n+i] = c*ai - s*aj
					a[k*n+j] = s*ai + c*aj
				}
				for k := 0; k < n; k++ {
					vi := v[k*n+i]
					vj := v[k*n+j]
					v[k*n+i] = c*vi - s*vj
					v[k*n+j] = s*vi + c*vj
				}
			}
		}
		if converged {
			break
		}
	}
	// Singular values are the norms of the columns of the rotated A.
	sigma := make([]float64, n)
	uCols := make([]float64, m*n)
	for j := 0; j < n; j++ {
		var norm float64
		for k := 0; k < m; k++ {
			norm += a[k*n+j] * a[k*n+j]
		}
		norm = math.Sqrt(norm)
		sigma[j] = norm
		if norm > 1e-300 {
			for k := 0; k < m; k++ {
				uCols[k*n+j] = a[k*n+j] / norm
			}
		}
	}
	// Sort by singular value descending.
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(i, j int) bool { return sigma[idx[i]] > sigma[idx[j]] })
	w = make([]float64, n)
	u = NewFloatMat(m, n)
	vt = NewFloatMat(n, n)
	for col, id := range idx {
		w[col] = sigma[id]
		for k := 0; k < m; k++ {
			u.Data[k*n+col] = uCols[k*n+id]
		}
		for k := 0; k < n; k++ {
			vt.Data[col*n+k] = v[k*n+id]
		}
	}
	return w, u, vt
}

// Mahalanobis returns the Mahalanobis distance between two vectors v1 and v2
// given the inverse covariance matrix icovar. The vectors may be laid out as
// rows or columns; their total element count must match the dimension of
// icovar. It panics on a dimension mismatch.
func Mahalanobis(v1, v2, icovar *FloatMat) float64 {
	n := len(v1.Data)
	if len(v2.Data) != n || icovar.Rows != n || icovar.Cols != n {
		panic(fmt.Sprintf("cv: Mahalanobis dimension mismatch len(v1)=%d len(v2)=%d icovar=%dx%d",
			n, len(v2.Data), icovar.Rows, icovar.Cols))
	}
	d := make([]float64, n)
	for i := 0; i < n; i++ {
		d[i] = v1.Data[i] - v2.Data[i]
	}
	var sum float64
	for i := 0; i < n; i++ {
		var row float64
		for j := 0; j < n; j++ {
			row += icovar.Data[i*n+j] * d[j]
		}
		sum += d[i] * row
	}
	if sum < 0 {
		sum = 0
	}
	return math.Sqrt(sum)
}

// CalcCovarMatrix computes the covariance matrix and mean of a set of samples.
// Each row of data is one observation and each column one variable. The
// returned covariance is a variables×variables matrix and mean is a
// 1×variables row vector. When normalize is true the covariance is divided by
// the number of samples, otherwise it is a scatter matrix (OpenCV's default is
// unnormalised). It panics on empty input.
func CalcCovarMatrix(data *FloatMat, normalize bool) (covar, mean *FloatMat) {
	rows, cols := data.Rows, data.Cols
	if rows == 0 || cols == 0 {
		panic("cv: CalcCovarMatrix on empty data")
	}
	mean = NewFloatMat(1, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			mean.Data[j] += data.Data[i*cols+j]
		}
	}
	for j := 0; j < cols; j++ {
		mean.Data[j] /= float64(rows)
	}
	covar = NewFloatMat(cols, cols)
	dev := make([]float64, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			dev[j] = data.Data[i*cols+j] - mean.Data[j]
		}
		for a := 0; a < cols; a++ {
			for b := 0; b < cols; b++ {
				covar.Data[a*cols+b] += dev[a] * dev[b]
			}
		}
	}
	if normalize {
		inv := 1 / float64(rows)
		for i := range covar.Data {
			covar.Data[i] *= inv
		}
	}
	return covar, mean
}

// PCACompute performs principal component analysis on a set of samples (one
// observation per row) and returns the sample mean (1×variables), the first
// maxComponents principal axes as the rows of eigenvectors, and the associated
// eigenvalues (variances) in descending order. Passing maxComponents <= 0 or
// greater than the number of variables keeps all components. It panics on empty
// input.
func PCACompute(data *FloatMat, maxComponents int) (mean, eigenvectors *FloatMat, eigenvalues []float64) {
	rows, cols := data.Rows, data.Cols
	if rows == 0 || cols == 0 {
		panic("cv: PCACompute on empty data")
	}
	covar, mean := CalcCovarMatrix(data, true)
	vals, vecs := Eigen(covar)
	if maxComponents <= 0 || maxComponents > cols {
		maxComponents = cols
	}
	eigenvectors = NewFloatMat(maxComponents, cols)
	eigenvalues = make([]float64, maxComponents)
	for k := 0; k < maxComponents; k++ {
		eigenvalues[k] = vals[k]
		copy(eigenvectors.Data[k*cols:(k+1)*cols], vecs.Data[k*cols:(k+1)*cols])
	}
	return mean, eigenvectors, eigenvalues
}

// PCAProject projects data (one observation per row) onto the principal
// subspace defined by mean and eigenvectors (one axis per row), returning the
// coefficients with one row per observation and one column per component. It
// panics on a dimension mismatch.
func PCAProject(data, mean, eigenvectors *FloatMat) *FloatMat {
	rows, cols := data.Rows, data.Cols
	if mean.Cols != cols || eigenvectors.Cols != cols {
		panic(fmt.Sprintf("cv: PCAProject dimension mismatch data cols=%d mean cols=%d ev cols=%d",
			cols, mean.Cols, eigenvectors.Cols))
	}
	k := eigenvectors.Rows
	out := NewFloatMat(rows, k)
	dev := make([]float64, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			dev[j] = data.Data[i*cols+j] - mean.Data[j]
		}
		for c := 0; c < k; c++ {
			var s float64
			for j := 0; j < cols; j++ {
				s += eigenvectors.Data[c*cols+j] * dev[j]
			}
			out.Data[i*k+c] = s
		}
	}
	return out
}

// PCABackProject reconstructs observations from their principal-component
// coefficients (one observation per row) using mean and eigenvectors (one axis
// per row), returning approximate observations with one row each and one column
// per original variable. It is the inverse of [PCAProject]. It panics on a
// dimension mismatch.
func PCABackProject(coeffs, mean, eigenvectors *FloatMat) *FloatMat {
	rows := coeffs.Rows
	k := coeffs.Cols
	cols := eigenvectors.Cols
	if eigenvectors.Rows != k || mean.Cols != cols {
		panic(fmt.Sprintf("cv: PCABackProject dimension mismatch coeffs cols=%d ev rows=%d mean cols=%d",
			k, eigenvectors.Rows, mean.Cols))
	}
	out := NewFloatMat(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			v := mean.Data[j]
			for c := 0; c < k; c++ {
				v += coeffs.Data[i*k+c] * eigenvectors.Data[c*cols+j]
			}
			out.Data[i*cols+j] = v
		}
	}
	return out
}
