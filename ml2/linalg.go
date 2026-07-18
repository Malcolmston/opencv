package ml2

import "math"

// ml2jacobiEigen computes the eigenvalues and eigenvectors of a real symmetric
// n-by-n matrix using the cyclic Jacobi rotation method. The input matrix is
// not modified. It returns the eigenvalues and a matrix whose column j is the
// unit eigenvector for eigenvalue j (eigvecs[i][j]); the pairs are unsorted.
func ml2jacobiEigen(sym [][]float64) (eigvals []float64, eigvecs [][]float64) {
	n := len(sym)
	a := make([][]float64, n)
	v := make([][]float64, n)
	for i := 0; i < n; i++ {
		a[i] = make([]float64, n)
		copy(a[i], sym[i])
		v[i] = make([]float64, n)
		v[i][i] = 1
	}
	const maxSweeps = 100
	for sweep := 0; sweep < maxSweeps; sweep++ {
		// Sum of off-diagonal magnitudes; stop when negligible.
		off := 0.0
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				off += math.Abs(a[p][q])
			}
		}
		if off < 1e-14 {
			break
		}
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				if math.Abs(a[p][q]) < 1e-300 {
					continue
				}
				theta := (a[q][q] - a[p][p]) / (2 * a[p][q])
				t := math.Copysign(1, theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				if theta == 0 {
					t = 1
				}
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				// Apply rotation to a.
				for k := 0; k < n; k++ {
					akp := a[k][p]
					akq := a[k][q]
					a[k][p] = c*akp - s*akq
					a[k][q] = s*akp + c*akq
				}
				for k := 0; k < n; k++ {
					apk := a[p][k]
					aqk := a[q][k]
					a[p][k] = c*apk - s*aqk
					a[q][k] = s*apk + c*aqk
				}
				// Accumulate rotation into v.
				for k := 0; k < n; k++ {
					vkp := v[k][p]
					vkq := v[k][q]
					v[k][p] = c*vkp - s*vkq
					v[k][q] = s*vkp + c*vkq
				}
			}
		}
	}
	eigvals = make([]float64, n)
	for i := 0; i < n; i++ {
		eigvals[i] = a[i][i]
	}
	return eigvals, v
}

// ml2covariance returns the (biased, divided by n) covariance matrix of a
// centred or uncentred data matrix; the mean is subtracted internally.
func ml2covariance(x [][]float64) [][]float64 {
	n := len(x)
	if n == 0 {
		return nil
	}
	d := len(x[0])
	mean := ml2columnMean(x)
	cov := make([][]float64, d)
	for i := range cov {
		cov[i] = make([]float64, d)
	}
	for _, row := range x {
		for i := 0; i < d; i++ {
			di := row[i] - mean[i]
			for j := i; j < d; j++ {
				cov[i][j] += di * (row[j] - mean[j])
			}
		}
	}
	fn := float64(n)
	for i := 0; i < d; i++ {
		for j := i; j < d; j++ {
			cov[i][j] /= fn
			cov[j][i] = cov[i][j]
		}
	}
	return cov
}

// ml2cholesky returns the lower-triangular Cholesky factor L of a symmetric
// positive-definite matrix, so that L*Lᵀ == m. It returns ok==false if the
// matrix is not positive definite.
func ml2cholesky(m [][]float64) (l [][]float64, ok bool) {
	n := len(m)
	l = make([][]float64, n)
	for i := range l {
		l[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum := m[i][j]
			for k := 0; k < j; k++ {
				sum -= l[i][k] * l[j][k]
			}
			if i == j {
				if sum <= 0 {
					return nil, false
				}
				l[i][j] = math.Sqrt(sum)
			} else {
				l[i][j] = sum / l[j][j]
			}
		}
	}
	return l, true
}

// ml2forwardSolve solves L*x == b for a lower-triangular L.
func ml2forwardSolve(l [][]float64, b []float64) []float64 {
	n := len(l)
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := b[i]
		for k := 0; k < i; k++ {
			sum -= l[i][k] * x[k]
		}
		x[i] = sum / l[i][i]
	}
	return x
}

// ml2backSolveT solves Lᵀ*x == b for a lower-triangular L (upper-triangular Lᵀ).
func ml2backSolveT(l [][]float64, b []float64) []float64 {
	n := len(l)
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := b[i]
		for k := i + 1; k < n; k++ {
			sum -= l[k][i] * x[k]
		}
		x[i] = sum / l[i][i]
	}
	return x
}

// ml2addRidge returns a copy of the symmetric matrix m with a small positive
// value added to its diagonal, improving numerical conditioning.
func ml2addRidge(m [][]float64, ridge float64) [][]float64 {
	n := len(m)
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, n)
		copy(out[i], m[i])
		out[i][i] += ridge
	}
	return out
}
