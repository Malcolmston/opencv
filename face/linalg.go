package face

import (
	"math"
	"sort"
)

// This file contains the small, self-contained linear-algebra kernels the
// recognizers need: a cyclic Jacobi eigensolver for real symmetric matrices and
// a principal-component analysis built on it. Everything operates on dense
// row-major [][]float64 matrices and uses only the standard library, so the
// package depends on nothing beyond the root cv package.

// jacobiEigen computes the eigenvalues and eigenvectors of the real symmetric
// matrix a (n×n, row-major) using the classical cyclic Jacobi rotation method.
// The input is not modified; a working copy is made. The results are sorted by
// eigenvalue in descending order: values[j] is the j-th eigenvalue and
// vectors[j] is its unit-length eigenvector (length n). The eigenvectors are
// mutually orthonormal.
//
// a is assumed symmetric; only correctness for symmetric input is guaranteed.
func jacobiEigen(a [][]float64) (values []float64, vectors [][]float64) {
	n := len(a)
	if n == 0 {
		return nil, nil
	}
	// Working copy A and accumulated rotation V (starts as the identity).
	A := make([][]float64, n)
	V := make([][]float64, n)
	for i := 0; i < n; i++ {
		A[i] = append([]float64(nil), a[i]...)
		V[i] = make([]float64, n)
		V[i][i] = 1
	}

	const maxSweeps = 100
	for sweep := 0; sweep < maxSweeps; sweep++ {
		// Sum of squared off-diagonal magnitudes; stop once negligible.
		off := 0.0
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				off += A[p][q] * A[p][q]
			}
		}
		if off <= 1e-300 {
			break
		}
		for p := 0; p < n-1; p++ {
			for q := p + 1; q < n; q++ {
				apq := A[p][q]
				if apq == 0 {
					continue
				}
				app := A[p][p]
				aqq := A[q][q]
				// Rotation angle that zeroes A[p][q]:
				// tan(2θ) = 2·apq / (app − aqq).
				theta := 0.5 * math.Atan2(2*apq, app-aqq)
				c := math.Cos(theta)
				s := math.Sin(theta)

				// Update rows/columns p and q of the symmetric matrix.
				for i := 0; i < n; i++ {
					if i == p || i == q {
						continue
					}
					aip := A[i][p]
					aiq := A[i][q]
					A[i][p] = c*aip + s*aiq
					A[p][i] = A[i][p]
					A[i][q] = -s*aip + c*aiq
					A[q][i] = A[i][q]
				}
				A[p][p] = c*c*app + 2*c*s*apq + s*s*aqq
				A[q][q] = s*s*app - 2*c*s*apq + c*c*aqq
				A[p][q] = 0
				A[q][p] = 0

				// Accumulate the rotation into the eigenvector matrix.
				for i := 0; i < n; i++ {
					vip := V[i][p]
					viq := V[i][q]
					V[i][p] = c*vip + s*viq
					V[i][q] = -s*vip + c*viq
				}
			}
		}
	}

	// Extract eigenvalues (diagonal) and eigenvectors (columns of V).
	type pair struct {
		val float64
		vec []float64
	}
	pairs := make([]pair, n)
	for j := 0; j < n; j++ {
		vec := make([]float64, n)
		for i := 0; i < n; i++ {
			vec[i] = V[i][j]
		}
		pairs[j] = pair{val: A[j][j], vec: vec}
	}
	sort.SliceStable(pairs, func(i, k int) bool {
		return pairs[i].val > pairs[k].val
	})
	values = make([]float64, n)
	vectors = make([][]float64, n)
	for j := range pairs {
		values[j] = pairs[j].val
		vectors[j] = pairs[j].vec
	}
	return values, vectors
}

// pcaModel is a fitted principal-component analysis: a mean vector and a set of
// orthonormal principal axes (the "eigenfaces" for face data) ordered by
// descending variance.
type pcaModel struct {
	mean        []float64   // length d
	components  [][]float64 // k unit vectors, each length d
	eigenvalues []float64   // k variances matching components
}

// computePCA fits a PCA to data (n samples × d features) keeping at most
// maxComponents axes. When maxComponents <= 0 every axis with non-negligible
// variance is kept.
//
// It uses the Turk–Pentland "small matrix" trick: rather than the d×d
// covariance it eigendecomposes the n×n Gram matrix (1/n)·XXᵀ of the
// mean-centred data, whose non-zero eigenvalues equal those of the covariance,
// and maps each eigenvector v back to a covariance eigenvector Xᵀv. This is
// exact and stays cheap even when d (the number of pixels) is large.
func computePCA(data [][]float64, maxComponents int) *pcaModel {
	n := len(data)
	if n == 0 {
		panic("face: computePCA requires at least one sample")
	}
	d := len(data[0])

	mean := make([]float64, d)
	for _, row := range data {
		for j, v := range row {
			mean[j] += v
		}
	}
	for j := range mean {
		mean[j] /= float64(n)
	}

	// Mean-centred data X (n×d).
	X := make([][]float64, n)
	for i := range data {
		row := make([]float64, d)
		for j := range data[i] {
			row[j] = data[i][j] - mean[j]
		}
		X[i] = row
	}

	// Gram matrix L = (1/n)·XXᵀ (n×n, symmetric).
	L := make([][]float64, n)
	for i := 0; i < n; i++ {
		L[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for k := i; k < n; k++ {
			var s float64
			xi, xk := X[i], X[k]
			for j := 0; j < d; j++ {
				s += xi[j] * xk[j]
			}
			s /= float64(n)
			L[i][k] = s
			L[k][i] = s
		}
	}

	vals, vecs := jacobiEigen(L)

	// Keep axes with variance well above the largest one's floor.
	var maxVal float64
	if len(vals) > 0 && vals[0] > 0 {
		maxVal = vals[0]
	}
	threshold := maxVal * 1e-8

	model := &pcaModel{mean: mean}
	for idx := 0; idx < len(vals); idx++ {
		if maxComponents > 0 && len(model.components) >= maxComponents {
			break
		}
		if vals[idx] <= threshold {
			break
		}
		// Map Gram eigenvector back to a covariance eigenvector: Xᵀ·v.
		v := vecs[idx]
		comp := make([]float64, d)
		for i := 0; i < n; i++ {
			vi := v[i]
			if vi == 0 {
				continue
			}
			xi := X[i]
			for j := 0; j < d; j++ {
				comp[j] += vi * xi[j]
			}
		}
		// Normalise to unit length.
		var norm float64
		for j := 0; j < d; j++ {
			norm += comp[j] * comp[j]
		}
		norm = math.Sqrt(norm)
		if norm < 1e-12 {
			continue
		}
		inv := 1 / norm
		for j := 0; j < d; j++ {
			comp[j] *= inv
		}
		model.components = append(model.components, comp)
		model.eigenvalues = append(model.eigenvalues, vals[idx])
	}
	return model
}

// dim returns the number of retained principal axes.
func (p *pcaModel) dim() int { return len(p.components) }

// project returns the coordinates of v in the principal subspace: the dot
// product of each component with the mean-centred vector.
func (p *pcaModel) project(v []float64) []float64 {
	out := make([]float64, len(p.components))
	for k, comp := range p.components {
		var s float64
		for j := range comp {
			s += comp[j] * (v[j] - p.mean[j])
		}
		out[k] = s
	}
	return out
}

// reconstruct rebuilds an approximation of the original vector from projection
// coefficients, using the first len(coeffs) components (never more than are
// available). The result is mean + Σ coeffs[k]·component[k].
func (p *pcaModel) reconstruct(coeffs []float64) []float64 {
	d := len(p.mean)
	out := make([]float64, d)
	copy(out, p.mean)
	k := len(coeffs)
	if k > len(p.components) {
		k = len(p.components)
	}
	for i := 0; i < k; i++ {
		comp := p.components[i]
		ci := coeffs[i]
		for j := 0; j < d; j++ {
			out[j] += ci * comp[j]
		}
	}
	return out
}
