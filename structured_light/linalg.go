package structured_light

import "math"

// jacobiEigenSymmetric computes the eigenvalues and eigenvectors of a real
// symmetric matrix with the cyclic Jacobi rotation method. input is an n×n
// symmetric matrix (only used as read-only). It returns the eigenvalues and a
// slice of eigenvectors where vecs[k] is the unit eigenvector for vals[k]. The
// method is unconditionally convergent for symmetric matrices and needs no
// external dependency, which suits the small (4×4) systems solved during
// triangulation.
func jacobiEigenSymmetric(input [][]float64) (vals []float64, vecs [][]float64) {
	n := len(input)
	a := make([][]float64, n)
	for i := range a {
		a[i] = append([]float64(nil), input[i]...)
	}
	v := make([][]float64, n)
	for i := range v {
		v[i] = make([]float64, n)
		v[i][i] = 1
	}

	for sweep := 0; sweep < 100; sweep++ {
		off := 0.0
		for p := 0; p < n; p++ {
			for q := p + 1; q < n; q++ {
				off += a[p][q] * a[p][q]
			}
		}
		if off < 1e-30 {
			break
		}
		for p := 0; p < n-1; p++ {
			for q := p + 1; q < n; q++ {
				apq := a[p][q]
				if math.Abs(apq) < 1e-300 {
					continue
				}
				theta := (a[q][q] - a[p][p]) / (2 * apq)
				t := sign(theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				if theta == 0 {
					t = 1
				}
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				rotate(a, v, p, q, c, s, n)
			}
		}
	}

	vals = make([]float64, n)
	for i := 0; i < n; i++ {
		vals[i] = a[i][i]
	}
	vecs = make([][]float64, n)
	for k := 0; k < n; k++ {
		col := make([]float64, n)
		for i := 0; i < n; i++ {
			col[i] = v[i][k]
		}
		vecs[k] = col
	}
	return vals, vecs
}

// rotate applies a Givens rotation with cosine c and sine s in the (p,q) plane
// to the symmetric working matrix a (as Gᵀ·a·G) and accumulates it into the
// eigenvector matrix v.
func rotate(a, v [][]float64, p, q int, c, s float64, n int) {
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
	for k := 0; k < n; k++ {
		vkp := v[k][p]
		vkq := v[k][q]
		v[k][p] = c*vkp - s*vkq
		v[k][q] = s*vkp + c*vkq
	}
}

// sign returns -1 for negative x and +1 otherwise.
func sign(x float64) float64 {
	if x < 0 {
		return -1
	}
	return 1
}

// smallestEigenvector returns the eigenvector associated with the smallest
// eigenvalue, i.e. the best homogeneous solution of an over-determined A·x=0.
func smallestEigenvector(a [][]float64) []float64 {
	vals, vecs := jacobiEigenSymmetric(a)
	min := 0
	for i := 1; i < len(vals); i++ {
		if vals[i] < vals[min] {
			min = i
		}
	}
	return vecs[min]
}
