package transforms2

import "math"

// transforms2solve solves the dense linear system a*x = b for x using Gaussian
// elimination with partial pivoting. a is an n x n matrix (row-major slice of
// rows) and b has length n. It reports false if the system is singular. a and b
// are modified in place.
func transforms2solve(a [][]float64, b []float64) ([]float64, bool) {
	n := len(b)
	for col := 0; col < n; col++ {
		// Partial pivot.
		pivot := col
		best := math.Abs(a[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(a[r][col]); v > best {
				best = v
				pivot = r
			}
		}
		if best < 1e-14 {
			return nil, false
		}
		if pivot != col {
			a[col], a[pivot] = a[pivot], a[col]
			b[col], b[pivot] = b[pivot], b[col]
		}
		// Eliminate below.
		inv := 1 / a[col][col]
		for r := col + 1; r < n; r++ {
			f := a[r][col] * inv
			if f == 0 {
				continue
			}
			for k := col; k < n; k++ {
				a[r][k] -= f * a[col][k]
			}
			b[r] -= f * b[col]
		}
	}
	// Back substitution.
	x := make([]float64, n)
	for r := n - 1; r >= 0; r-- {
		s := b[r]
		for k := r + 1; k < n; k++ {
			s -= a[r][k] * x[k]
		}
		x[r] = s / a[r][r]
	}
	return x, true
}

// transforms2solve3 solves a 3x3 system using the generic solver.
func transforms2solve3(a [3][3]float64, b [3]float64) ([3]float64, bool) {
	m := [][]float64{{a[0][0], a[0][1], a[0][2]}, {a[1][0], a[1][1], a[1][2]}, {a[2][0], a[2][1], a[2][2]}}
	v := []float64{b[0], b[1], b[2]}
	x, ok := transforms2solve(m, v)
	if !ok {
		return [3]float64{}, false
	}
	return [3]float64{x[0], x[1], x[2]}, true
}
