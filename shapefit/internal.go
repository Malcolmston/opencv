package shapefit

import "math"

// shapefitEps is the numerical tolerance used to guard divisions and detect
// degenerate configurations throughout the package.
const shapefitEps = 1e-12

// shapefitSolve3 solves the 3×3 linear system a·x = b by Gaussian elimination
// with partial pivoting. It returns the solution and true, or the zero vector
// and false when the matrix is singular.
func shapefitSolve3(a [3][3]float64, b [3]float64) ([3]float64, bool) {
	// Work on an augmented copy.
	m := [3][4]float64{
		{a[0][0], a[0][1], a[0][2], b[0]},
		{a[1][0], a[1][1], a[1][2], b[1]},
		{a[2][0], a[2][1], a[2][2], b[2]},
	}
	for col := 0; col < 3; col++ {
		// Partial pivot.
		piv := col
		best := math.Abs(m[col][col])
		for r := col + 1; r < 3; r++ {
			if v := math.Abs(m[r][col]); v > best {
				best = v
				piv = r
			}
		}
		if best < shapefitEps {
			return [3]float64{}, false
		}
		m[col], m[piv] = m[piv], m[col]
		// Eliminate below.
		for r := 0; r < 3; r++ {
			if r == col {
				continue
			}
			f := m[r][col] / m[col][col]
			for c := col; c < 4; c++ {
				m[r][c] -= f * m[col][c]
			}
		}
	}
	return [3]float64{
		m[0][3] / m[0][0],
		m[1][3] / m[1][1],
		m[2][3] / m[2][2],
	}, true
}

// shapefitInv3 returns the inverse of a 3×3 matrix and true, or a zero matrix
// and false if the matrix is singular.
func shapefitInv3(m [3][3]float64) ([3][3]float64, bool) {
	c00 := m[1][1]*m[2][2] - m[1][2]*m[2][1]
	c01 := m[1][2]*m[2][0] - m[1][0]*m[2][2]
	c02 := m[1][0]*m[2][1] - m[1][1]*m[2][0]
	det := m[0][0]*c00 + m[0][1]*c01 + m[0][2]*c02
	if math.Abs(det) < shapefitEps {
		return [3][3]float64{}, false
	}
	inv := 1.0 / det
	var out [3][3]float64
	out[0][0] = c00 * inv
	out[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * inv
	out[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * inv
	out[1][0] = c01 * inv
	out[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * inv
	out[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * inv
	out[2][0] = c02 * inv
	out[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * inv
	out[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * inv
	return out, true
}

// shapefitMul3 returns the matrix product a·b of two 3×3 matrices.
func shapefitMul3(a, b [3][3]float64) [3][3]float64 {
	var out [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			s := 0.0
			for k := 0; k < 3; k++ {
				s += a[i][k] * b[k][j]
			}
			out[i][j] = s
		}
	}
	return out
}

// shapefitMatVec3 returns the matrix-vector product m·v.
func shapefitMatVec3(m [3][3]float64, v [3]float64) [3]float64 {
	return [3]float64{
		m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2],
		m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2],
		m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2],
	}
}

// shapefitTranspose3 returns the transpose of a 3×3 matrix.
func shapefitTranspose3(m [3][3]float64) [3][3]float64 {
	return [3][3]float64{
		{m[0][0], m[1][0], m[2][0]},
		{m[0][1], m[1][1], m[2][1]},
		{m[0][2], m[1][2], m[2][2]},
	}
}

// shapefitCubicRoots returns the real roots of the monic cubic
// t³ + a·t² + b·t + c = 0. It returns between one and three roots.
func shapefitCubicRoots(a, b, c float64) []float64 {
	// Depress: t = y - a/3, giving y³ + p·y + q = 0.
	p := b - a*a/3
	q := 2*a*a*a/27 - a*b/3 + c
	shift := a / 3
	if math.Abs(p) < shapefitEps && math.Abs(q) < shapefitEps {
		return []float64{-shift}
	}
	disc := q*q/4 + p*p*p/27
	if disc > shapefitEps {
		// One real root.
		s := math.Sqrt(disc)
		u := math.Cbrt(-q/2 + s)
		v := math.Cbrt(-q/2 - s)
		return []float64{u + v - shift}
	}
	if disc > -shapefitEps {
		// Triple or double root (disc ≈ 0).
		if math.Abs(q) < shapefitEps {
			return []float64{-shift}
		}
		y1 := 3 * q / p
		y2 := -3 * q / (2 * p)
		return []float64{y1 - shift, y2 - shift}
	}
	// Three distinct real roots (p < 0 necessarily here).
	m := 2 * math.Sqrt(-p/3)
	arg := 3 * q / (p * m)
	if arg > 1 {
		arg = 1
	} else if arg < -1 {
		arg = -1
	}
	phi := math.Acos(arg)
	roots := make([]float64, 3)
	for k := 0; k < 3; k++ {
		roots[k] = m*math.Cos((phi-2*math.Pi*float64(k))/3) - shift
	}
	return roots
}

// shapefitNull3 returns a nonzero vector v such that m·v ≈ 0, assuming m is
// (numerically) rank-deficient. It picks the largest cross product of the row
// vectors, which spans the null direction of a rank-2 matrix.
func shapefitNull3(m [3][3]float64) [3]float64 {
	rows := [3][3]float64{m[0], m[1], m[2]}
	best := [3]float64{}
	bestN := -1.0
	pairs := [3][2]int{{0, 1}, {0, 2}, {1, 2}}
	for _, pr := range pairs {
		x := shapefitCross3(rows[pr[0]], rows[pr[1]])
		n := x[0]*x[0] + x[1]*x[1] + x[2]*x[2]
		if n > bestN {
			bestN = n
			best = x
		}
	}
	if bestN <= 0 {
		return [3]float64{1, 0, 0}
	}
	return best
}

// shapefitCross3 returns the 3D cross product a×b.
func shapefitCross3(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

// shapefitSym2Eig returns the eigenvalues and eigenvectors of the symmetric
// 2×2 matrix [[a, b], [b, d]]. It returns (l1, v1, l2, v2) with the
// eigenvectors unit length.
func shapefitSym2Eig(a, b, d float64) (l1 float64, v1 [2]float64, l2 float64, v2 [2]float64) {
	tr := a + d
	s := math.Sqrt((a-d)*(a-d) + 4*b*b)
	l1 = (tr + s) / 2
	l2 = (tr - s) / 2
	if math.Abs(b) > shapefitEps {
		// Eigenvector for l: ((l - d), b) is a valid choice.
		v1 = shapefitNorm2(l1-d, b)
		v2 = shapefitNorm2(l2-d, b)
	} else {
		// Already diagonal; axes aligned. l1 is the larger eigenvalue, so its
		// eigenvector is the axis of whichever diagonal entry is larger.
		if a >= d {
			v1 = [2]float64{1, 0}
			v2 = [2]float64{0, 1}
		} else {
			v1 = [2]float64{0, 1}
			v2 = [2]float64{1, 0}
		}
	}
	return
}

// shapefitNorm2 returns the unit vector along (x, y), or (1, 0) if degenerate.
func shapefitNorm2(x, y float64) [2]float64 {
	n := math.Hypot(x, y)
	if n < shapefitEps {
		return [2]float64{1, 0}
	}
	return [2]float64{x / n, y / n}
}

// shapefitWrapPi maps an angle to the half-open interval (-π/2, π/2], the
// canonical range for an undirected line or axis orientation.
func shapefitWrapPi(a float64) float64 {
	for a > math.Pi/2 {
		a -= math.Pi
	}
	for a <= -math.Pi/2 {
		a += math.Pi
	}
	return a
}
