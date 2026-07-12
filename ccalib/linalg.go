package ccalib

import "math"

// This file holds the small, self-contained linear-algebra kernels used across
// the package. Everything here is implemented from scratch against the Go
// standard library so that ccalib depends only on the root cv package and the
// stdlib. None of these helpers are exported; they exist purely to support the
// public calibration and pattern-matching API.

// mul3 returns the matrix product a·b of two 3×3 matrices.
func mul3(a, b [3][3]float64) [3][3]float64 {
	var r [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			var s float64
			for k := 0; k < 3; k++ {
				s += a[i][k] * b[k][j]
			}
			r[i][j] = s
		}
	}
	return r
}

// transpose3 returns the transpose of a 3×3 matrix.
func transpose3(a [3][3]float64) [3][3]float64 {
	var r [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			r[j][i] = a[i][j]
		}
	}
	return r
}

// matVec3 returns the matrix–vector product a·v.
func matVec3(a [3][3]float64, v [3]float64) [3]float64 {
	return [3]float64{
		a[0][0]*v[0] + a[0][1]*v[1] + a[0][2]*v[2],
		a[1][0]*v[0] + a[1][1]*v[1] + a[1][2]*v[2],
		a[2][0]*v[0] + a[2][1]*v[1] + a[2][2]*v[2],
	}
}

// det3 returns the determinant of a 3×3 matrix.
func det3(m [3][3]float64) float64 {
	return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) -
		m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) +
		m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
}

// inv3 returns the inverse of a 3×3 matrix and reports whether it is
// invertible (a determinant of near-zero magnitude yields ok == false).
func inv3(m [3][3]float64) ([3][3]float64, bool) {
	det := det3(m)
	if math.Abs(det) < 1e-18 {
		return [3][3]float64{}, false
	}
	id := 1 / det
	var r [3][3]float64
	r[0][0] = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * id
	r[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * id
	r[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * id
	r[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * id
	r[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * id
	r[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * id
	r[2][0] = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * id
	r[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * id
	r[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * id
	return r, true
}

// cross3 returns the vector cross product a×b.
func cross3(a, b [3]float64) [3]float64 {
	return [3]float64{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

// norm3 returns the Euclidean length of a 3-vector.
func norm3(v [3]float64) float64 {
	return math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
}

// dot3 returns the dot product of two 3-vectors.
func dot3(a, b [3]float64) float64 { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }

// sub3 returns a − b.
func sub3(a, b [3]float64) [3]float64 { return [3]float64{a[0] - b[0], a[1] - b[1], a[2] - b[2]} }

// add3 returns a + b.
func add3(a, b [3]float64) [3]float64 { return [3]float64{a[0] + b[0], a[1] + b[1], a[2] + b[2]} }

// scale3 returns s·v.
func scale3(v [3]float64, s float64) [3]float64 { return [3]float64{v[0] * s, v[1] * s, v[2] * s} }

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

// signf returns -1, 0 or +1 following the sign of x.
func signf(x float64) float64 {
	switch {
	case x > 0:
		return 1
	case x < 0:
		return -1
	default:
		return 0
	}
}

// sq returns x².
func sq(x float64) float64 { return x * x }

// jacobiEigenSym computes the eigenvalues and eigenvectors of a real symmetric
// matrix using the cyclic Jacobi rotation method. The input a is treated as
// symmetric and is not modified. It returns the eigenvalues and a matrix whose
// columns are the corresponding (orthonormal) eigenvectors: vecs[i][j] is the
// i-th component of the eigenvector for vals[j]. The routine is numerically
// robust for the small (3×3 to 12×12) matrices used throughout the package.
func jacobiEigenSym(a [][]float64) (vals []float64, vecs [][]float64) {
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
		if off < 1e-30 {
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
					t = signf(theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
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
	vals = make([]float64, n)
	for i := 0; i < n; i++ {
		vals[i] = m[i][i]
	}
	return vals, v
}

// smallestEigvec returns the eigenvector of the symmetric matrix a associated
// with its smallest eigenvalue, as a freshly allocated slice. It is the null
// space of a rank-deficient normal-equation matrix and underpins the
// homography and pose-from-rays solvers.
func smallestEigvec(a [][]float64) []float64 {
	vals, vecs := jacobiEigenSym(a)
	best := 0
	for i := 1; i < len(vals); i++ {
		if vals[i] < vals[best] {
			best = i
		}
	}
	n := len(a)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = vecs[i][best]
	}
	return out
}

// svd3 computes the singular value decomposition A = U·diag(S)·Vᵀ of a 3×3
// matrix. The singular values S are returned in non-increasing order with the
// columns of U and V ordered to match. It is used to orthonormalise rotation
// matrices.
func svd3(a [3][3]float64) (u [3][3]float64, s [3]float64, v [3][3]float64) {
	at := transpose3(a)
	ata := mul3(at, a)
	dyn := [][]float64{
		{ata[0][0], ata[0][1], ata[0][2]},
		{ata[1][0], ata[1][1], ata[1][2]},
		{ata[2][0], ata[2][1], ata[2][2]},
	}
	vals, vecs := jacobiEigenSym(dyn)
	idx := []int{0, 1, 2}
	for i := 0; i < 3; i++ {
		for j := i + 1; j < 3; j++ {
			if vals[idx[j]] > vals[idx[i]] {
				idx[i], idx[j] = idx[j], idx[i]
			}
		}
	}
	for c := 0; c < 3; c++ {
		col := idx[c]
		ev := vals[col]
		if ev < 0 {
			ev = 0
		}
		s[c] = math.Sqrt(ev)
		for r := 0; r < 3; r++ {
			v[r][c] = vecs[r][col]
		}
	}
	for c := 0; c < 3; c++ {
		vc := [3]float64{v[0][c], v[1][c], v[2][c]}
		av := matVec3(a, vc)
		if s[c] > 1e-12 && s[c] > 1e-6*s[0] {
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
		cr := cross3(a1, a0)
		nn := norm3(cr)
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
		piv := col
		best := math.Abs(m[col][col])
		for r := col + 1; r < n; r++ {
			if val := math.Abs(m[r][col]); val > best {
				best = val
				piv = r
			}
		}
		if best < 1e-18 {
			return nil, false
		}
		m[col], m[piv] = m[piv], m[col]
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

// nullspaceVec returns the unit vector x minimising ‖A·x‖ subject to ‖x‖ = 1,
// i.e. the right singular vector of A for its smallest singular value, obtained
// as the smallest-eigenvalue eigenvector of AᵀA. rows lists the design-matrix
// rows, each of length n.
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

// rodriguesToMatrix converts a rotation vector (axis-angle, whose direction is
// the rotation axis and whose magnitude is the rotation angle in radians) into
// the corresponding 3×3 rotation matrix. A zero vector yields the identity.
func rodriguesToMatrix(rvec [3]float64) [3][3]float64 {
	theta := norm3(rvec)
	if theta < 1e-12 {
		return [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	}
	kx := rvec[0] / theta
	ky := rvec[1] / theta
	kz := rvec[2] / theta
	c := math.Cos(theta)
	s := math.Sin(theta)
	c1 := 1 - c
	return [3][3]float64{
		{c + kx*kx*c1, kx*ky*c1 - kz*s, kx*kz*c1 + ky*s},
		{ky*kx*c1 + kz*s, c + ky*ky*c1, ky*kz*c1 - kx*s},
		{kz*kx*c1 - ky*s, kz*ky*c1 + kx*s, c + kz*kz*c1},
	}
}

// rodriguesToVector converts a 3×3 rotation matrix into the equivalent rotation
// vector (axis-angle) via the inverse of Rodrigues' formula, handling the
// small-angle and near-π degenerate cases.
func rodriguesToVector(r [3][3]float64) [3]float64 {
	trace := r[0][0] + r[1][1] + r[2][2]
	cosTheta := (trace - 1) / 2
	if cosTheta > 1 {
		cosTheta = 1
	} else if cosTheta < -1 {
		cosTheta = -1
	}
	theta := math.Acos(cosTheta)
	if theta < 1e-9 {
		return [3]float64{0, 0, 0}
	}
	if math.Pi-theta < 1e-6 {
		xx := (r[0][0] + 1) / 2
		yy := (r[1][1] + 1) / 2
		zz := (r[2][2] + 1) / 2
		xy := (r[0][1] + r[1][0]) / 4
		xz := (r[0][2] + r[2][0]) / 4
		yz := (r[1][2] + r[2][1]) / 4
		var ax, ay, az float64
		switch {
		case xx >= yy && xx >= zz:
			ax = math.Sqrt(math.Max(xx, 0))
			ay = xy / ax
			az = xz / ax
		case yy >= zz:
			ay = math.Sqrt(math.Max(yy, 0))
			ax = xy / ay
			az = yz / ay
		default:
			az = math.Sqrt(math.Max(zz, 0))
			ax = xz / az
			ay = yz / az
		}
		n := math.Sqrt(ax*ax + ay*ay + az*az)
		if n < 1e-15 {
			return [3]float64{0, 0, 0}
		}
		return [3]float64{theta * ax / n, theta * ay / n, theta * az / n}
	}
	k := theta / (2 * math.Sin(theta))
	return [3]float64{
		k * (r[2][1] - r[1][2]),
		k * (r[0][2] - r[2][0]),
		k * (r[1][0] - r[0][1]),
	}
}

// levenbergMarquardt minimises ‖residual(p)‖² over the parameter vector p using
// the Levenberg–Marquardt algorithm with a numerically-estimated Jacobian. It
// returns the refined parameters and the final root-mean-square residual. The
// residual closure must return a fixed-length vector for any p. It is used to
// polish the closed-form calibration estimates.
func levenbergMarquardt(p []float64, residual func([]float64) []float64, maxIter int) ([]float64, float64) {
	n := len(p)
	cur := make([]float64, n)
	copy(cur, p)
	r := residual(cur)
	cost := dotSlice(r, r)
	lambda := 1e-3
	for iter := 0; iter < maxIter; iter++ {
		m := len(r)
		// Numerical Jacobian (forward differences).
		jac := make([][]float64, m)
		for i := 0; i < m; i++ {
			jac[i] = make([]float64, n)
		}
		for j := 0; j < n; j++ {
			h := 1e-6 * (math.Abs(cur[j]) + 1e-6)
			saved := cur[j]
			cur[j] = saved + h
			rp := residual(cur)
			cur[j] = saved
			for i := 0; i < m; i++ {
				jac[i][j] = (rp[i] - r[i]) / h
			}
		}
		// Normal equations JᵀJ and Jᵀr.
		jtj := make([][]float64, n)
		jtr := make([]float64, n)
		for i := 0; i < n; i++ {
			jtj[i] = make([]float64, n)
		}
		for i := 0; i < m; i++ {
			for a := 0; a < n; a++ {
				jtr[a] += jac[i][a] * r[i]
				for b := 0; b < n; b++ {
					jtj[a][b] += jac[i][a] * jac[i][b]
				}
			}
		}
		improved := false
		for try := 0; try < 8; try++ {
			aug := make([][]float64, n)
			rhs := make([]float64, n)
			for i := 0; i < n; i++ {
				aug[i] = make([]float64, n)
				copy(aug[i], jtj[i])
				aug[i][i] += lambda * (jtj[i][i] + 1e-12)
				rhs[i] = -jtr[i]
			}
			delta, ok := solveLinear(aug, rhs)
			if !ok {
				lambda *= 10
				continue
			}
			cand := make([]float64, n)
			for i := 0; i < n; i++ {
				cand[i] = cur[i] + delta[i]
			}
			rc := residual(cand)
			cc := dotSlice(rc, rc)
			if cc < cost {
				copy(cur, cand)
				r = rc
				cost = cc
				lambda *= 0.5
				improved = true
				break
			}
			lambda *= 10
		}
		if !improved {
			break
		}
		if lambda > 1e12 {
			break
		}
	}
	rms := 0.0
	if len(r) > 0 {
		rms = math.Sqrt(cost / float64(len(r)))
	}
	return cur, rms
}

// dotSlice returns the dot product of two equal-length slices.
func dotSlice(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
