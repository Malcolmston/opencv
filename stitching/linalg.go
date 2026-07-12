package stitching

import "math"

// This file collects small dense-linear-algebra helpers shared by the camera
// estimation, bundle adjustment and wave-correction code. They operate on plain
// Go slices and arrays so nothing here depends on the cv package.

// solveDense solves the n×n linear system a·x = b with Gauss–Jordan elimination
// and partial pivoting, returning the solution and whether the matrix was
// non-singular. The inputs are not modified.
func solveDense(a [][]float64, b []float64) ([]float64, bool) {
	n := len(b)
	if n == 0 || len(a) != n {
		return nil, false
	}
	m := make([][]float64, n)
	for i := range m {
		if len(a[i]) != n {
			return nil, false
		}
		m[i] = append([]float64(nil), a[i]...)
	}
	x := append([]float64(nil), b...)
	for col := 0; col < n; col++ {
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
		x[col], x[piv] = x[piv], x[col]
		p := m[col][col]
		for c := col; c < n; c++ {
			m[col][c] /= p
		}
		x[col] /= p
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := m[r][col]
			if f == 0 {
				continue
			}
			for c := col; c < n; c++ {
				m[r][c] -= f * m[col][c]
			}
			x[r] -= f * x[col]
		}
	}
	return x, true
}

// mat3 is a row-major 3×3 matrix used for camera intrinsics and rotations. Its
// memory layout matches CameraParams.R ([9]float64) so the two convert directly.
type mat3 [9]float64

// mat3Ident returns the 3×3 identity matrix.
func mat3Ident() mat3 { return mat3{1, 0, 0, 0, 1, 0, 0, 0, 1} }

// mul returns the matrix product a·b.
func (a mat3) mul(b mat3) mat3 {
	var out mat3
	for r := 0; r < 3; r++ {
		for c := 0; c < 3; c++ {
			var s float64
			for k := 0; k < 3; k++ {
				s += a[r*3+k] * b[k*3+c]
			}
			out[r*3+c] = s
		}
	}
	return out
}

// transpose returns aᵀ.
func (a mat3) transpose() mat3 {
	return mat3{a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]}
}

// vec applies the matrix to the column vector (x, y, z).
func (a mat3) vec(x, y, z float64) (float64, float64, float64) {
	return a[0]*x + a[1]*y + a[2]*z,
		a[3]*x + a[4]*y + a[5]*z,
		a[6]*x + a[7]*y + a[8]*z
}

// det returns the determinant.
func (a mat3) det() float64 {
	return a[0]*(a[4]*a[8]-a[5]*a[7]) -
		a[1]*(a[3]*a[8]-a[5]*a[6]) +
		a[2]*(a[3]*a[7]-a[4]*a[6])
}

// inv3 returns the inverse of a and whether it is invertible.
func (a mat3) inv3() (mat3, bool) {
	d := a.det()
	if math.Abs(d) < 1e-18 {
		return mat3{}, false
	}
	id := 1 / d
	var inv mat3
	inv[0] = (a[4]*a[8] - a[5]*a[7]) * id
	inv[1] = (a[2]*a[7] - a[1]*a[8]) * id
	inv[2] = (a[1]*a[5] - a[2]*a[4]) * id
	inv[3] = (a[5]*a[6] - a[3]*a[8]) * id
	inv[4] = (a[0]*a[8] - a[2]*a[6]) * id
	inv[5] = (a[2]*a[3] - a[0]*a[5]) * id
	inv[6] = (a[3]*a[7] - a[4]*a[6]) * id
	inv[7] = (a[1]*a[6] - a[0]*a[7]) * id
	inv[8] = (a[0]*a[4] - a[1]*a[3]) * id
	return inv, true
}

// orthonormalize returns the closest proper rotation to a by Gram–Schmidt
// orthonormalisation of its rows, flipping the last row if needed so the
// determinant is +1. It is used to project a noisy homography-derived matrix
// back onto SO(3).
func (a mat3) orthonormalize() mat3 {
	// Rows of the matrix.
	r0 := [3]float64{a[0], a[1], a[2]}
	r1 := [3]float64{a[3], a[4], a[5]}
	r2 := [3]float64{a[6], a[7], a[8]}
	norm := func(v [3]float64) [3]float64 {
		n := math.Sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
		if n < 1e-18 {
			return v
		}
		return [3]float64{v[0] / n, v[1] / n, v[2] / n}
	}
	dot := func(u, v [3]float64) float64 { return u[0]*v[0] + u[1]*v[1] + u[2]*v[2] }
	sub := func(u, v [3]float64, s float64) [3]float64 {
		return [3]float64{u[0] - s*v[0], u[1] - s*v[1], u[2] - s*v[2]}
	}
	r0 = norm(r0)
	r1 = norm(sub(r1, r0, dot(r1, r0)))
	r2 = sub(r2, r0, dot(r2, r0))
	r2 = norm(sub(r2, r1, dot(r2, r1)))
	m := mat3{
		r0[0], r0[1], r0[2],
		r1[0], r1[1], r1[2],
		r2[0], r2[1], r2[2],
	}
	if m.det() < 0 {
		m[6], m[7], m[8] = -m[6], -m[7], -m[8]
	}
	return m
}

// rodriguesToMat converts an axis-angle (Rodrigues) rotation vector into a 3×3
// rotation matrix.
func rodriguesToMat(rx, ry, rz float64) mat3 {
	theta := math.Sqrt(rx*rx + ry*ry + rz*rz)
	if theta < 1e-12 {
		// First-order approximation I + [r]_x for a tiny rotation.
		return mat3{
			1, -rz, ry,
			rz, 1, -rx,
			-ry, rx, 1,
		}
	}
	kx, ky, kz := rx/theta, ry/theta, rz/theta
	c, s := math.Cos(theta), math.Sin(theta)
	v := 1 - c
	return mat3{
		c + kx*kx*v, kx*ky*v - kz*s, kx*kz*v + ky*s,
		ky*kx*v + kz*s, c + ky*ky*v, ky*kz*v - kx*s,
		kz*kx*v - ky*s, kz*ky*v + kx*s, c + kz*kz*v,
	}
}

// rodriguesFromMat converts a rotation matrix into its axis-angle (Rodrigues)
// vector.
func rodriguesFromMat(m mat3) (rx, ry, rz float64) {
	trace := m[0] + m[4] + m[8]
	cosT := (trace - 1) / 2
	cosT = math.Max(-1, math.Min(1, cosT))
	theta := math.Acos(cosT)
	if theta < 1e-9 {
		return 0.5 * (m[7] - m[5]), 0.5 * (m[2] - m[6]), 0.5 * (m[3] - m[1])
	}
	if math.Abs(math.Pi-theta) < 1e-6 {
		// Near-180° rotation: recover the axis from the diagonal.
		xx := (m[0] + 1) / 2
		yy := (m[4] + 1) / 2
		zz := (m[8] + 1) / 2
		ax := math.Sqrt(math.Max(0, xx))
		ay := math.Sqrt(math.Max(0, yy))
		az := math.Sqrt(math.Max(0, zz))
		if m[1]+m[3] < 0 {
			ay = -ay
		}
		if m[2]+m[6] < 0 {
			az = -az
		}
		return theta * ax, theta * ay, theta * az
	}
	k := theta / (2 * math.Sin(theta))
	return k * (m[7] - m[5]), k * (m[2] - m[6]), k * (m[3] - m[1])
}

// jacobiEigenSym3 computes the eigenvalues and eigenvectors of a symmetric 3×3
// matrix with the cyclic Jacobi method. The returned vecs holds one eigenvector
// per column (vecs[r*3+c] is component r of eigenvector c) and vals the matching
// eigenvalues. Eigenpairs are sorted by ascending eigenvalue.
func jacobiEigenSym3(sym mat3) (vals [3]float64, vecs mat3) {
	a := sym
	v := mat3Ident()
	for sweep := 0; sweep < 50; sweep++ {
		// Largest off-diagonal magnitude.
		off := math.Abs(a[1]) + math.Abs(a[2]) + math.Abs(a[5])
		if off < 1e-18 {
			break
		}
		for _, pq := range [3][2]int{{0, 1}, {0, 2}, {1, 2}} {
			p, q := pq[0], pq[1]
			apq := a[p*3+q]
			if math.Abs(apq) < 1e-20 {
				continue
			}
			app := a[p*3+p]
			aqq := a[q*3+q]
			phi := 0.5 * math.Atan2(2*apq, aqq-app)
			c := math.Cos(phi)
			s := math.Sin(phi)
			// Apply Jacobi rotation J on both sides: a = Jᵀ a J.
			rot := mat3Ident()
			rot[p*3+p] = c
			rot[q*3+q] = c
			rot[p*3+q] = s
			rot[q*3+p] = -s
			a = rot.transpose().mul(a).mul(rot)
			v = v.mul(rot)
		}
	}
	vals = [3]float64{a[0], a[4], a[8]}
	vecs = v
	// Sort ascending by eigenvalue, permuting the eigenvector columns to match.
	idx := [3]int{0, 1, 2}
	for i := 0; i < 3; i++ {
		for j := i + 1; j < 3; j++ {
			if vals[idx[j]] < vals[idx[i]] {
				idx[i], idx[j] = idx[j], idx[i]
			}
		}
	}
	var ov [3]float64
	var om mat3
	for c := 0; c < 3; c++ {
		ov[c] = vals[idx[c]]
		for r := 0; r < 3; r++ {
			om[r*3+c] = vecs[r*3+idx[c]]
		}
	}
	return ov, om
}
