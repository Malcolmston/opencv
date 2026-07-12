package rapid

import "math"

// This file holds the self-contained linear-algebra kernels the tracker relies
// on: Rodrigues conversions in both directions, small 3×3 helpers, and a
// symmetric linear solver for the 6×6 Gauss-Newton normal equations. Nothing
// here is exported; the helpers exist purely to support the public API.

// clampf clamps v to the closed interval [lo, hi].
func clampf(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// rodrigues converts an axis-angle rotation vector into a 3×3 rotation matrix
// using Rodrigues' formula.
func rodrigues(rvec [3]float64) [3][3]float64 {
	theta := math.Sqrt(rvec[0]*rvec[0] + rvec[1]*rvec[1] + rvec[2]*rvec[2])
	if theta < 1e-12 {
		return [3][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	}
	kx, ky, kz := rvec[0]/theta, rvec[1]/theta, rvec[2]/theta
	c, s := math.Cos(theta), math.Sin(theta)
	v := 1 - c
	return [3][3]float64{
		{c + kx*kx*v, kx*ky*v - kz*s, kx*kz*v + ky*s},
		{ky*kx*v + kz*s, c + ky*ky*v, ky*kz*v - kx*s},
		{kz*kx*v - ky*s, kz*ky*v + kx*s, c + kz*kz*v},
	}
}

// rotationToRvec converts a 3×3 rotation matrix back into an axis-angle
// rotation vector, robustly handling the near-zero and near-π cases.
func rotationToRvec(r [3][3]float64) [3]float64 {
	rx := r[2][1] - r[1][2]
	ry := r[0][2] - r[2][0]
	rz := r[1][0] - r[0][1]
	s := math.Sqrt(rx*rx+ry*ry+rz*rz) * 0.5
	c := clampf((r[0][0]+r[1][1]+r[2][2]-1)*0.5, -1, 1)
	theta := math.Acos(c)
	if theta < 1e-9 {
		return [3]float64{0, 0, 0}
	}
	if s < 1e-9 {
		// theta is close to pi: R = 2 k k^T - I, so the diagonal gives |k_i|.
		xx := (r[0][0] + 1) * 0.5
		yy := (r[1][1] + 1) * 0.5
		zz := (r[2][2] + 1) * 0.5
		xy := (r[0][1] + r[1][0]) * 0.25
		xz := (r[0][2] + r[2][0]) * 0.25
		yz := (r[1][2] + r[2][1]) * 0.25
		var kx, ky, kz float64
		switch {
		case xx >= yy && xx >= zz:
			kx = math.Sqrt(math.Max(xx, 0))
			ky = xy / kx
			kz = xz / kx
		case yy >= zz:
			ky = math.Sqrt(math.Max(yy, 0))
			kx = xy / ky
			kz = yz / ky
		default:
			kz = math.Sqrt(math.Max(zz, 0))
			kx = xz / kz
			ky = yz / kz
		}
		n := math.Sqrt(kx*kx + ky*ky + kz*kz)
		if n < 1e-12 {
			return [3]float64{0, 0, 0}
		}
		return [3]float64{theta * kx / n, theta * ky / n, theta * kz / n}
	}
	mul := theta / (2 * s)
	return [3]float64{rx * mul, ry * mul, rz * mul}
}

// mul3 returns the matrix product a·b of two 3×3 matrices.
func mul3(a, b [3][3]float64) [3][3]float64 {
	var r [3][3]float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			r[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j]
		}
	}
	return r
}

// solveSPD solves the n×n symmetric linear system a·x = b in place using
// Gauss-Jordan elimination with partial pivoting. It reports ok == false if the
// system is singular. a and b are modified.
func solveSPD(a [][]float64, b []float64, n int) (x []float64, ok bool) {
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
		if best < 1e-15 {
			return nil, false
		}
		if pivot != col {
			a[col], a[pivot] = a[pivot], a[col]
			b[col], b[pivot] = b[pivot], b[col]
		}
		inv := 1 / a[col][col]
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := a[r][col] * inv
			if f == 0 {
				continue
			}
			for c := col; c < n; c++ {
				a[r][c] -= f * a[col][c]
			}
			b[r] -= f * b[col]
		}
	}
	x = make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = b[i] / a[i][i]
	}
	return x, true
}
