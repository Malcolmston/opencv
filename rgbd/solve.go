package rgbd

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// This file holds the small dense-solver and image-sampling kernels shared by
// the odometry, warping and volumetric routines. They are unexported and, like
// the rest of the package, depend only on the Go standard library and the root
// cv package.

// solve6 solves the 6×6 linear system A·x = b by Gaussian elimination with
// partial pivoting. It reports ok=false when the system is singular (a pivot
// collapses to zero), leaving the caller to fall back to a zero update. The
// 6-vector unknown is the twist ξ = (ω, ν) used by the Gauss–Newton odometry
// steps: a rotation increment ω and a translation increment ν.
func solve6(a [6][6]float64, b [6]float64) (x [6]float64, ok bool) {
	// Work on augmented copies so the caller's inputs are untouched.
	var m [6][7]float64
	for i := 0; i < 6; i++ {
		for j := 0; j < 6; j++ {
			m[i][j] = a[i][j]
		}
		m[i][6] = b[i]
	}
	for col := 0; col < 6; col++ {
		// Partial pivot: pick the row with the largest magnitude in this column.
		piv := col
		best := math.Abs(m[col][col])
		for r := col + 1; r < 6; r++ {
			if v := math.Abs(m[r][col]); v > best {
				best = v
				piv = r
			}
		}
		if best < 1e-15 {
			return x, false
		}
		m[col], m[piv] = m[piv], m[col]
		// Eliminate below.
		for r := col + 1; r < 6; r++ {
			f := m[r][col] / m[col][col]
			if f == 0 {
				continue
			}
			for c := col; c < 7; c++ {
				m[r][c] -= f * m[col][c]
			}
		}
	}
	// Back-substitution.
	for i := 5; i >= 0; i-- {
		s := m[i][6]
		for j := i + 1; j < 6; j++ {
			s -= m[i][j] * x[j]
		}
		x[i] = s / m[i][i]
	}
	return x, true
}

// solveGaussNewton damps the normal equations with a small Levenberg–Marquardt
// term on the diagonal before solving them with [solve6]. The damping keeps the
// system invertible when the scene under-constrains some degrees of freedom (a
// single plane, say, leaves three unobservable): the observable directions are
// solved essentially unchanged while the unobservable ones receive a negligible
// update instead of aborting the whole step.
func solveGaussNewton(a [6][6]float64, b [6]float64) ([6]float64, bool) {
	for i := 0; i < 6; i++ {
		a[i][i] += 1e-9*a[i][i] + 1e-12
	}
	return solve6(a, b)
}

// accumulateNormal folds one linearised residual into the normal equations
// A·ξ = b of a Gauss–Newton step: for a residual r with Jacobian row j and
// weight w it adds w·jⱼ·jₖ to A and −w·jᵢ·r to b, so that solving A·ξ = b
// minimises Σ w·(r + j·ξ)².
func accumulateNormal(a *[6][6]float64, b *[6]float64, j [6]float64, r, w float64) {
	for p := 0; p < 6; p++ {
		wj := w * j[p]
		for q := 0; q < 6; q++ {
			a[p][q] += wj * j[q]
		}
		b[p] -= wj * r
	}
}

// inBounds reports whether the continuous image coordinate (x, y) lies inside a
// rows×cols grid with a two-pixel border to spare, which keeps both the bilinear
// taps and the central-difference gradient (a bilinear sample one pixel further
// out) in range.
func inBounds(x, y float64, rows, cols int) bool {
	return x >= 2 && y >= 2 && x <= float64(cols-3) && y <= float64(rows-3)
}

// sampleBilinear returns the bilinearly interpolated value of m at the
// continuous coordinate (x, y). The caller is responsible for ensuring the
// point is in range (see [inBounds]).
func sampleBilinear(m *cv.FloatMat, x, y float64) float64 {
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	fx := x - float64(x0)
	fy := y - float64(y0)
	v00 := m.Data[y0*m.Cols+x0]
	v01 := m.Data[y0*m.Cols+x0+1]
	v10 := m.Data[(y0+1)*m.Cols+x0]
	v11 := m.Data[(y0+1)*m.Cols+x0+1]
	a := v00*(1-fx) + v01*fx
	b := v10*(1-fx) + v11*fx
	return a*(1-fy) + b*fy
}

// gradientAt returns the horizontal and vertical image gradients of m at the
// continuous coordinate (x, y), estimated by central differences of the
// bilinear sampler. The point must satisfy [inBounds] with a further one-pixel
// margin, which the odometry loops guarantee.
func gradientAt(m *cv.FloatMat, x, y float64) (gx, gy float64) {
	gx = 0.5 * (sampleBilinear(m, x+1, y) - sampleBilinear(m, x-1, y))
	gy = 0.5 * (sampleBilinear(m, x, y+1) - sampleBilinear(m, x, y-1))
	return gx, gy
}
