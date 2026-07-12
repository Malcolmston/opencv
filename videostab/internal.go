package videostab

import (
	"math"

	"github.com/malcolmston/opencv/video"
)

// solveLinear solves the dense linear system a·x = b by Gauss-Jordan
// elimination with partial pivoting. a is an n×n matrix (modified in place) and
// b is the length-n right-hand side. The second result is false when the system
// is singular to within a small tolerance.
func solveLinear(a [][]float64, b []float64) ([]float64, bool) {
	n := len(a)
	m := make([][]float64, n)
	for i := 0; i < n; i++ {
		m[i] = make([]float64, n+1)
		copy(m[i], a[i])
		m[i][n] = b[i]
	}
	for col := 0; col < n; col++ {
		pivot := col
		best := math.Abs(m[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(m[r][col]); v > best {
				best = v
				pivot = r
			}
		}
		if best < 1e-12 {
			return nil, false
		}
		m[col], m[pivot] = m[pivot], m[col]
		pv := m[col][col]
		for j := col; j <= n; j++ {
			m[col][j] /= pv
		}
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			f := m[r][col]
			if f == 0 {
				continue
			}
			for j := col; j <= n; j++ {
				m[r][j] -= f * m[col][j]
			}
		}
	}
	x := make([]float64, n)
	for i := 0; i < n; i++ {
		x[i] = m[i][n]
	}
	return x, true
}

// fitMotion fits the given motion model to the correspondences from→to in the
// total least-squares sense, weighting each correspondence by weights[i] (pass
// nil for equal weights). It returns the fitted transform and false when there
// are too few points or the underlying linear system is singular.
func fitMotion(from, to []video.PointF, model MotionModel, weights []float64) (Motion, bool) {
	n := len(from)
	if n != len(to) || n < model.minPoints() {
		return Motion{}, false
	}
	w := func(i int) float64 {
		if weights == nil {
			return 1
		}
		return weights[i]
	}

	switch model {
	case MotionModelTranslation:
		var sx, sy, sw float64
		for i := 0; i < n; i++ {
			wi := w(i)
			sx += wi * (to[i].X - from[i].X)
			sy += wi * (to[i].Y - from[i].Y)
			sw += wi
		}
		if sw == 0 {
			return Motion{}, false
		}
		return TranslationMotion(sx/sw, sy/sw), true

	case MotionModelRotation:
		var num, den float64
		for i := 0; i < n; i++ {
			wi := w(i)
			num += wi * (from[i].X*to[i].Y - from[i].Y*to[i].X)
			den += wi * (from[i].X*to[i].X + from[i].Y*to[i].Y)
		}
		theta := math.Atan2(num, den)
		return SimilarityMotion(1, theta, 0, 0), true

	case MotionModelTranslationAndScale:
		// Unknowns u = [s, tx, ty]; rows [fx,1,0]->tox and [fy,0,1]->toy.
		nm := [][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}
		rhs := []float64{0, 0, 0}
		acc := func(a [3]float64, t, wi float64) {
			for r := 0; r < 3; r++ {
				for c := 0; c < 3; c++ {
					nm[r][c] += wi * a[r] * a[c]
				}
				rhs[r] += wi * a[r] * t
			}
		}
		for i := 0; i < n; i++ {
			wi := w(i)
			acc([3]float64{from[i].X, 1, 0}, to[i].X, wi)
			acc([3]float64{from[i].Y, 0, 1}, to[i].Y, wi)
		}
		sol, ok := solveLinear(nm, rhs)
		if !ok {
			return Motion{}, false
		}
		s := sol[0]
		return Motion{s, 0, sol[1], 0, s, sol[2], 0, 0, 1}, true

	case MotionModelRigid:
		return fitRigid(from, to, w), true

	case MotionModelSimilarity:
		tf, ok := video.EstimateAffinePartial2D(from, to)
		if !ok {
			return Motion{}, false
		}
		return SimilarityMotion(tf.Scale, tf.Angle, tf.Tx, tf.Ty), true

	case MotionModelAffine:
		// Two independent 3-parameter systems share the basis [fx, fy, 1].
		nm := [][]float64{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}
		rx := []float64{0, 0, 0}
		ry := []float64{0, 0, 0}
		for i := 0; i < n; i++ {
			wi := w(i)
			a := [3]float64{from[i].X, from[i].Y, 1}
			for r := 0; r < 3; r++ {
				for c := 0; c < 3; c++ {
					nm[r][c] += wi * a[r] * a[c]
				}
				rx[r] += wi * a[r] * to[i].X
				ry[r] += wi * a[r] * to[i].Y
			}
		}
		row1, ok1 := solveLinear(cloneRows(nm), rx)
		row2, ok2 := solveLinear(cloneRows(nm), ry)
		if !ok1 || !ok2 {
			return Motion{}, false
		}
		return Motion{row1[0], row1[1], row1[2], row2[0], row2[1], row2[2], 0, 0, 1}, true

	case MotionModelHomography:
		return fitHomography(from, to, w)

	default:
		return Motion{}, false
	}
}

// fitRigid recovers a Euclidean transform (unit scale) by aligning centroids and
// solving for the optimal in-plane rotation.
func fitRigid(from, to []video.PointF, w func(int) float64) Motion {
	var cfx, cfy, ctx, cty, sw float64
	for i := range from {
		wi := w(i)
		cfx += wi * from[i].X
		cfy += wi * from[i].Y
		ctx += wi * to[i].X
		cty += wi * to[i].Y
		sw += wi
	}
	if sw == 0 {
		return IdentityMotion()
	}
	cfx, cfy, ctx, cty = cfx/sw, cfy/sw, ctx/sw, cty/sw
	var num, den float64
	for i := range from {
		wi := w(i)
		ax, ay := from[i].X-cfx, from[i].Y-cfy
		bx, by := to[i].X-ctx, to[i].Y-cty
		num += wi * (ax*by - ay*bx)
		den += wi * (ax*bx + ay*by)
	}
	theta := math.Atan2(num, den)
	c, s := math.Cos(theta), math.Sin(theta)
	tx := ctx - (c*cfx - s*cfy)
	ty := cty - (s*cfx + c*cfy)
	return Motion{c, -s, tx, s, c, ty, 0, 0, 1}
}

// fitHomography fits a projective transform by fixing the bottom-right element
// to 1 and solving the resulting 8-parameter linear least-squares problem.
func fitHomography(from, to []video.PointF, w func(int) float64) (Motion, bool) {
	nm := make([][]float64, 8)
	for i := range nm {
		nm[i] = make([]float64, 8)
	}
	rhs := make([]float64, 8)
	acc := func(a []float64, t, wi float64) {
		for r := 0; r < 8; r++ {
			for c := 0; c < 8; c++ {
				nm[r][c] += wi * a[r] * a[c]
			}
			rhs[r] += wi * a[r] * t
		}
	}
	for i := range from {
		wi := w(i)
		fx, fy := from[i].X, from[i].Y
		tx, ty := to[i].X, to[i].Y
		acc([]float64{fx, fy, 1, 0, 0, 0, -fx * tx, -fy * tx}, tx, wi)
		acc([]float64{0, 0, 0, fx, fy, 1, -fx * ty, -fy * ty}, ty, wi)
	}
	sol, ok := solveLinear(nm, rhs)
	if !ok {
		return Motion{}, false
	}
	return Motion{sol[0], sol[1], sol[2], sol[3], sol[4], sol[5], sol[6], sol[7], 1}, true
}

// cloneRows returns a deep copy of a square matrix so a solver may consume it.
func cloneRows(a [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range a {
		out[i] = make([]float64, len(a[i]))
		copy(out[i], a[i])
	}
	return out
}

// residual returns the Euclidean reprojection error of correspondence i under m.
func residual(m Motion, a, b video.PointF) float64 {
	x, y := m.Apply(a.X, a.Y)
	return math.Hypot(x-b.X, y-b.Y)
}
