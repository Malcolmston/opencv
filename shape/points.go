package shape

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Point2D is a two-dimensional point with floating-point coordinates. Several
// shape routines in this package — the shape transformers, the shape-context and
// Hausdorff distance extractors, and [RotatedRectangleIntersection] — operate on
// or return sub-pixel positions, for which the integer [cv.Point] is too coarse.
// The X/Y convention matches the parent package: X is the column and Y the row.
type Point2D struct {
	X float64
	Y float64
}

// FloatPoints converts a slice of integer [cv.Point] to [Point2D] values.
func FloatPoints(pts []cv.Point) []Point2D {
	out := make([]Point2D, len(pts))
	for i, p := range pts {
		out[i] = Point2D{X: float64(p.X), Y: float64(p.Y)}
	}
	return out
}

// RoundPoints converts a slice of [Point2D] to integer [cv.Point] values,
// rounding each coordinate to the nearest pixel.
func RoundPoints(pts []Point2D) []cv.Point {
	out := make([]cv.Point, len(pts))
	for i, p := range pts {
		out[i] = cv.Point{X: int(math.Round(p.X)), Y: int(math.Round(p.Y))}
	}
	return out
}

// dist2D returns the Euclidean distance between two [Point2D] values.
func dist2D(a, b Point2D) float64 {
	return math.Hypot(a.X-b.X, a.Y-b.Y)
}

// solveLinearSystem solves the dense linear system a·x = b for x using Gaussian
// elimination with partial pivoting. a is n×n (row-major) and b is n×k (so the
// system is solved for k right-hand sides at once). The inputs are copied and
// left unmodified. It reports false when a is singular to working precision.
func solveLinearSystem(a [][]float64, b [][]float64) ([][]float64, bool) {
	n := len(a)
	if n == 0 {
		return nil, false
	}
	k := len(b[0])
	// Work on copies so callers keep their matrices.
	m := make([][]float64, n)
	rhs := make([][]float64, n)
	for i := 0; i < n; i++ {
		m[i] = make([]float64, n)
		copy(m[i], a[i])
		rhs[i] = make([]float64, k)
		copy(rhs[i], b[i])
	}
	for col := 0; col < n; col++ {
		// Partial pivot: largest magnitude in the column.
		pivot := col
		best := math.Abs(m[col][col])
		for r := col + 1; r < n; r++ {
			if v := math.Abs(m[r][col]); v > best {
				best = v
				pivot = r
			}
		}
		if best < 1e-15 {
			return nil, false
		}
		if pivot != col {
			m[col], m[pivot] = m[pivot], m[col]
			rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
		}
		inv := 1 / m[col][col]
		for r := 0; r < n; r++ {
			if r == col {
				continue
			}
			factor := m[r][col] * inv
			if factor == 0 {
				continue
			}
			for c := col; c < n; c++ {
				m[r][c] -= factor * m[col][c]
			}
			for c := 0; c < k; c++ {
				rhs[r][c] -= factor * rhs[col][c]
			}
		}
	}
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, k)
		d := m[i][i]
		for c := 0; c < k; c++ {
			out[i][c] = rhs[i][c] / d
		}
	}
	return out, true
}
