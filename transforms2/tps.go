package transforms2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ThinPlateSpline is a smooth non-rigid coordinate mapping fitted from a set of
// control-point correspondences. It interpolates a pair of thin-plate-spline
// functions (one per output coordinate) that pass through the correspondences
// while minimising bending energy. Construct one with [NewThinPlateSpline].
type ThinPlateSpline struct {
	centers []cv.Point2f // the "from" control points
	// wx, wy hold the non-affine weights (len == len(centers)); ax, ay hold the
	// affine part [a0, a1, a2].
	wx, wy []float64
	ax, ay [3]float64
	lambda float64
}

// transforms2tpsU is the thin-plate-spline radial basis U(r) = r^2 * log(r^2),
// with U(0) = 0.
func transforms2tpsU(r2 float64) float64 {
	if r2 <= 0 {
		return 0
	}
	return r2 * math.Log(r2)
}

// NewThinPlateSpline fits a thin-plate spline that maps the points in from to
// the corresponding points in to. The regularization parameter lambda (>= 0)
// relaxes exact interpolation for smoother mappings; use 0 for exact
// interpolation. from and to must have the same length (>= 3) and from must not
// be entirely collinear. It panics if these preconditions are violated or the
// system is singular.
func NewThinPlateSpline(from, to []cv.Point2f, lambda float64) *ThinPlateSpline {
	n := len(from)
	if n != len(to) {
		panic("transforms2: NewThinPlateSpline point count mismatch")
	}
	if n < 3 {
		panic("transforms2: NewThinPlateSpline needs at least three control points")
	}
	if lambda < 0 {
		panic("transforms2: NewThinPlateSpline lambda must be non-negative")
	}
	// Assemble the (n+3) x (n+3) system L = [[K, P], [P^T, 0]].
	size := n + 3
	build := func() [][]float64 {
		a := make([][]float64, size)
		for i := range a {
			a[i] = make([]float64, size)
		}
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				dx := from[i].X - from[j].X
				dy := from[i].Y - from[j].Y
				a[i][j] = transforms2tpsU(dx*dx + dy*dy)
			}
			a[i][i] += lambda
			a[i][n] = 1
			a[i][n+1] = from[i].X
			a[i][n+2] = from[i].Y
			a[n][i] = 1
			a[n+1][i] = from[i].X
			a[n+2][i] = from[i].Y
		}
		return a
	}
	bx := make([]float64, size)
	by := make([]float64, size)
	for i := 0; i < n; i++ {
		bx[i] = to[i].X
		by[i] = to[i].Y
	}
	sx, okx := transforms2solve(build(), bx)
	sy, oky := transforms2solve(build(), by)
	if !okx || !oky {
		panic("transforms2: NewThinPlateSpline control points are degenerate")
	}
	tps := &ThinPlateSpline{
		centers: append([]cv.Point2f(nil), from...),
		wx:      append([]float64(nil), sx[:n]...),
		wy:      append([]float64(nil), sy[:n]...),
		ax:      [3]float64{sx[n], sx[n+1], sx[n+2]},
		ay:      [3]float64{sy[n], sy[n+1], sy[n+2]},
		lambda:  lambda,
	}
	return tps
}

// Transform maps a single point (x, y) through the spline, returning its image
// in the target coordinate system.
func (t *ThinPlateSpline) Transform(x, y float64) (float64, float64) {
	fx := t.ax[0] + t.ax[1]*x + t.ax[2]*y
	fy := t.ay[0] + t.ay[1]*x + t.ay[2]*y
	for i, c := range t.centers {
		dx := x - c.X
		dy := y - c.Y
		u := transforms2tpsU(dx*dx + dy*dy)
		fx += t.wx[i] * u
		fy += t.wy[i] * u
	}
	return fx, fy
}

// Warp renders src deformed by the spline into a width x height output. Because
// warping needs the inverse mapping (destination to source), the spline passed
// to Warp must map destination coordinates to source coordinates; construct it
// with NewThinPlateSpline(dstControlPoints, srcControlPoints, lambda). Pixels
// are resampled with the chosen interpolation and border handling.
func (t *ThinPlateSpline) Warp(src *cv.Mat, width, height int, interp Interpolation, border BorderMode, fill float64) *cv.Mat {
	return transforms2warpInverse(src, width, height, interp, border, fill, t.Transform)
}
