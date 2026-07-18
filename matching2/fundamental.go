package matching2

import (
	"math"

	"github.com/malcolmston/opencv/core"
)

// FindFundamentalMat estimates the 3×3 fundamental matrix F relating two views
// from at least eight point correspondences, using Hartley's normalized
// eight-point algorithm. F satisfies (pts2ᵢ,1)ᵀ · F · (pts1ᵢ,1) = 0 for every
// correspondence. The rank-2 constraint is enforced by nulling the smallest
// singular value, and F is scaled so its Frobenius norm is 1. It reports false
// when there are too few points or the configuration is degenerate.
func FindFundamentalMat(pts1, pts2 []core.Point2d) ([3][3]float64, bool) {
	if len(pts1) != len(pts2) || len(pts1) < 8 {
		return [3][3]float64{}, false
	}
	n1, T1 := NormalizePoints2D(pts1)
	n2, T2 := NormalizePoints2D(pts2)
	a := make([][]float64, len(pts1))
	for i := range n1 {
		x, y := n1[i].X, n1[i].Y
		xp, yp := n2[i].X, n2[i].Y
		a[i] = []float64{xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1}
	}
	f := matching2nullVector(a)
	Fn := [3][3]float64{
		{f[0], f[1], f[2]},
		{f[3], f[4], f[5]},
		{f[6], f[7], f[8]},
	}
	Fn = enforceRank2(Fn)
	// Denormalize: F = T2ᵀ · Fn · T1.
	F := Mat3Mul(Mat3Transpose(T2), Mat3Mul(Fn, T1))
	F = normalizeFrobenius(F)
	if !matching2finite9(F) {
		return [3][3]float64{}, false
	}
	return F, true
}

// FindFundamentalMatRANSAC robustly estimates the fundamental matrix from
// correspondences containing outliers. A correspondence is an inlier when its
// Sampson distance is at most threshold pixels. iters bounds the number of
// random eight-point samples; seed makes the result deterministic. The final
// model is refit over all inliers. It reports Ok false when no sample yields at
// least eight inliers.
func FindFundamentalMatRANSAC(pts1, pts2 []core.Point2d, threshold float64, iters int, seed int64) RANSACResult[[3][3]float64] {
	var empty RANSACResult[[3][3]float64]
	if len(pts1) != len(pts2) || len(pts1) < 8 {
		return empty
	}
	fit := func(sample []int) ([3][3]float64, bool) {
		p1 := make([]core.Point2d, len(sample))
		p2 := make([]core.Point2d, len(sample))
		for i, idx := range sample {
			p1[i] = pts1[idx]
			p2[i] = pts2[idx]
		}
		return FindFundamentalMat(p1, p2)
	}
	inliers := func(F [3][3]float64) []bool {
		mask := make([]bool, len(pts1))
		for i := range pts1 {
			mask[i] = SampsonDistance(F, pts1[i], pts2[i]) <= threshold
		}
		return mask
	}
	return RANSAC(len(pts1), 8, iters, 8, seed, fit, inliers, fit)
}

// SampsonDistance returns the first-order geometric (Sampson) approximation of
// the reprojection error for a correspondence under the fundamental matrix F. It
// is the standard residual for robust fundamental-matrix estimation.
func SampsonDistance(F [3][3]float64, p1, p2 core.Point2d) float64 {
	x1 := [3]float64{p1.X, p1.Y, 1}
	x2 := [3]float64{p2.X, p2.Y, 1}
	Fx1 := Mat3VecMul(F, x1)
	Ftx2 := Mat3VecMul(Mat3Transpose(F), x2)
	num := x2[0]*Fx1[0] + x2[1]*Fx1[1] + x2[2]*Fx1[2]
	denom := Fx1[0]*Fx1[0] + Fx1[1]*Fx1[1] + Ftx2[0]*Ftx2[0] + Ftx2[1]*Ftx2[1]
	if denom < 1e-300 {
		return 0
	}
	return math.Abs(num) / math.Sqrt(denom)
}

// EpipolarConstraint returns the algebraic residual x2ᵀ·F·x1 for a
// correspondence, which is exactly zero for a noise-free inlier. Unlike
// [SampsonDistance] it is not in pixel units.
func EpipolarConstraint(F [3][3]float64, p1, p2 core.Point2d) float64 {
	x1 := [3]float64{p1.X, p1.Y, 1}
	Fx1 := Mat3VecMul(F, x1)
	return p2.X*Fx1[0] + p2.Y*Fx1[1] + Fx1[2]
}

// EpipolarLine returns the epipolar line induced in the other image by a point.
// The line is returned as (a, b, c) for a·x + b·y + c = 0 with (a, b)
// normalized to unit length. When whichImage is 1, p lies in image 1 and the
// line l = F·(p,1) is the corresponding line in image 2; when whichImage is 2,
// p lies in image 2 and the line l = Fᵀ·(p,1) is in image 1.
func EpipolarLine(F [3][3]float64, p core.Point2d, whichImage int) [3]float64 {
	x := [3]float64{p.X, p.Y, 1}
	var l [3]float64
	if whichImage == 2 {
		l = Mat3VecMul(Mat3Transpose(F), x)
	} else {
		l = Mat3VecMul(F, x)
	}
	nrm := math.Hypot(l[0], l[1])
	if nrm > 1e-300 {
		l[0] /= nrm
		l[1] /= nrm
		l[2] /= nrm
	}
	return l
}

// PointLineDistance returns the perpendicular distance from point p to the line
// (a, b, c) representing a·x + b·y + c = 0. When (a, b) is a unit normal (as
// produced by [EpipolarLine]) the result is in pixels.
func PointLineDistance(line [3]float64, p core.Point2d) float64 {
	nrm := math.Hypot(line[0], line[1])
	if nrm < 1e-300 {
		return 0
	}
	return math.Abs(line[0]*p.X+line[1]*p.Y+line[2]) / nrm
}

// Epipoles returns the epipoles e1 (in image 1) and e2 (in image 2) of the
// fundamental matrix F. e1 is the right null vector of F and e2 is the left null
// vector, each dehomogenized. It reports false when an epipole lies at infinity
// (its homogeneous third coordinate is ~0).
func Epipoles(F [3][3]float64) (e1, e2 core.Point2d, ok bool) {
	// e1: F·e1 = 0 → right null vector of F.
	rows := [][]float64{
		{F[0][0], F[0][1], F[0][2]},
		{F[1][0], F[1][1], F[1][2]},
		{F[2][0], F[2][1], F[2][2]},
	}
	v1 := matching2nullVector(rows)
	// e2: Fᵀ·e2 = 0 → right null vector of Fᵀ.
	Ft := Mat3Transpose(F)
	rowsT := [][]float64{
		{Ft[0][0], Ft[0][1], Ft[0][2]},
		{Ft[1][0], Ft[1][1], Ft[1][2]},
		{Ft[2][0], Ft[2][1], Ft[2][2]},
	}
	v2 := matching2nullVector(rowsT)
	if math.Abs(v1[2]) < 1e-12 || math.Abs(v2[2]) < 1e-12 {
		return core.Point2d{}, core.Point2d{}, false
	}
	e1 = core.Point2d{X: v1[0] / v1[2], Y: v1[1] / v1[2]}
	e2 = core.Point2d{X: v2[0] / v2[2], Y: v2[1] / v2[2]}
	return e1, e2, true
}

// enforceRank2 projects a 3×3 matrix to the nearest rank-2 matrix (Frobenius
// norm) by zeroing its smallest singular value, the standard correction applied
// to a fundamental matrix.
func enforceRank2(F [3][3]float64) [3][3]float64 {
	u, s, v := matching2svd3(F)
	s[2] = 0
	d := [3][3]float64{{s[0], 0, 0}, {0, s[1], 0}, {0, 0, 0}}
	return Mat3Mul(u, Mat3Mul(d, Mat3Transpose(v)))
}

// normalizeFrobenius scales a matrix so that its Frobenius norm is 1; a zero
// matrix is returned unchanged.
func normalizeFrobenius(m [3][3]float64) [3][3]float64 {
	var s float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			s += m[i][j] * m[i][j]
		}
	}
	if s < 1e-300 {
		return m
	}
	return Mat3Scale(m, 1/math.Sqrt(s))
}
