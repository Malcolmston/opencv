package imgprocx

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// Point2f is a sub-pixel image coordinate (x is the column, y is the row). It is
// the floating-point counterpart of [cv.Point] and is used to report refined or
// estimated positions.
type Point2f struct {
	X float64
	Y float64
}

// ransacSeed fixes the pseudo-random sampling used by the RANSAC estimators so
// their output is deterministic across runs.
const ransacSeed = 0x1c7a55e

// defaultReprojThreshold is the inlier distance (in pixels) used by
// [EstimateAffine2D] and [EstimateAffinePartial2D] when classifying a
// correspondence as an inlier: a point pair is an inlier when the model maps the
// source to within this distance of the destination.
const defaultReprojThreshold = 3.0

// defaultRANSACIters is the number of random minimal-sample trials the RANSAC
// estimators perform.
const defaultRANSACIters = 2000

// GetAffineTransform computes the 2×3 affine transform that maps the three
// source points src to the three destination points dst exactly, mirroring
// cv2.getAffineTransform. It solves the two 3×3 linear systems (one for the
// output x-coordinate, one for y) that share the coefficient matrix built from
// the source points. The three source points must not be collinear; it panics if
// they are (the system is singular).
//
// The result is a [cv.AffineMatrix] and can be passed directly to
// [cv.WarpAffine].
func GetAffineTransform(src, dst [3]cv.Point) cv.AffineMatrix {
	a := [3][3]float64{
		{float64(src[0].X), float64(src[0].Y), 1},
		{float64(src[1].X), float64(src[1].Y), 1},
		{float64(src[2].X), float64(src[2].Y), 1},
	}
	bx := [3]float64{float64(dst[0].X), float64(dst[1].X), float64(dst[2].X)}
	by := [3]float64{float64(dst[0].Y), float64(dst[1].Y), float64(dst[2].Y)}
	mx, ok1 := solve3(a, bx)
	my, ok2 := solve3(a, by)
	if !ok1 || !ok2 {
		panic("imgprocx: GetAffineTransform source points are collinear")
	}
	return cv.AffineMatrix{mx[0], mx[1], mx[2], my[0], my[1], my[2]}
}

// ApplyAffine maps the point p through the 2×3 affine matrix m (stored
// row-major, so m[0] holds the coefficients of the output x-coordinate) and
// returns the sub-pixel result.
func ApplyAffine(m [2][3]float64, p cv.Point) Point2f {
	x := float64(p.X)
	y := float64(p.Y)
	return Point2f{
		X: m[0][0]*x + m[0][1]*y + m[0][2],
		Y: m[1][0]*x + m[1][1]*y + m[1][2],
	}
}

// ToAffineMatrix converts a 2×3 matrix into the root package's [cv.AffineMatrix]
// (a flat [6]float64) so it can be used with [cv.WarpAffine].
func ToAffineMatrix(m [2][3]float64) cv.AffineMatrix {
	return cv.AffineMatrix{m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2]}
}

// FromAffineMatrix converts a [cv.AffineMatrix] into the 2×3 form used by the
// estimators in this package.
func FromAffineMatrix(a cv.AffineMatrix) [2][3]float64 {
	return [2][3]float64{
		{a[0], a[1], a[2]},
		{a[3], a[4], a[5]},
	}
}

// EstimateAffine2D robustly fits the 2×3 affine transform (six degrees of
// freedom) mapping src to dst from a set of noisy, possibly outlier-contaminated
// point correspondences. src and dst must have equal length of at least three.
//
// It runs a deterministic RANSAC loop — repeatedly fitting an exact transform to
// three random correspondences and counting inliers within
// defaultReprojThreshold pixels — then refits by least squares over the largest
// inlier set. The returned inliers slice is parallel to the inputs and flags the
// correspondences consistent with the recovered model. If no non-degenerate
// model is found (for example fewer than three non-collinear points) the second
// return value is nil and m is the least-squares fit over all points.
func EstimateAffine2D(src, dst []cv.Point) (m [2][3]float64, inliers []bool) {
	if len(src) != len(dst) {
		panic("imgprocx: EstimateAffine2D src and dst length mismatch")
	}
	n := len(src)
	if n < 3 {
		panic("imgprocx: EstimateAffine2D needs at least 3 correspondences")
	}
	best, ok := ransacAffine(src, dst, 3, fitAffineMinimal, fitAffineLS)
	if !ok {
		fit, _ := fitAffineLS(src, dst, allIndices(n))
		return fit, nil
	}
	return best.model, best.inliers
}

// EstimateAffinePartial2D robustly fits a partial (similarity) 2×3 transform
// with four degrees of freedom — a rotation, a single uniform scale and a
// translation — mapping src to dst. The recovered matrix has the constrained
// form
//
//	[ a -b tx ]
//	[ b  a ty ]
//
// so it never shears or scales the axes independently. src and dst must have
// equal length of at least two. Like [EstimateAffine2D] it uses a deterministic
// RANSAC loop (with two-point minimal samples) followed by a least-squares refit
// over the inliers, and returns a parallel inlier mask (nil when no model is
// found).
func EstimateAffinePartial2D(src, dst []cv.Point) (m [2][3]float64, inliers []bool) {
	if len(src) != len(dst) {
		panic("imgprocx: EstimateAffinePartial2D src and dst length mismatch")
	}
	n := len(src)
	if n < 2 {
		panic("imgprocx: EstimateAffinePartial2D needs at least 2 correspondences")
	}
	best, ok := ransacAffine(src, dst, 2, fitPartialMinimal, fitPartialLS)
	if !ok {
		fit, _ := fitPartialLS(src, dst, allIndices(n))
		return fit, nil
	}
	return best.model, best.inliers
}

// affineFit bundles a candidate model with its inlier mask and score.
type affineFit struct {
	model   [2][3]float64
	inliers []bool
	count   int
}

// fitFunc fits a model to the correspondences indexed by idx, reporting success.
type fitFunc func(src, dst []cv.Point, idx []int) ([2][3]float64, bool)

// ransacAffine implements the shared RANSAC loop for the affine estimators.
// sample is the minimal sample size, minimal fits an exact model to a minimal
// sample and refine performs the final least-squares refit over the inliers.
func ransacAffine(src, dst []cv.Point, sample int, minimal, refine fitFunc) (affineFit, bool) {
	n := len(src)
	rng := rand.New(rand.NewSource(ransacSeed))
	thr2 := defaultReprojThreshold * defaultReprojThreshold
	var best affineFit
	found := false
	pick := make([]int, sample)
	for iter := 0; iter < defaultRANSACIters; iter++ {
		chooseDistinct(rng, n, pick)
		model, ok := minimal(src, dst, pick)
		if !ok {
			continue
		}
		mask, count := scoreModel(src, dst, model, thr2)
		if !found || count > best.count {
			best = affineFit{model: model, inliers: mask, count: count}
			found = true
		}
	}
	if !found || best.count < sample {
		return affineFit{}, false
	}
	// Refit over the inliers for a lower-variance estimate, then rescore.
	idx := maskToIndices(best.inliers)
	if refined, ok := refine(src, dst, idx); ok {
		mask, count := scoreModel(src, dst, refined, thr2)
		if count >= best.count {
			best = affineFit{model: refined, inliers: mask, count: count}
		}
	}
	return best, true
}

// scoreModel returns the inlier mask and inlier count of model against every
// correspondence, using the squared distance threshold thr2.
func scoreModel(src, dst []cv.Point, model [2][3]float64, thr2 float64) ([]bool, int) {
	mask := make([]bool, len(src))
	count := 0
	for i := range src {
		p := ApplyAffine(model, src[i])
		dx := p.X - float64(dst[i].X)
		dy := p.Y - float64(dst[i].Y)
		if dx*dx+dy*dy <= thr2 {
			mask[i] = true
			count++
		}
	}
	return mask, count
}

// fitAffineMinimal fits an exact 6-DOF affine to three correspondences.
func fitAffineMinimal(src, dst []cv.Point, idx []int) ([2][3]float64, bool) {
	a := [3][3]float64{
		{float64(src[idx[0]].X), float64(src[idx[0]].Y), 1},
		{float64(src[idx[1]].X), float64(src[idx[1]].Y), 1},
		{float64(src[idx[2]].X), float64(src[idx[2]].Y), 1},
	}
	bx := [3]float64{float64(dst[idx[0]].X), float64(dst[idx[1]].X), float64(dst[idx[2]].X)}
	by := [3]float64{float64(dst[idx[0]].Y), float64(dst[idx[1]].Y), float64(dst[idx[2]].Y)}
	mx, ok1 := solve3(a, bx)
	my, ok2 := solve3(a, by)
	if !ok1 || !ok2 {
		return [2][3]float64{}, false
	}
	return [2][3]float64{{mx[0], mx[1], mx[2]}, {my[0], my[1], my[2]}}, true
}

// fitAffineLS fits a 6-DOF affine by least squares over the correspondences in
// idx, solving the 3×3 normal equations (shared coefficient matrix for x and y).
func fitAffineLS(src, dst []cv.Point, idx []int) ([2][3]float64, bool) {
	if len(idx) < 3 {
		return [2][3]float64{}, false
	}
	var ata [3][3]float64
	var atbx, atby [3]float64
	for _, i := range idx {
		x := float64(src[i].X)
		y := float64(src[i].Y)
		u := float64(dst[i].X)
		v := float64(dst[i].Y)
		row := [3]float64{x, y, 1}
		for r := 0; r < 3; r++ {
			for c := 0; c < 3; c++ {
				ata[r][c] += row[r] * row[c]
			}
			atbx[r] += row[r] * u
			atby[r] += row[r] * v
		}
	}
	mx, ok1 := solve3(ata, atbx)
	my, ok2 := solve3(ata, atby)
	if !ok1 || !ok2 {
		return [2][3]float64{}, false
	}
	return [2][3]float64{{mx[0], mx[1], mx[2]}, {my[0], my[1], my[2]}}, true
}

// fitPartialMinimal fits an exact 4-DOF similarity transform to two
// correspondences.
func fitPartialMinimal(src, dst []cv.Point, idx []int) ([2][3]float64, bool) {
	p1, p2 := src[idx[0]], src[idx[1]]
	q1, q2 := dst[idx[0]], dst[idx[1]]
	dpx := float64(p2.X - p1.X)
	dpy := float64(p2.Y - p1.Y)
	dqx := float64(q2.X - q1.X)
	dqy := float64(q2.Y - q1.Y)
	den := dpx*dpx + dpy*dpy
	if den == 0 {
		return [2][3]float64{}, false
	}
	a := (dpx*dqx + dpy*dqy) / den
	b := (dpx*dqy - dpy*dqx) / den
	tx := float64(q1.X) - (a*float64(p1.X) - b*float64(p1.Y))
	ty := float64(q1.Y) - (b*float64(p1.X) + a*float64(p1.Y))
	return [2][3]float64{{a, -b, tx}, {b, a, ty}}, true
}

// fitPartialLS fits a 4-DOF similarity transform by least squares over the
// correspondences in idx, solving the 4×4 normal equations for the parameters
// (a, b, tx, ty).
func fitPartialLS(src, dst []cv.Point, idx []int) ([2][3]float64, bool) {
	if len(idx) < 2 {
		return [2][3]float64{}, false
	}
	// Each correspondence contributes two rows of the design matrix J for the
	// unknown vector [a, b, tx, ty]:
	//   u = a*x - b*y + tx  -> row [ x, -y, 1, 0]
	//   v = b*x + a*y + ty  -> row [ y,  x, 0, 1]
	var jtj [4][4]float64
	var jtb [4]float64
	acc := func(row [4]float64, rhs float64) {
		for r := 0; r < 4; r++ {
			for c := 0; c < 4; c++ {
				jtj[r][c] += row[r] * row[c]
			}
			jtb[r] += row[r] * rhs
		}
	}
	for _, i := range idx {
		x := float64(src[i].X)
		y := float64(src[i].Y)
		u := float64(dst[i].X)
		v := float64(dst[i].Y)
		acc([4]float64{x, -y, 1, 0}, u)
		acc([4]float64{y, x, 0, 1}, v)
	}
	sol, ok := solveLinearSystem(mat4ToSlice(jtj), jtb[:])
	if !ok {
		return [2][3]float64{}, false
	}
	a, b, tx, ty := sol[0], sol[1], sol[2], sol[3]
	return [2][3]float64{{a, -b, tx}, {b, a, ty}}, true
}

// chooseDistinct fills pick with len(pick) distinct indices in [0,n) drawn from
// rng. It assumes n >= len(pick).
func chooseDistinct(rng *rand.Rand, n int, pick []int) {
	for i := range pick {
		for {
			candidate := rng.Intn(n)
			dup := false
			for j := 0; j < i; j++ {
				if pick[j] == candidate {
					dup = true
					break
				}
			}
			if !dup {
				pick[i] = candidate
				break
			}
		}
	}
}

// allIndices returns the slice [0, 1, ..., n-1].
func allIndices(n int) []int {
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	return idx
}

// maskToIndices returns the indices where mask is true.
func maskToIndices(mask []bool) []int {
	var idx []int
	for i, v := range mask {
		if v {
			idx = append(idx, i)
		}
	}
	return idx
}

// solve3 solves the 3×3 system a·x = b by Gauss–Jordan elimination with partial
// pivoting, reporting whether a is non-singular. It does not modify its inputs.
func solve3(a [3][3]float64, b [3]float64) ([3]float64, bool) {
	sol, ok := solveLinearSystem(mat3ToSlice(a), b[:])
	if !ok {
		return [3]float64{}, false
	}
	return [3]float64{sol[0], sol[1], sol[2]}, true
}

// mat3ToSlice copies a 3×3 array into a freshly allocated [][]float64.
func mat3ToSlice(a [3][3]float64) [][]float64 {
	out := make([][]float64, 3)
	for r := 0; r < 3; r++ {
		out[r] = []float64{a[r][0], a[r][1], a[r][2]}
	}
	return out
}

// mat4ToSlice copies a 4×4 array into a freshly allocated [][]float64.
func mat4ToSlice(a [4][4]float64) [][]float64 {
	out := make([][]float64, 4)
	for r := 0; r < 4; r++ {
		out[r] = []float64{a[r][0], a[r][1], a[r][2], a[r][3]}
	}
	return out
}

// solveLinearSystem solves the square linear system a·x = b by Gauss–Jordan
// elimination with partial pivoting, reporting whether the matrix is
// non-singular. It copies a and b, leaving the caller's data untouched.
func solveLinearSystem(a [][]float64, b []float64) ([]float64, bool) {
	n := len(b)
	m := make([][]float64, n)
	for i := range m {
		m[i] = make([]float64, n)
		copy(m[i], a[i])
	}
	x := make([]float64, n)
	copy(x, b)
	for col := 0; col < n; col++ {
		piv := col
		best := math.Abs(m[col][col])
		for r := col + 1; r < n; r++ {
			if math.Abs(m[r][col]) > best {
				best = math.Abs(m[r][col])
				piv = r
			}
		}
		if best < 1e-12 {
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
