package calib3d

import (
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// Method selects the estimator used by [FindHomography]. The integer values
// match the corresponding OpenCV flags so callers can pass familiar constants.
const (
	// MethodDirect fits a single homography to all correspondences with the
	// normalized Direct Linear Transform. With exactly four points the fit is
	// exact; with more it is the least-squares solution. Every correspondence is
	// reported as an inlier.
	MethodDirect = 0
	// MethodRANSAC robustly fits a homography with Random Sample Consensus,
	// tolerating gross outliers. The returned inlier mask flags the
	// correspondences that agree with the recovered model.
	MethodRANSAC = 8
)

// ransacSeed is a fixed seed so that [FindHomography] with [MethodRANSAC] is
// fully deterministic: identical inputs always yield identical output. This is
// a deliberate departure from OpenCV, whose RANSAC is non-deterministic.
const ransacSeed = 0x5eed1234

// FindHomography estimates the 3×3 projective transform H that maps the source
// points src onto the destination points dst, so that for each i the homogeneous
// product H·(srcᵢ,1) is proportional to (dstᵢ,1). src and dst must have equal
// length of at least four.
//
// method selects the estimator: [MethodDirect] fits all points with the
// normalized Direct Linear Transform (exact for four points, least-squares for
// more), while [MethodRANSAC] robustly rejects outliers using ransacThresh as
// the maximum symmetric reprojection error, in pixels, for a correspondence to
// count as an inlier. ransacThresh is ignored by [MethodDirect].
//
// The returned H is normalized so that H[2][2] == 1 when possible. The inliers
// slice has the same length as src: for [MethodDirect] every entry is true; for
// [MethodRANSAC] an entry is true when that correspondence agrees with the
// recovered model. If estimation fails (degenerate or insufficient input) H is
// the zero matrix and inliers is nil.
func FindHomography(src, dst []cv.Point, method int, ransacThresh float64) (H [3][3]float64, inliers []bool) {
	if len(src) != len(dst) || len(src) < 4 {
		return [3][3]float64{}, nil
	}
	n := len(src)
	sp := pointsToF(src)
	dp := pointsToF(dst)

	if method == MethodRANSAC && n > 4 {
		return ransacHomography(sp, dp, ransacThresh)
	}

	// Exact four-point case: reuse the root package's GetPerspectiveTransform,
	// which solves the minimal 8×8 system directly.
	if n == 4 {
		if h, ok := safePerspective([4]cv.Point{src[0], src[1], src[2], src[3]},
			[4]cv.Point{dst[0], dst[1], dst[2], dst[3]}); ok {
			mask := make([]bool, n)
			for i := range mask {
				mask[i] = true
			}
			return h, mask
		}
	}

	h, ok := dltHomography(sp, dp)
	if !ok {
		return [3][3]float64{}, nil
	}
	mask := make([]bool, n)
	for i := range mask {
		mask[i] = true
	}
	return h, mask
}

// safePerspective wraps the root package's GetPerspectiveTransform, which
// panics on degenerate input, converting the panic into an ok == false result
// and reshaping the [9]float64 into a [3][3]float64.
func safePerspective(src, dst [4]cv.Point) (h [3][3]float64, ok bool) {
	defer func() {
		if recover() != nil {
			h = [3][3]float64{}
			ok = false
		}
	}()
	m := cv.GetPerspectiveTransform(src, dst)
	return [3][3]float64{
		{m[0], m[1], m[2]},
		{m[3], m[4], m[5]},
		{m[6], m[7], m[8]},
	}, true
}

// dltHomography computes a homography from n≥4 correspondences using the
// normalized Direct Linear Transform (Hartley normalization + null space of the
// 2n×9 design matrix). It reports ok == false on degenerate input.
func dltHomography(src, dst [][2]float64) ([3][3]float64, bool) {
	if len(src) != len(dst) || len(src) < 4 {
		return [3][3]float64{}, false
	}
	ts, ns, ok1 := normalizePoints(src)
	td, nd, ok2 := normalizePoints(dst)
	if !ok1 || !ok2 {
		return [3][3]float64{}, false
	}
	// Build the 9×9 normal matrix AᵀA from the 2n×9 DLT rows.
	var ata [9][9]float64
	for i := range ns {
		x, y := ns[i][0], ns[i][1]
		u, vv := nd[i][0], nd[i][1]
		rows := [2][9]float64{
			{-x, -y, -1, 0, 0, 0, u * x, u * y, u},
			{0, 0, 0, -x, -y, -1, vv * x, vv * y, vv},
		}
		for _, r := range rows {
			for a := 0; a < 9; a++ {
				for b := 0; b < 9; b++ {
					ata[a][b] += r[a] * r[b]
				}
			}
		}
	}
	dyn := make([][]float64, 9)
	for i := 0; i < 9; i++ {
		dyn[i] = make([]float64, 9)
		copy(dyn[i], ata[i][:])
	}
	h := smallestEigvec(dyn)
	hn := [3][3]float64{
		{h[0], h[1], h[2]},
		{h[3], h[4], h[5]},
		{h[6], h[7], h[8]},
	}
	// Denormalize: H = Td⁻¹ · Hn · Ts.
	tdInv, ok := inv3(td)
	if !ok {
		return [3][3]float64{}, false
	}
	res := mul3(tdInv, mul3(hn, ts))
	if math.Abs(res[2][2]) > 1e-15 {
		s := 1 / res[2][2]
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				res[i][j] *= s
			}
		}
	}
	return res, true
}

// ransacHomography runs deterministic RANSAC: it repeatedly samples four
// correspondences, fits a minimal homography, scores it by the number of inliers
// under the symmetric transfer error, then refits the best consensus set with
// the DLT.
func ransacHomography(src, dst [][2]float64, thresh float64) ([3][3]float64, []bool) {
	n := len(src)
	if thresh <= 0 {
		thresh = 3.0
	}
	t2 := thresh * thresh
	rng := rand.New(rand.NewSource(ransacSeed))
	const iters = 2000

	var bestMask []bool
	bestCount := -1
	var bestH [3][3]float64

	for it := 0; it < iters; it++ {
		i0, i1, i2, i3 := sample4(rng, n)
		sc := [4]cv.Point{fToPoint(src[i0]), fToPoint(src[i1]), fToPoint(src[i2]), fToPoint(src[i3])}
		dc := [4]cv.Point{fToPoint(dst[i0]), fToPoint(dst[i1]), fToPoint(dst[i2]), fToPoint(dst[i3])}
		h, ok := safePerspective(sc, dc)
		if !ok {
			continue
		}
		mask, count := scoreHomography(h, src, dst, t2)
		if count > bestCount {
			bestCount = count
			bestMask = mask
			bestH = h
		}
	}
	if bestCount < 4 {
		return [3][3]float64{}, nil
	}
	// Refit on all inliers with the normalized DLT for a stable final estimate.
	var is, id [][2]float64
	for i := 0; i < n; i++ {
		if bestMask[i] {
			is = append(is, src[i])
			id = append(id, dst[i])
		}
	}
	if h, ok := dltHomography(is, id); ok {
		// Re-score with the refined model so the mask matches the returned H.
		mask, count := scoreHomography(h, src, dst, t2)
		if count >= bestCount {
			return h, mask
		}
	}
	return bestH, bestMask
}

// scoreHomography classifies each correspondence as an inlier when the squared
// symmetric transfer error (forward plus backward reprojection) is within t2.
func scoreHomography(h [3][3]float64, src, dst [][2]float64, t2 float64) ([]bool, int) {
	hInv, ok := inv3(h)
	mask := make([]bool, len(src))
	count := 0
	for i := range src {
		fx, fy, okf := applyH(h, src[i])
		if !okf {
			continue
		}
		e := sq(fx-dst[i][0]) + sq(fy-dst[i][1])
		if ok {
			bx, by, okb := applyH(hInv, dst[i])
			if okb {
				e += sq(bx-src[i][0]) + sq(by-src[i][1])
			}
		}
		if e <= t2 {
			mask[i] = true
			count++
		}
	}
	return mask, count
}

// applyH applies a homography to a 2D point, returning the projected point and
// reporting whether the homogeneous denominator was non-zero.
func applyH(h [3][3]float64, p [2]float64) (x, y float64, ok bool) {
	w := h[2][0]*p[0] + h[2][1]*p[1] + h[2][2]
	if math.Abs(w) < 1e-15 {
		return 0, 0, false
	}
	x = (h[0][0]*p[0] + h[0][1]*p[1] + h[0][2]) / w
	y = (h[1][0]*p[0] + h[1][1]*p[1] + h[1][2]) / w
	return x, y, true
}

// FindFundamentalMat estimates the 3×3 fundamental matrix F relating two views
// from corresponding points, using the normalized eight-point algorithm. For
// every correspondence, (pts2ᵢ,1)ᵀ · F · (pts1ᵢ,1) is driven to zero. pts1 and
// pts2 must have equal length of at least eight.
//
// The estimate is Hartley-normalized for conditioning and the rank-2 constraint
// is enforced by zeroing the smallest singular value. F is scaled to unit
// Frobenius norm. ok is false when the input is insufficient or degenerate.
func FindFundamentalMat(pts1, pts2 []cv.Point) (F [3][3]float64, ok bool) {
	if len(pts1) != len(pts2) || len(pts1) < 8 {
		return [3][3]float64{}, false
	}
	p1 := pointsToF(pts1)
	p2 := pointsToF(pts2)
	t1, n1, ok1 := normalizePoints(p1)
	t2, n2, ok2 := normalizePoints(p2)
	if !ok1 || !ok2 {
		return [3][3]float64{}, false
	}
	// Build AᵀA (9×9) from the epipolar constraint rows.
	var ata [9][9]float64
	for i := range n1 {
		x, y := n1[i][0], n1[i][1]
		xp, yp := n2[i][0], n2[i][1]
		r := [9]float64{xp * x, xp * y, xp, yp * x, yp * y, yp, x, y, 1}
		for a := 0; a < 9; a++ {
			for b := 0; b < 9; b++ {
				ata[a][b] += r[a] * r[b]
			}
		}
	}
	dyn := make([][]float64, 9)
	for i := 0; i < 9; i++ {
		dyn[i] = make([]float64, 9)
		copy(dyn[i], ata[i][:])
	}
	f := smallestEigvec(dyn)
	fn := [3][3]float64{
		{f[0], f[1], f[2]},
		{f[3], f[4], f[5]},
		{f[6], f[7], f[8]},
	}
	// Enforce rank 2 by zeroing the smallest singular value.
	u, s, v := svd3(fn)
	s[2] = 0
	fn = mul3(u, mul3([3][3]float64{{s[0], 0, 0}, {0, s[1], 0}, {0, 0, s[2]}}, transpose3(v)))
	// Denormalize: F = T2ᵀ · Fn · T1.
	res := mul3(transpose3(t2), mul3(fn, t1))
	// Scale to unit Frobenius norm.
	var fro float64
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			fro += res[i][j] * res[i][j]
		}
	}
	fro = math.Sqrt(fro)
	if fro < 1e-18 {
		return [3][3]float64{}, false
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			res[i][j] /= fro
		}
	}
	return res, true
}

// normalizePoints computes the Hartley similarity transform that centres a
// point set at the origin and scales it so the mean distance to the origin is
// √2, returning the 3×3 transform, the transformed points and whether the set
// was non-degenerate.
func normalizePoints(pts [][2]float64) (t [3][3]float64, out [][2]float64, ok bool) {
	n := len(pts)
	if n == 0 {
		return [3][3]float64{}, nil, false
	}
	var cx, cy float64
	for _, p := range pts {
		cx += p[0]
		cy += p[1]
	}
	cx /= float64(n)
	cy /= float64(n)
	var meanDist float64
	for _, p := range pts {
		meanDist += math.Hypot(p[0]-cx, p[1]-cy)
	}
	meanDist /= float64(n)
	if meanDist < 1e-15 {
		return [3][3]float64{}, nil, false
	}
	s := math.Sqrt2 / meanDist
	t = [3][3]float64{
		{s, 0, -s * cx},
		{0, s, -s * cy},
		{0, 0, 1},
	}
	out = make([][2]float64, n)
	for i, p := range pts {
		out[i] = [2]float64{s * (p[0] - cx), s * (p[1] - cy)}
	}
	return t, out, true
}

// sample4 draws four distinct indices in [0,n) from rng.
func sample4(rng *rand.Rand, n int) (a, b, c, d int) {
	a = rng.Intn(n)
	for {
		b = rng.Intn(n)
		if b != a {
			break
		}
	}
	for {
		c = rng.Intn(n)
		if c != a && c != b {
			break
		}
	}
	for {
		d = rng.Intn(n)
		if d != a && d != b && d != c {
			break
		}
	}
	return a, b, c, d
}

// pointsToF converts integer image points to float pairs.
func pointsToF(pts []cv.Point) [][2]float64 {
	out := make([][2]float64, len(pts))
	for i, p := range pts {
		out[i] = [2]float64{float64(p.X), float64(p.Y)}
	}
	return out
}

// fToPoint rounds a float pair back to an integer image point.
func fToPoint(p [2]float64) cv.Point {
	return cv.Point{X: int(math.Round(p[0])), Y: int(math.Round(p[1]))}
}

func sq(x float64) float64 { return x * x }
