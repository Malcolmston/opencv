package matching2

import (
	"math"
	"math/rand"
	"sort"

	"github.com/malcolmston/opencv/core"
)

// DefaultRANSACSeed is the fixed seed used by the RANSAC-based estimators in
// this package (for example [FindHomographyRANSAC]) so that their output is
// fully deterministic. Callers of the generic [RANSAC] and [LMedS] helpers pass
// their own seed.
const DefaultRANSACSeed int64 = 0x5eed1234

// RANSACResult holds the outcome of a generic robust fit: the recovered model,
// the boolean inlier mask over the input observations and the inlier count.
type RANSACResult[M any] struct {
	// Model is the best model found, valid only when Ok is true.
	Model M
	// Inliers marks, for each observation, whether it agrees with Model.
	Inliers []bool
	// NumInliers is the number of true entries in Inliers.
	NumInliers int
	// Ok reports whether a model meeting the requirements was found.
	Ok bool
}

// RANSAC performs Random Sample Consensus over n observations. On each of iters
// iterations it draws sampleSize distinct observation indices, fits a candidate
// model with fit, and scores it with inliers, which must return a mask marking
// the agreeing observations. The model with the most inliers is refit (if refit
// is non-nil) over its full inlier set and returned. A model is accepted only
// when it has at least minInliers inliers.
//
// fit returns false when the sampled observations are degenerate; such samples
// are skipped. Sampling is driven by a generator seeded with seed, so the result
// is deterministic for fixed inputs. This is the general engine used by the
// geometry estimators and is exported for fitting custom models.
func RANSAC[M any](
	n, sampleSize, iters, minInliers int,
	seed int64,
	fit func(sample []int) (M, bool),
	inliers func(model M) []bool,
	refit func(sample []int) (M, bool),
) RANSACResult[M] {
	var res RANSACResult[M]
	if n < sampleSize || sampleSize <= 0 {
		return res
	}
	rng := rand.New(rand.NewSource(seed))
	bestCount := -1
	var bestMask []bool
	for it := 0; it < iters; it++ {
		sample := matching2sampleK(rng, n, sampleSize)
		model, ok := fit(sample)
		if !ok {
			continue
		}
		mask := inliers(model)
		c := countTrue(mask)
		if c > bestCount {
			bestCount = c
			bestMask = mask
		}
	}
	if bestCount < minInliers || bestMask == nil {
		return res
	}
	sample := maskIndices(bestMask)
	final := sample
	fitFn := refit
	if fitFn == nil {
		fitFn = fit
	}
	model, ok := fitFn(final)
	if !ok {
		return res
	}
	// Recompute the mask from the refined model for a consistent report.
	mask := inliers(model)
	if countTrue(mask) < bestCount {
		// Refit worsened the consensus; keep the sample-based model instead.
		model, ok = fit(sample)
		if !ok {
			return res
		}
		mask = inliers(model)
	}
	res.Model = model
	res.Inliers = mask
	res.NumInliers = countTrue(mask)
	res.Ok = res.NumInliers >= minInliers
	return res
}

// LMedS performs least-median-of-squares robust fitting over n observations. On
// each of iters iterations it draws sampleSize indices, fits a model with fit
// and evaluates residuals for every observation with residual. The model
// minimising the median squared residual is kept; an observation is flagged an
// inlier when its residual is within a robust threshold derived from that median.
// Unlike [RANSAC], LMedS needs no inlier threshold, but it assumes fewer than
// half the observations are outliers. Sampling uses seed and is deterministic.
func LMedS[M any](
	n, sampleSize, iters int,
	seed int64,
	fit func(sample []int) (M, bool),
	residual func(model M, i int) float64,
) RANSACResult[M] {
	var res RANSACResult[M]
	if n < sampleSize || sampleSize <= 0 {
		return res
	}
	rng := rand.New(rand.NewSource(seed))
	bestMed := math.Inf(1)
	var bestModel M
	found := false
	for it := 0; it < iters; it++ {
		sample := matching2sampleK(rng, n, sampleSize)
		model, ok := fit(sample)
		if !ok {
			continue
		}
		sq := make([]float64, n)
		for i := 0; i < n; i++ {
			r := residual(model, i)
			sq[i] = r * r
		}
		med := median(sq)
		if med < bestMed {
			bestMed = med
			bestModel = model
			found = true
		}
	}
	if !found {
		return res
	}
	// Robust standard deviation estimate (Rousseeuw & Leroy). The 1.4826 factor
	// makes the median absolute deviation a consistent estimator of sigma for
	// Gaussian noise; the (1 + 5/(n-sampleSize)) term corrects small-sample bias.
	sigma := 1.4826 * (1 + 5.0/math.Max(1, float64(n-sampleSize))) * math.Sqrt(bestMed)
	thr := 2.5 * sigma
	if thr <= 0 {
		thr = 1e-9
	}
	mask := make([]bool, n)
	for i := 0; i < n; i++ {
		mask[i] = math.Abs(residual(bestModel, i)) <= thr
	}
	res.Model = bestModel
	res.Inliers = mask
	res.NumInliers = countTrue(mask)
	res.Ok = true
	return res
}

// Line2D represents the 2-D line a·x + b·y + c = 0 with (a, b) a unit normal, so
// that a·px + b·py + c is the signed perpendicular distance from a point p to the
// line.
type Line2D struct {
	A, B, C float64
}

// Distance returns the absolute perpendicular distance from point p to the line.
func (l Line2D) Distance(p core.Point2d) float64 {
	return math.Abs(l.A*p.X + l.B*p.Y + l.C)
}

// FitLine returns the total-least-squares line through the given points (at
// least two), minimising the sum of squared perpendicular distances. It reports
// false when the points are too few or coincident.
func FitLine(points []core.Point2d) (Line2D, bool) {
	n := len(points)
	if n < 2 {
		return Line2D{}, false
	}
	var mx, my float64
	for _, p := range points {
		mx += p.X
		my += p.Y
	}
	mx /= float64(n)
	my /= float64(n)
	var sxx, sxy, syy float64
	for _, p := range points {
		dx := p.X - mx
		dy := p.Y - my
		sxx += dx * dx
		sxy += dx * dy
		syy += dy * dy
	}
	// The line direction is the dominant eigenvector of the scatter matrix; the
	// normal is the eigenvector of the smallest eigenvalue.
	cov := [][]float64{{sxx, sxy}, {sxy, syy}}
	vals, vecs := matching2symEig(cov)
	_ = vals
	nx, ny := vecs[0][0], vecs[0][1] // smallest-eigenvalue eigenvector = normal
	norm := math.Hypot(nx, ny)
	if norm < 1e-300 {
		return Line2D{}, false
	}
	nx /= norm
	ny /= norm
	c := -(nx*mx + ny*my)
	return Line2D{A: nx, B: ny, C: c}, true
}

// FitLineRANSAC robustly fits a line to points, tolerating outliers. A point is
// an inlier when its perpendicular distance to the line is at most threshold.
// iters bounds the number of random samples and seed makes the result
// deterministic. It reports Ok false when fewer than two inliers are found.
func FitLineRANSAC(points []core.Point2d, threshold float64, iters int, seed int64) RANSACResult[Line2D] {
	fit := func(sample []int) (Line2D, bool) {
		pts := make([]core.Point2d, len(sample))
		for i, s := range sample {
			pts[i] = points[s]
		}
		return FitLine(pts)
	}
	inliers := func(l Line2D) []bool {
		mask := make([]bool, len(points))
		for i, p := range points {
			mask[i] = l.Distance(p) <= threshold
		}
		return mask
	}
	refit := func(sample []int) (Line2D, bool) {
		pts := make([]core.Point2d, len(sample))
		for i, s := range sample {
			pts[i] = points[s]
		}
		return FitLine(pts)
	}
	return RANSAC(len(points), 2, iters, 2, seed, fit, inliers, refit)
}

// NormalizePoints2D applies Hartley's isotropic normalization: it returns points
// translated so their centroid is the origin and scaled so their mean distance
// from the origin is √2, together with the 3×3 similarity transform T that
// achieves it (so normalized = T·homogeneous(point)). Normalization greatly
// improves the conditioning of the direct-linear-transform solvers.
func NormalizePoints2D(pts []core.Point2d) ([]core.Point2d, [3][3]float64) {
	n := len(pts)
	if n == 0 {
		return nil, Mat3Identity()
	}
	var cx, cy float64
	for _, p := range pts {
		cx += p.X
		cy += p.Y
	}
	cx /= float64(n)
	cy /= float64(n)
	var meanDist float64
	for _, p := range pts {
		meanDist += math.Hypot(p.X-cx, p.Y-cy)
	}
	meanDist /= float64(n)
	scale := 1.0
	if meanDist > 1e-300 {
		scale = math.Sqrt2 / meanDist
	}
	T := [3][3]float64{
		{scale, 0, -scale * cx},
		{0, scale, -scale * cy},
		{0, 0, 1},
	}
	out := make([]core.Point2d, n)
	for i, p := range pts {
		out[i] = core.Point2d{X: scale * (p.X - cx), Y: scale * (p.Y - cy)}
	}
	return out, T
}

// matching2sampleK returns k distinct indices in [0, n) drawn from rng using
// partial Fisher–Yates, in ascending order for stable downstream behaviour.
func matching2sampleK(rng *rand.Rand, n, k int) []int {
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	for i := 0; i < k; i++ {
		j := i + rng.Intn(n-i)
		perm[i], perm[j] = perm[j], perm[i]
	}
	out := perm[:k]
	sort.Ints(out)
	return out
}

// countTrue returns the number of true entries in mask.
func countTrue(mask []bool) int {
	c := 0
	for _, b := range mask {
		if b {
			c++
		}
	}
	return c
}

// maskIndices returns the indices where mask is true.
func maskIndices(mask []bool) []int {
	var out []int
	for i, b := range mask {
		if b {
			out = append(out, i)
		}
	}
	return out
}

// median returns the median of vals; it copies and sorts, leaving vals intact.
func median(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	c := make([]float64, len(vals))
	copy(c, vals)
	sort.Float64s(c)
	n := len(c)
	if n%2 == 1 {
		return c[n/2]
	}
	return 0.5 * (c[n/2-1] + c[n/2])
}
