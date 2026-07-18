package shapefit

import (
	"errors"
	"math"
	"math/rand"

	cv "github.com/malcolmston/opencv"
)

// RANSACParams configures the randomized fitting routines. The randomness is
// fully determined by Seed, so a given (input, params) pair always yields the
// same result.
type RANSACParams struct {
	// MaxIterations is the number of random minimal subsets to try.
	MaxIterations int
	// Threshold is the maximum residual distance, in pixels, for a point to
	// count as an inlier of a candidate model.
	Threshold float64
	// MinInliers is the minimum inlier count for a model to be accepted. Zero
	// means "accept the best model found regardless of count".
	MinInliers int
	// Seed seeds the internal pseudo-random generator for reproducibility.
	Seed int64
	// RefitOnInliers, when true, re-estimates the final model from all its
	// inliers using the corresponding least-squares fit.
	RefitOnInliers bool
}

// DefaultRANSACParams returns a reasonable default configuration: 1000
// iterations, a 2-pixel inlier threshold, no minimum inlier count, seed 1 and
// least-squares refinement enabled.
func DefaultRANSACParams() RANSACParams {
	return RANSACParams{
		MaxIterations:  1000,
		Threshold:      2.0,
		MinInliers:     0,
		Seed:           1,
		RefitOnInliers: true,
	}
}

// CircleThrough returns the unique circle passing through the three points. It
// reports false when the points are collinear.
func CircleThrough(a, b, c cv.Point2f) (Circle, bool) {
	ax, ay := a.X, a.Y
	bx, by := b.X, b.Y
	cx, cy := c.X, c.Y
	d := 2 * (ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
	if math.Abs(d) < shapefitEps {
		return Circle{}, false
	}
	a2 := ax*ax + ay*ay
	b2 := bx*bx + by*by
	c2 := cx*cx + cy*cy
	ux := (a2*(by-cy) + b2*(cy-ay) + c2*(ay-by)) / d
	uy := (a2*(cx-bx) + b2*(ax-cx) + c2*(bx-ax)) / d
	center := cv.Point2f{X: ux, Y: uy}
	r := math.Hypot(ax-ux, ay-uy)
	return Circle{Center: center, Radius: r}, true
}

// Distance returns an approximate signed-free radial distance from p to the
// ellipse boundary, measured along the ray from the center through p. It is
// exact for points on the major/minor axes and a close approximation elsewhere,
// which is sufficient for inlier tests. It returns a large value for the
// degenerate center point.
func (e Ellipse) Distance(p cv.Point2f) float64 {
	if e.SemiMajor < shapefitEps || e.SemiMinor < shapefitEps {
		return math.MaxFloat64
	}
	ca := math.Cos(e.Angle)
	sa := math.Sin(e.Angle)
	dx := p.X - e.Center.X
	dy := p.Y - e.Center.Y
	u := dx*ca + dy*sa
	v := -dx*sa + dy*ca
	uu := u / e.SemiMajor
	vv := v / e.SemiMinor
	r := math.Hypot(uu, vv)
	if r < shapefitEps {
		return math.Min(e.SemiMajor, e.SemiMinor)
	}
	dist := math.Hypot(dx, dy)
	return math.Abs(1-1/r) * dist
}

// countInliers returns the indices of points whose residual under dist is at
// most threshold.
func shapefitInliers(pts []cv.Point2f, dist func(cv.Point2f) float64, threshold float64) []int {
	var idx []int
	for i, p := range pts {
		if dist(p) <= threshold {
			idx = append(idx, i)
		}
	}
	return idx
}

// shapefitSubset returns k distinct random indices in [0, n) drawn from rng.
func shapefitSubset(rng *rand.Rand, n, k int) []int {
	if k > n {
		return nil
	}
	idx := make([]int, k)
	chosen := make(map[int]bool, k)
	for i := 0; i < k; i++ {
		for {
			j := rng.Intn(n)
			if !chosen[j] {
				chosen[j] = true
				idx[i] = j
				break
			}
		}
	}
	return idx
}

// RANSACLine robustly fits a line to points containing outliers. It repeatedly
// fits a line through a random pair, counts inliers within params.Threshold and
// keeps the model with the most inliers, optionally refitting on the inlier set.
// It returns the fitted line and the sorted indices of its inliers, or an error
// when there are too few points or no model meets params.MinInliers.
func RANSACLine(pts []cv.Point2f, params RANSACParams) (Line, []int, error) {
	if len(pts) < 2 {
		return Line{}, nil, errors.New("shapefit: RANSACLine needs at least 2 points")
	}
	rng := rand.New(rand.NewSource(params.Seed))
	var bestLine Line
	var bestInliers []int
	for it := 0; it < params.MaxIterations; it++ {
		s := shapefitSubset(rng, len(pts), 2)
		cand := LineThroughPoints(pts[s[0]], pts[s[1]])
		if cand == (Line{}) {
			continue
		}
		in := shapefitInliers(pts, cand.Distance, params.Threshold)
		if len(in) > len(bestInliers) {
			bestInliers = in
			bestLine = cand
		}
	}
	if len(bestInliers) < params.MinInliers || len(bestInliers) < 2 {
		return Line{}, nil, errors.New("shapefit: RANSACLine found no acceptable model")
	}
	if params.RefitOnInliers {
		sub := shapefitGather(pts, bestInliers)
		if refit, err := FitLine(sub); err == nil {
			bestLine = refit
			bestInliers = shapefitInliers(pts, bestLine.Distance, params.Threshold)
		}
	}
	return bestLine, bestInliers, nil
}

// RANSACCircle robustly fits a circle to points containing outliers, sampling
// random point triples as minimal circle models. It returns the fitted circle
// and the sorted indices of its inliers, or an error when there are too few
// points or no model meets params.MinInliers.
func RANSACCircle(pts []cv.Point2f, params RANSACParams) (Circle, []int, error) {
	if len(pts) < 3 {
		return Circle{}, nil, errors.New("shapefit: RANSACCircle needs at least 3 points")
	}
	rng := rand.New(rand.NewSource(params.Seed))
	var best Circle
	var bestInliers []int
	for it := 0; it < params.MaxIterations; it++ {
		s := shapefitSubset(rng, len(pts), 3)
		cand, ok := CircleThrough(pts[s[0]], pts[s[1]], pts[s[2]])
		if !ok {
			continue
		}
		in := shapefitInliers(pts, cand.Distance, params.Threshold)
		if len(in) > len(bestInliers) {
			bestInliers = in
			best = cand
		}
	}
	if len(bestInliers) < params.MinInliers || len(bestInliers) < 3 {
		return Circle{}, nil, errors.New("shapefit: RANSACCircle found no acceptable model")
	}
	if params.RefitOnInliers {
		sub := shapefitGather(pts, bestInliers)
		if refit, err := FitCircleTaubin(sub); err == nil {
			best = refit
			bestInliers = shapefitInliers(pts, best.Distance, params.Threshold)
		}
	}
	return best, bestInliers, nil
}

// RANSACEllipse robustly fits an ellipse to points containing outliers,
// sampling random subsets of five points as minimal ellipse models and scoring
// by approximate radial distance. It returns the fitted ellipse and the sorted
// indices of its inliers, or an error when there are too few points or no model
// meets params.MinInliers.
func RANSACEllipse(pts []cv.Point2f, params RANSACParams) (Ellipse, []int, error) {
	if len(pts) < 5 {
		return Ellipse{}, nil, errors.New("shapefit: RANSACEllipse needs at least 5 points")
	}
	rng := rand.New(rand.NewSource(params.Seed))
	var best Ellipse
	var bestInliers []int
	for it := 0; it < params.MaxIterations; it++ {
		s := shapefitSubset(rng, len(pts), 5)
		sub := shapefitGather(pts, s)
		cand, err := FitEllipse(sub)
		if err != nil {
			continue
		}
		in := shapefitInliers(pts, cand.Distance, params.Threshold)
		if len(in) > len(bestInliers) {
			bestInliers = in
			best = cand
		}
	}
	if len(bestInliers) < params.MinInliers || len(bestInliers) < 5 {
		return Ellipse{}, nil, errors.New("shapefit: RANSACEllipse found no acceptable model")
	}
	if params.RefitOnInliers {
		sub := shapefitGather(pts, bestInliers)
		if refit, err := FitEllipse(sub); err == nil {
			best = refit
			bestInliers = shapefitInliers(pts, best.Distance, params.Threshold)
		}
	}
	return best, bestInliers, nil
}

// shapefitGather returns the points at the given indices.
func shapefitGather(pts []cv.Point2f, idx []int) []cv.Point2f {
	out := make([]cv.Point2f, len(idx))
	for i, j := range idx {
		out[i] = pts[j]
	}
	return out
}

// DetectLines extracts up to maxLines lines from a point set by sequential
// RANSAC: it fits the dominant line, removes its inliers, and repeats on the
// remaining points until maxLines lines are found or too few points remain. The
// lines are returned in order of decreasing support.
func DetectLines(pts []cv.Point2f, params RANSACParams, maxLines int) []Line {
	remaining := append([]cv.Point2f(nil), pts...)
	var out []Line
	for len(out) < maxLines && len(remaining) >= 2 {
		line, inliers, err := RANSACLine(remaining, params)
		if err != nil || len(inliers) < 2 {
			break
		}
		out = append(out, line)
		remaining = shapefitRemove(remaining, inliers)
	}
	return out
}

// DetectCircles extracts up to maxCircles circles from a point set by
// sequential RANSAC: it fits the dominant circle, removes its inliers, and
// repeats on the remaining points. The circles are returned in order of
// decreasing support.
func DetectCircles(pts []cv.Point2f, params RANSACParams, maxCircles int) []Circle {
	remaining := append([]cv.Point2f(nil), pts...)
	var out []Circle
	for len(out) < maxCircles && len(remaining) >= 3 {
		c, inliers, err := RANSACCircle(remaining, params)
		if err != nil || len(inliers) < 3 {
			break
		}
		out = append(out, c)
		remaining = shapefitRemove(remaining, inliers)
	}
	return out
}

// shapefitRemove returns the points whose indices are not in idx.
func shapefitRemove(pts []cv.Point2f, idx []int) []cv.Point2f {
	drop := make(map[int]bool, len(idx))
	for _, j := range idx {
		drop[j] = true
	}
	out := make([]cv.Point2f, 0, len(pts)-len(idx))
	for i, p := range pts {
		if !drop[i] {
			out = append(out, p)
		}
	}
	return out
}
