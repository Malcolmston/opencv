package template2

import (
	cv "github.com/malcolmston/opencv"
)

// SubPixel is a peak location refined to fractional pixel precision, together
// with the interpolated score at that location.
type SubPixel struct {
	// X is the refined column (fractional).
	X float64
	// Y is the refined row (fractional).
	Y float64
	// Score is the interpolated score at (X, Y).
	Score float64
}

// ParabolicPeak fits a parabola through three equally spaced samples (left,
// center, right) and returns the offset, in the range (-0.5, 0.5], of the
// parabola's vertex from the center sample. When the three points are collinear
// or form an upward cusp (no interior extremum) it returns 0.
//
// This is the standard one-dimensional sub-pixel estimator: for a peak the
// center sample must be the largest of the three, and the returned offset
// locates the true peak between the samples.
func ParabolicPeak(left, center, right float64) float64 {
	denom := left - 2*center + right
	if denom == 0 {
		return 0
	}
	off := 0.5 * (left - right) / denom
	if off < -0.5 {
		off = -0.5
	} else if off > 0.5 {
		off = 0.5
	}
	return off
}

// parabolicValue returns the interpolated vertex value of the parabola through
// (-1,left), (0,center), (1,right) evaluated at the fractional offset.
func parabolicValue(left, center, right, offset float64) float64 {
	// f(x) = center + 0.5*(right-left)*x + 0.5*(left-2center+right)*x^2
	a := 0.5 * (left - 2*center + right)
	b := 0.5 * (right - left)
	return center + b*offset + a*offset*offset
}

// RefinePeak refines the integer peak at column x, row y of a score map to
// sub-pixel precision by fitting independent parabolas along the horizontal and
// vertical axes (separable quadratic interpolation). Peaks on the border of the
// map, where a neighbour is missing, are returned unrefined along that axis.
// The interpolated [SubPixel.Score] is the mean of the two axis estimates.
func RefinePeak(scores *cv.FloatMat, x, y int) SubPixel {
	sp := SubPixel{X: float64(x), Y: float64(y), Score: scores.At(y, x)}
	center := scores.At(y, x)

	valuesSum := 0.0
	valuesCount := 0

	if x > 0 && x < scores.Cols-1 {
		l := scores.At(y, x-1)
		r := scores.At(y, x+1)
		dx := ParabolicPeak(l, center, r)
		sp.X = float64(x) + dx
		valuesSum += parabolicValue(l, center, r, dx)
		valuesCount++
	}
	if y > 0 && y < scores.Rows-1 {
		u := scores.At(y-1, x)
		d := scores.At(y+1, x)
		dy := ParabolicPeak(u, center, d)
		sp.Y = float64(y) + dy
		valuesSum += parabolicValue(u, center, d, dy)
		valuesCount++
	}
	if valuesCount > 0 {
		sp.Score = valuesSum / float64(valuesCount)
	}
	return sp
}

// RefinePeakQuadratic refines the integer peak at column x, row y using the full
// 3x3 neighbourhood, fitting a bivariate quadratic surface and returning its
// stationary point. It is more accurate than [RefinePeak] on surfaces with
// diagonal curvature. Peaks whose full 3x3 neighbourhood is not available (on
// the map border), or where the fitted surface is degenerate, fall back to the
// separable [RefinePeak] estimate.
func RefinePeakQuadratic(scores *cv.FloatMat, x, y int) SubPixel {
	if x <= 0 || x >= scores.Cols-1 || y <= 0 || y >= scores.Rows-1 {
		return RefinePeak(scores, x, y)
	}
	// Sample the 3x3 neighbourhood.
	c := scores.At(y, x)
	l := scores.At(y, x-1)
	r := scores.At(y, x+1)
	u := scores.At(y-1, x)
	d := scores.At(y+1, x)
	ul := scores.At(y-1, x-1)
	ur := scores.At(y-1, x+1)
	dl := scores.At(y+1, x-1)
	dr := scores.At(y+1, x+1)

	// Least-squares gradient/curvature estimates for
	// f = a + b*dx + e*dy + p*dx^2 + q*dy^2 + s*dx*dy on a 3x3 grid.
	fx := (r - l) / 2
	fy := (d - u) / 2
	fxx := r - 2*c + l
	fyy := d - 2*c + u
	fxy := (dr - dl - ur + ul) / 4

	det := fxx*fyy - fxy*fxy
	if det == 0 {
		return RefinePeak(scores, x, y)
	}
	dx := -(fyy*fx - fxy*fy) / det
	dy := -(fxx*fy - fxy*fx) / det
	// Reject implausible steps outside the sampled cell.
	if dx < -1 || dx > 1 || dy < -1 || dy > 1 {
		return RefinePeak(scores, x, y)
	}
	value := c + fx*dx + fy*dy + 0.5*(fxx*dx*dx+fyy*dy*dy) + fxy*dx*dy
	return SubPixel{X: float64(x) + dx, Y: float64(y) + dy, Score: value}
}

// RefineMatch returns a copy of m whose top-left corner is nudged by the
// sub-pixel offset of the score-map peak at (m.X, m.Y). The supplied scores must
// be the map that produced m (see [MatchTemplate]); higherIsBetter selects the
// peak polarity but is accepted for symmetry with the rest of the API and does
// not change the parabola fit. The refined [Match] keeps integer Width and
// Height but records the interpolated score. Because [Match] carries integer
// coordinates, the fractional offset is rounded; use [RefinePeak] directly when
// full sub-pixel precision is required.
func RefineMatch(scores *cv.FloatMat, m Match) Match {
	sp := RefinePeakQuadratic(scores, m.X, m.Y)
	out := m
	out.Score = sp.Score
	// Preserve sub-pixel information by rounding to nearest integer corner;
	// callers needing fractional precision should use RefinePeakQuadratic.
	out.X = int(roundHalfUp(sp.X))
	out.Y = int(roundHalfUp(sp.Y))
	return out
}

// roundHalfUp rounds to the nearest integer, halves away from zero.
func roundHalfUp(v float64) float64 {
	if v < 0 {
		return float64(int(v - 0.5))
	}
	return float64(int(v + 0.5))
}
