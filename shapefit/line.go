package shapefit

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Line is an infinite 2D line in normalized normal form a·x + b·y + c = 0 with
// a² + b² = 1. In this form the value a·x + b·y + c evaluated at a point equals
// the signed perpendicular distance from the point to the line, which makes
// residual computations direct.
type Line struct {
	// A and B are the components of the unit normal vector.
	A, B float64
	// C is the signed distance from the origin to the line along the normal,
	// negated: the line is the locus where A·x + B·y + C = 0.
	C float64
}

// LineThroughPoints returns the normalized line passing through the two points.
// It returns the zero line when the points coincide.
func LineThroughPoints(p, q cv.Point2f) Line {
	// Normal is perpendicular to the direction (q-p).
	dx := q.X - p.X
	dy := q.Y - p.Y
	n := math.Hypot(dx, dy)
	if n < shapefitEps {
		return Line{}
	}
	a := -dy / n
	b := dx / n
	c := -(a*p.X + b*p.Y)
	return Line{A: a, B: b, C: c}
}

// Normalized returns an equivalent line whose normal (A, B) is unit length. It
// returns the receiver unchanged when the normal is already degenerate.
func (l Line) Normalized() Line {
	n := math.Hypot(l.A, l.B)
	if n < shapefitEps {
		return l
	}
	return Line{A: l.A / n, B: l.B / n, C: l.C / n}
}

// SignedDistance returns the signed perpendicular distance from p to the line;
// its magnitude is the distance and its sign indicates which side of the line p
// lies on. It assumes the line is normalized.
func (l Line) SignedDistance(p cv.Point2f) float64 {
	return l.A*p.X + l.B*p.Y + l.C
}

// Distance returns the unsigned perpendicular distance from p to the line.
func (l Line) Distance(p cv.Point2f) float64 {
	return math.Abs(l.SignedDistance(p))
}

// Angle returns the orientation of the line in radians in the range
// (-π/2, π/2], measured from the positive x-axis to the line's direction.
func (l Line) Angle() float64 {
	// Direction is perpendicular to the normal (A, B).
	return shapefitWrapPi(math.Atan2(-l.A, l.B))
}

// Direction returns a unit vector along the line (perpendicular to its normal).
func (l Line) Direction() cv.Point2f {
	return cv.Point2f{X: -l.B, Y: l.A}
}

// Normal returns the line's unit normal vector (A, B).
func (l Line) Normal() cv.Point2f {
	return cv.Point2f{X: l.A, Y: l.B}
}

// ClosestPoint returns the point on the line nearest to p (its orthogonal
// projection onto the line). It assumes the line is normalized.
func (l Line) ClosestPoint(p cv.Point2f) cv.Point2f {
	d := l.SignedDistance(p)
	return cv.Point2f{X: p.X - d*l.A, Y: p.Y - d*l.B}
}

// FitLine fits a line to the points by total least squares, minimizing the sum
// of squared perpendicular distances. It returns an error when fewer than two
// points are supplied or the points are coincident. The returned line is
// normalized and passes through the point centroid.
func FitLine(pts []cv.Point2f) (Line, error) {
	if len(pts) < 2 {
		return Line{}, errors.New("shapefit: FitLine needs at least 2 points")
	}
	c := Centroid(pts)
	var sxx, syy, sxy float64
	for _, p := range pts {
		dx := p.X - c.X
		dy := p.Y - c.Y
		sxx += dx * dx
		syy += dy * dy
		sxy += dx * dy
	}
	if sxx+syy < shapefitEps {
		return Line{}, errors.New("shapefit: FitLine points are coincident")
	}
	// Direction of maximum variance is the principal eigenvector of the
	// covariance matrix; its angle is 0.5·atan2(2·Sxy, Sxx-Syy).
	theta := 0.5 * math.Atan2(2*sxy, sxx-syy)
	dirx := math.Cos(theta)
	diry := math.Sin(theta)
	// Normal is perpendicular to the direction.
	a := -diry
	b := dirx
	cc := -(a*c.X + b*c.Y)
	return Line{A: a, B: b, C: cc}.Normalized(), nil
}

// LineResiduals returns the unsigned perpendicular distance from each point to
// the line, in the same order as the input.
func LineResiduals(l Line, pts []cv.Point2f) []float64 {
	out := make([]float64, len(pts))
	for i, p := range pts {
		out[i] = l.Distance(p)
	}
	return out
}

// LineRMSE returns the root-mean-square perpendicular distance from the points
// to the line. It returns 0 for an empty point set.
func LineRMSE(l Line, pts []cv.Point2f) float64 {
	if len(pts) == 0 {
		return 0
	}
	var s float64
	for _, p := range pts {
		d := l.SignedDistance(p)
		s += d * d
	}
	return math.Sqrt(s / float64(len(pts)))
}
