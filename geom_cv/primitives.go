package geom_cv

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// geom_cvEps is the tolerance used throughout the package to decide whether a
// floating-point quantity should be treated as zero (collinearity, degenerate
// triangles, points lying on a boundary, and similar tests).
const geom_cvEps = 1e-9

// Add returns the component-wise sum a+b of two vectors.
func Add(a, b cv.Point2f) cv.Point2f {
	return cv.Point2f{X: a.X + b.X, Y: a.Y + b.Y}
}

// Sub returns the component-wise difference a-b of two vectors.
func Sub(a, b cv.Point2f) cv.Point2f {
	return cv.Point2f{X: a.X - b.X, Y: a.Y - b.Y}
}

// Scale returns the vector a multiplied by the scalar s.
func Scale(a cv.Point2f, s float64) cv.Point2f {
	return cv.Point2f{X: a.X * s, Y: a.Y * s}
}

// Dot returns the dot product a·b = a.X*b.X + a.Y*b.Y.
func Dot(a, b cv.Point2f) float64 {
	return a.X*b.X + a.Y*b.Y
}

// Cross returns the scalar 2-D cross product a×b = a.X*b.Y - a.Y*b.X. Its sign
// tells whether b lies counter-clockwise (positive) or clockwise (negative)
// from a in a right-handed frame.
func Cross(a, b cv.Point2f) float64 {
	return a.X*b.Y - a.Y*b.X
}

// Norm returns the Euclidean length (L2 norm) of the vector a.
func Norm(a cv.Point2f) float64 {
	return math.Hypot(a.X, a.Y)
}

// Normalize returns a unit-length vector pointing in the same direction as a.
// If a has (near) zero length the zero vector is returned unchanged.
func Normalize(a cv.Point2f) cv.Point2f {
	n := Norm(a)
	if n < geom_cvEps {
		return cv.Point2f{}
	}
	return cv.Point2f{X: a.X / n, Y: a.Y / n}
}

// Perpendicular returns the vector a rotated 90 degrees counter-clockwise in a
// right-handed frame, i.e. (-a.Y, a.X).
func Perpendicular(a cv.Point2f) cv.Point2f {
	return cv.Point2f{X: -a.Y, Y: a.X}
}

// Distance returns the Euclidean distance between points a and b.
func Distance(a, b cv.Point2f) float64 {
	return math.Hypot(a.X-b.X, a.Y-b.Y)
}

// DistanceSquared returns the squared Euclidean distance between a and b. It is
// cheaper than [Distance] and preserves ordering, so prefer it when only
// comparing distances.
func DistanceSquared(a, b cv.Point2f) float64 {
	dx := a.X - b.X
	dy := a.Y - b.Y
	return dx*dx + dy*dy
}

// Midpoint returns the point halfway between a and b.
func Midpoint(a, b cv.Point2f) cv.Point2f {
	return cv.Point2f{X: (a.X + b.X) / 2, Y: (a.Y + b.Y) / 2}
}

// Lerp returns the linear interpolation (1-t)*a + t*b. With t=0 it returns a
// and with t=1 it returns b; values outside [0,1] extrapolate.
func Lerp(a, b cv.Point2f, t float64) cv.Point2f {
	return cv.Point2f{X: a.X + (b.X-a.X)*t, Y: a.Y + (b.Y-a.Y)*t}
}

// Orientation classifies the turn made going from a to b to c. It returns +1
// when a→b→c turns counter-clockwise, -1 when it turns clockwise and 0 when the
// three points are collinear, all in a right-handed frame.
func Orientation(a, b, c cv.Point2f) int {
	v := Cross(Sub(b, a), Sub(c, a))
	if v > geom_cvEps {
		return 1
	}
	if v < -geom_cvEps {
		return -1
	}
	return 0
}

// Collinear reports whether the three points a, b and c lie on a common line
// within the package tolerance.
func Collinear(a, b, c cv.Point2f) bool {
	return Orientation(a, b, c) == 0
}

// VectorAngle returns the direction of the vector a as an angle in radians in
// the range (-π, π], measured with [math.Atan2] (positive toward +Y).
func VectorAngle(a cv.Point2f) float64 {
	return math.Atan2(a.Y, a.X)
}

// AngleBetween returns the unsigned angle in radians, in [0, π], between the
// vectors a and b. It returns 0 if either vector has (near) zero length.
func AngleBetween(a, b cv.Point2f) float64 {
	na, nb := Norm(a), Norm(b)
	if na < geom_cvEps || nb < geom_cvEps {
		return 0
	}
	c := Dot(a, b) / (na * nb)
	if c > 1 {
		c = 1
	} else if c < -1 {
		c = -1
	}
	return math.Acos(c)
}

// ToPoint rounds a floating-point point to the nearest integer pixel and
// returns it as the parent library's [github.com/malcolmston/opencv.Point].
func ToPoint(p cv.Point2f) cv.Point {
	return cv.Point{X: int(math.Round(p.X)), Y: int(math.Round(p.Y))}
}

// ToPoint2f converts an integer pixel to a floating-point point.
func ToPoint2f(p cv.Point) cv.Point2f {
	return cv.Point2f{X: float64(p.X), Y: float64(p.Y)}
}
