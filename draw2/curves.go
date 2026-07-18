package draw2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// PointF is a floating-point 2-D coordinate used by the curve routines for
// sub-pixel control points.
type PointF struct {
	// X is the horizontal coordinate (column).
	X float64
	// Y is the vertical coordinate (row).
	Y float64
}

// Add returns the component-wise sum of p and q.
func (p PointF) Add(q PointF) PointF { return PointF{p.X + q.X, p.Y + q.Y} }

// Sub returns the component-wise difference p minus q.
func (p PointF) Sub(q PointF) PointF { return PointF{p.X - q.X, p.Y - q.Y} }

// Scale returns p with both components multiplied by s.
func (p PointF) Scale(s float64) PointF { return PointF{p.X * s, p.Y * s} }

// ToPoint rounds p to the nearest integer [github.com/malcolmston/opencv.Point].
func (p PointF) ToPoint() cv.Point { return cv.Point{X: draw2round(p.X), Y: draw2round(p.Y)} }

// PointFOf converts an integer Point into a PointF.
func PointFOf(p cv.Point) PointF { return PointF{float64(p.X), float64(p.Y)} }

// QuadBezierPoint evaluates a quadratic Bezier curve with control points p0,
// p1 and p2 at parameter t in [0,1] and returns the resulting position.
func QuadBezierPoint(p0, p1, p2 PointF, t float64) PointF {
	u := 1 - t
	a := u * u
	b := 2 * u * t
	c := t * t
	return PointF{
		X: a*p0.X + b*p1.X + c*p2.X,
		Y: a*p0.Y + b*p1.Y + c*p2.Y,
	}
}

// CubicBezierPoint evaluates a cubic Bezier curve with control points p0, p1,
// p2 and p3 at parameter t in [0,1] and returns the resulting position.
func CubicBezierPoint(p0, p1, p2, p3 PointF, t float64) PointF {
	u := 1 - t
	a := u * u * u
	b := 3 * u * u * t
	c := 3 * u * t * t
	d := t * t * t
	return PointF{
		X: a*p0.X + b*p1.X + c*p2.X + d*p3.X,
		Y: a*p0.Y + b*p1.Y + c*p2.Y + d*p3.Y,
	}
}

// QuadraticBezier draws a quadratic Bezier curve through control points p0, p1
// and p2. steps sets the number of line segments used to approximate the
// curve (a value <= 0 selects an automatic count based on the control-point
// spread). thickness controls stroke width.
func QuadraticBezier(m *cv.Mat, p0, p1, p2 PointF, color cv.Scalar, thickness, steps int) {
	if steps <= 0 {
		steps = draw2curveSteps(p0, p1) + draw2curveSteps(p1, p2)
	}
	prev := p0.ToPoint()
	for i := 1; i <= steps; i++ {
		t := float64(i) / float64(steps)
		cur := QuadBezierPoint(p0, p1, p2, t).ToPoint()
		Line(m, prev, cur, color, thickness)
		prev = cur
	}
}

// CubicBezier draws a cubic Bezier curve through control points p0, p1, p2 and
// p3. steps sets the number of line segments used to approximate the curve (a
// value <= 0 selects an automatic count). thickness controls stroke width.
func CubicBezier(m *cv.Mat, p0, p1, p2, p3 PointF, color cv.Scalar, thickness, steps int) {
	if steps <= 0 {
		steps = draw2curveSteps(p0, p1) + draw2curveSteps(p1, p2) + draw2curveSteps(p2, p3)
	}
	prev := p0.ToPoint()
	for i := 1; i <= steps; i++ {
		t := float64(i) / float64(steps)
		cur := CubicBezierPoint(p0, p1, p2, p3, t).ToPoint()
		Line(m, prev, cur, color, thickness)
		prev = cur
	}
}

// CatmullRomSpline draws a smooth Catmull-Rom spline passing through pts. The
// curve interpolates every point (unlike a Bezier's control polygon). When
// closed is true the spline forms a closed loop. stepsPerSegment sets the
// tessellation per span (clamped to a minimum of 1) and thickness controls
// stroke width. At least two points are required.
func CatmullRomSpline(m *cv.Mat, pts []PointF, closed bool, color cv.Scalar, thickness, stepsPerSegment int) {
	n := len(pts)
	if n < 2 {
		return
	}
	if stepsPerSegment < 1 {
		stepsPerSegment = 1
	}
	get := func(i int) PointF {
		if closed {
			return pts[((i%n)+n)%n]
		}
		if i < 0 {
			i = 0
		}
		if i >= n {
			i = n - 1
		}
		return pts[i]
	}
	last := n - 1
	if closed {
		last = n
	}
	var prev cv.Point
	first := true
	for i := 0; i < last; i++ {
		p0 := get(i - 1)
		p1 := get(i)
		p2 := get(i + 1)
		p3 := get(i + 2)
		for s := 0; s <= stepsPerSegment; s++ {
			t := float64(s) / float64(stepsPerSegment)
			cur := draw2catmullRom(p0, p1, p2, p3, t).ToPoint()
			if first {
				prev = cur
				first = false
				continue
			}
			Line(m, prev, cur, color, thickness)
			prev = cur
		}
	}
}

// draw2catmullRom evaluates the uniform Catmull-Rom basis for the segment
// between p1 and p2 (with neighbours p0 and p3) at parameter t in [0,1].
func draw2catmullRom(p0, p1, p2, p3 PointF, t float64) PointF {
	t2 := t * t
	t3 := t2 * t
	f := func(a0, a1, a2, a3 float64) float64 {
		return 0.5 * ((2 * a1) +
			(-a0+a2)*t +
			(2*a0-5*a1+4*a2-a3)*t2 +
			(-a0+3*a1-3*a2+a3)*t3)
	}
	return PointF{
		X: f(p0.X, p1.X, p2.X, p3.X),
		Y: f(p0.Y, p1.Y, p2.Y, p3.Y),
	}
}

// draw2curveSteps estimates a reasonable tessellation count for the span
// between two control points based on their separation.
func draw2curveSteps(a, b PointF) int {
	d := math.Hypot(b.X-a.X, b.Y-a.Y)
	s := int(d / 3)
	if s < 4 {
		s = 4
	}
	return s
}
