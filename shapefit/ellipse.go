package shapefit

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Ellipse is an ellipse described by its center, semi-axis lengths and
// orientation. SemiMajor is always at least SemiMinor. Angle is the rotation of
// the major axis from the positive x-axis, in radians, in the range
// (-π/2, π/2].
type Ellipse struct {
	// Center is the ellipse center in image coordinates.
	Center cv.Point2f
	// SemiMajor is the larger semi-axis length.
	SemiMajor float64
	// SemiMinor is the smaller semi-axis length.
	SemiMinor float64
	// Angle is the major-axis orientation in radians.
	Angle float64
}

// Area returns the area π·a·b enclosed by the ellipse.
func (e Ellipse) Area() float64 { return math.Pi * e.SemiMajor * e.SemiMinor }

// Eccentricity returns the ellipse eccentricity in [0, 1); 0 is a circle.
func (e Ellipse) Eccentricity() float64 {
	if e.SemiMajor < shapefitEps {
		return 0
	}
	r := e.SemiMinor / e.SemiMajor
	v := 1 - r*r
	if v < 0 {
		v = 0
	}
	return math.Sqrt(v)
}

// PointAt returns the point on the ellipse at parametric angle t (radians): the
// point (a·cos t, b·sin t) in the ellipse's own frame, rotated by Angle and
// translated to Center.
func (e Ellipse) PointAt(t float64) cv.Point2f {
	ca := math.Cos(e.Angle)
	sa := math.Sin(e.Angle)
	x := e.SemiMajor * math.Cos(t)
	y := e.SemiMinor * math.Sin(t)
	return cv.Point2f{
		X: e.Center.X + x*ca - y*sa,
		Y: e.Center.Y + x*sa + y*ca,
	}
}

// Contains reports whether p lies inside or on the ellipse.
func (e Ellipse) Contains(p cv.Point2f) bool {
	if e.SemiMajor < shapefitEps || e.SemiMinor < shapefitEps {
		return false
	}
	ca := math.Cos(e.Angle)
	sa := math.Sin(e.Angle)
	dx := p.X - e.Center.X
	dy := p.Y - e.Center.Y
	// Rotate into the ellipse frame.
	u := dx*ca + dy*sa
	v := -dx*sa + dy*ca
	uu := u / e.SemiMajor
	vv := v / e.SemiMinor
	return uu*uu+vv*vv <= 1+1e-9
}

// Foci returns the two focal points of the ellipse.
func (e Ellipse) Foci() (cv.Point2f, cv.Point2f) {
	cval := math.Sqrt(math.Max(0, e.SemiMajor*e.SemiMajor-e.SemiMinor*e.SemiMinor))
	ca := math.Cos(e.Angle)
	sa := math.Sin(e.Angle)
	f1 := cv.Point2f{X: e.Center.X + cval*ca, Y: e.Center.Y + cval*sa}
	f2 := cv.Point2f{X: e.Center.X - cval*ca, Y: e.Center.Y - cval*sa}
	return f1, f2
}

// RotatedRect converts the ellipse to the parent library's
// [github.com/malcolmston/opencv.RotatedRect], whose Width and Height are the
// full axis lengths (twice the semi-axes) and whose Angle is in degrees. This
// makes fitted ellipses drawable with the parent library's ellipse rendering.
func (e Ellipse) RotatedRect() cv.RotatedRect {
	return cv.RotatedRect{
		CenterX: e.Center.X,
		CenterY: e.Center.Y,
		Width:   2 * e.SemiMajor,
		Height:  2 * e.SemiMinor,
		Angle:   e.Angle * 180 / math.Pi,
	}
}

// EllipseFromRotatedRect builds an [Ellipse] from a
// [github.com/malcolmston/opencv.RotatedRect], the inverse of
// [Ellipse.RotatedRect]. The larger of the rectangle's sides becomes the major
// axis, so the resulting Angle may differ from the rectangle's by 90°.
func EllipseFromRotatedRect(r cv.RotatedRect) Ellipse {
	a := r.Width / 2
	b := r.Height / 2
	ang := r.Angle * math.Pi / 180
	if b > a {
		a, b = b, a
		ang += math.Pi / 2
	}
	return Ellipse{
		Center:    cv.Point2f{X: r.CenterX, Y: r.CenterY},
		SemiMajor: a,
		SemiMinor: b,
		Angle:     shapefitWrapPi(ang),
	}
}

// Conic returns the coefficients (A, B, C, D, E, F) of the implicit conic
// A·x² + B·x·y + C·y² + D·x + E·y + F = 0 that represents the ellipse. The
// coefficients are scaled so that the quadratic part matches the ellipse's
// unit-level set.
func (e Ellipse) Conic() (A, B, C, D, E, F float64) {
	ca := math.Cos(e.Angle)
	sa := math.Sin(e.Angle)
	a2 := e.SemiMajor * e.SemiMajor
	b2 := e.SemiMinor * e.SemiMinor
	if a2 < shapefitEps || b2 < shapefitEps {
		return 0, 0, 0, 0, 0, 0
	}
	// Quadratic form M = R·diag(1/a², 1/b²)·Rᵀ.
	A = ca*ca/a2 + sa*sa/b2
	C = sa*sa/a2 + ca*ca/b2
	B = 2 * ca * sa * (1/a2 - 1/b2)
	cx := e.Center.X
	cy := e.Center.Y
	D = -(2*A*cx + B*cy)
	E = -(2*C*cy + B*cx)
	F = A*cx*cx + B*cx*cy + C*cy*cy - 1
	return
}

// EllipseFromConic converts the implicit conic
// A·x² + B·x·y + C·y² + D·x + E·y + F = 0 to a geometric [Ellipse]. It reports
// false when the conic does not describe a real ellipse (for example a
// parabola, hyperbola or degenerate/imaginary conic).
func EllipseFromConic(A, B, C, D, E, F float64) (Ellipse, bool) {
	disc := B*B - 4*A*C
	if disc >= -shapefitEps {
		return Ellipse{}, false // not an ellipse
	}
	x0 := (2*C*D - B*E) / disc
	y0 := (2*A*E - B*D) / disc
	// Constant of the conic translated to its center.
	k := A*x0*x0 + B*x0*y0 + C*y0*y0 + D*x0 + E*y0 + F
	// Eigen-decompose the quadratic form [[A, B/2], [B/2, C]].
	l1, v1, l2, v2 := shapefitSym2Eig(A, B/2, C)
	if math.Abs(l1) < shapefitEps || math.Abs(l2) < shapefitEps {
		return Ellipse{}, false
	}
	// Translated ellipse: l1·u² + l2·v² = -k. Semi-axis along eig i is
	// sqrt(-k / li); both must be positive for a real ellipse.
	s1 := -k / l1
	s2 := -k / l2
	if s1 <= 0 || s2 <= 0 {
		return Ellipse{}, false
	}
	ax1 := math.Sqrt(s1)
	ax2 := math.Sqrt(s2)
	var major, minor float64
	var dir [2]float64
	if ax1 >= ax2 {
		major, minor = ax1, ax2
		dir = v1
	} else {
		major, minor = ax2, ax1
		dir = v2
	}
	ang := shapefitWrapPi(math.Atan2(dir[1], dir[0]))
	return Ellipse{
		Center:    cv.Point2f{X: x0, Y: y0},
		SemiMajor: major,
		SemiMinor: minor,
		Angle:     ang,
	}, true
}

// FitEllipse fits an ellipse to the points using the Halíř–Flusser numerically
// stable formulation of Fitzgibbon's direct least-squares method, which
// guarantees an elliptical (rather than hyperbolic) solution. It returns an
// error when fewer than five points are supplied or the configuration is
// degenerate (for example collinear points, which admit no ellipse).
//
// Points are internally centered and scaled for conditioning; the result is
// mapped back to the input coordinate frame.
func FitEllipse(pts []cv.Point2f) (Ellipse, error) {
	if len(pts) < 5 {
		return Ellipse{}, errors.New("shapefit: FitEllipse needs at least 5 points")
	}
	c := Centroid(pts)
	// Isotropic scaling to unit mean distance improves conditioning.
	var meanDist float64
	for _, p := range pts {
		meanDist += math.Hypot(p.X-c.X, p.Y-c.Y)
	}
	meanDist /= float64(len(pts))
	if meanDist < shapefitEps {
		return Ellipse{}, errors.New("shapefit: FitEllipse points are coincident")
	}
	s := meanDist
	// Accumulate the scatter blocks S1 (quadratic), S2 (mixed), S3 (linear).
	var s1, s2, s3 [3][3]float64
	for _, p := range pts {
		u := (p.X - c.X) / s
		v := (p.Y - c.Y) / s
		d1 := [3]float64{u * u, u * v, v * v}
		d2 := [3]float64{u, v, 1}
		for i := 0; i < 3; i++ {
			for j := 0; j < 3; j++ {
				s1[i][j] += d1[i] * d1[j]
				s2[i][j] += d1[i] * d2[j]
				s3[i][j] += d2[i] * d2[j]
			}
		}
	}
	invS3, ok := shapefitInv3(s3)
	if !ok {
		return Ellipse{}, errors.New("shapefit: FitEllipse degenerate configuration")
	}
	// T = -inv(S3)·S2ᵀ ; M = S1 + S2·T ; reduced M' = C1⁻¹·M.
	t := shapefitMul3(invS3, shapefitTranspose3(s2))
	for i := range t {
		for j := range t[i] {
			t[i][j] = -t[i][j]
		}
	}
	m := shapefitMul3(s2, t)
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			m[i][j] += s1[i][j]
		}
	}
	// Apply C1⁻¹ = [[0,0,0.5],[0,-1,0],[0.5,0,0]] on the left.
	var mr [3][3]float64
	for j := 0; j < 3; j++ {
		mr[0][j] = 0.5 * m[2][j]
		mr[1][j] = -m[1][j]
		mr[2][j] = 0.5 * m[0][j]
	}
	// Eigen-decompose the general 3×3 matrix mr and pick the eigenvector that
	// satisfies the ellipse constraint 4·a·c - b² > 0.
	trace := mr[0][0] + mr[1][1] + mr[2][2]
	minors := mr[0][0]*mr[1][1] - mr[0][1]*mr[1][0] +
		mr[0][0]*mr[2][2] - mr[0][2]*mr[2][0] +
		mr[1][1]*mr[2][2] - mr[1][2]*mr[2][1]
	detM := mr[0][0]*(mr[1][1]*mr[2][2]-mr[1][2]*mr[2][1]) -
		mr[0][1]*(mr[1][0]*mr[2][2]-mr[1][2]*mr[2][0]) +
		mr[0][2]*(mr[1][0]*mr[2][1]-mr[1][1]*mr[2][0])
	// Characteristic poly: λ³ - trace·λ² + minors·λ - detM = 0.
	roots := shapefitCubicRoots(-trace, minors, -detM)
	var a1 [3]float64
	found := false
	for _, lam := range roots {
		nm := [3][3]float64{
			{mr[0][0] - lam, mr[0][1], mr[0][2]},
			{mr[1][0], mr[1][1] - lam, mr[1][2]},
			{mr[2][0], mr[2][1], mr[2][2] - lam},
		}
		v := shapefitNull3(nm)
		if 4*v[0]*v[2]-v[1]*v[1] > 0 {
			a1 = v
			found = true
			break
		}
	}
	if !found {
		return Ellipse{}, errors.New("shapefit: FitEllipse found no elliptical solution")
	}
	a2 := shapefitMatVec3(t, a1)
	// Conic in the normalized frame.
	nA, nB, nC := a1[0], a1[1], a1[2]
	nD, nE, nF := a2[0], a2[1], a2[2]
	e, ok := EllipseFromConic(nA, nB, nC, nD, nE, nF)
	if !ok {
		return Ellipse{}, errors.New("shapefit: FitEllipse produced a non-elliptical conic")
	}
	// Undo the normalization: scale by s and translate by the centroid. Axes
	// and angle scale/translate as follows (angle is invariant).
	e.Center = cv.Point2f{X: e.Center.X*s + c.X, Y: e.Center.Y*s + c.Y}
	e.SemiMajor *= s
	e.SemiMinor *= s
	return e, nil
}

// EllipseAlgebraicResiduals returns, for each point, the absolute value of the
// normalized conic polynomial |A·x²+B·x·y+C·y²+D·x+E·y+F| evaluated at the
// point. This algebraic residual is fast to compute and monotonic in the
// geometric distance for points near the ellipse, making it useful as a RANSAC
// scoring function.
func EllipseAlgebraicResiduals(e Ellipse, pts []cv.Point2f) []float64 {
	A, B, C, D, E, F := e.Conic()
	out := make([]float64, len(pts))
	for i, p := range pts {
		out[i] = math.Abs(A*p.X*p.X + B*p.X*p.Y + C*p.Y*p.Y + D*p.X + E*p.Y + F)
	}
	return out
}
