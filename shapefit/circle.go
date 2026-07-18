package shapefit

import (
	"errors"
	"math"

	cv "github.com/malcolmston/opencv"
)

// Circle is a circle described by its center and radius.
type Circle struct {
	// Center is the circle center in image coordinates.
	Center cv.Point2f
	// Radius is the circle radius; it is non-negative for a valid circle.
	Radius float64
}

// Distance returns the unsigned radial residual of p with respect to the
// circle: the absolute difference between the distance from p to the center and
// the radius. It is zero for points exactly on the circle.
func (c Circle) Distance(p cv.Point2f) float64 {
	d := math.Hypot(p.X-c.Center.X, p.Y-c.Center.Y)
	return math.Abs(d - c.Radius)
}

// Contains reports whether p lies inside or on the circle.
func (c Circle) Contains(p cv.Point2f) bool {
	return math.Hypot(p.X-c.Center.X, p.Y-c.Center.Y) <= c.Radius+shapefitEps
}

// Area returns the area π·r² enclosed by the circle.
func (c Circle) Area() float64 { return math.Pi * c.Radius * c.Radius }

// Circumference returns the perimeter 2·π·r of the circle.
func (c Circle) Circumference() float64 { return 2 * math.Pi * c.Radius }

// PointAt returns the point on the circle at angle theta (radians), measured
// counter-clockwise from the positive x-axis.
func (c Circle) PointAt(theta float64) cv.Point2f {
	return cv.Point2f{
		X: c.Center.X + c.Radius*math.Cos(theta),
		Y: c.Center.Y + c.Radius*math.Sin(theta),
	}
}

// FitCircle fits a circle to the points using the Kåsa algebraic method, a
// linear least-squares fit that minimizes the algebraic residual
// (x²+y²) + D·x + E·y + F. It is fast and closed-form but slightly biased
// toward smaller radii on sparse arcs; use [FitCircleTaubin] when that bias
// matters. It returns an error when fewer than three points are supplied or the
// points are collinear.
func FitCircle(pts []cv.Point2f) (Circle, error) {
	if len(pts) < 3 {
		return Circle{}, errors.New("shapefit: FitCircle needs at least 3 points")
	}
	c := Centroid(pts)
	// Solve the normal equations in coordinates centered on the centroid for
	// better conditioning; the center shifts back afterward.
	var suu, suv, svv, suuu, svvv, suvv, svuu float64
	for _, p := range pts {
		u := p.X - c.X
		v := p.Y - c.Y
		suu += u * u
		svv += v * v
		suv += u * v
		suuu += u * u * u
		svvv += v * v * v
		suvv += u * v * v
		svuu += v * u * u
	}
	// The linear system for the centered circle center (uc, vc):
	//   [suu suv][uc]   0.5[suuu+suvv]
	//   [suv svv][vc] =    [svvv+svuu]
	det := suu*svv - suv*suv
	if math.Abs(det) < shapefitEps {
		return Circle{}, errors.New("shapefit: FitCircle points are collinear")
	}
	b0 := 0.5 * (suuu + suvv)
	b1 := 0.5 * (svvv + svuu)
	uc := (b0*svv - b1*suv) / det
	vc := (suu*b1 - suv*b0) / det
	r := math.Sqrt(uc*uc + vc*vc + (suu+svv)/float64(len(pts)))
	return Circle{Center: cv.Point2f{X: uc + c.X, Y: vc + c.Y}, Radius: r}, nil
}

// FitCircleTaubin fits a circle using Taubin's method, which minimizes a
// gradient-weighted algebraic distance and largely removes the small-radius
// bias of the plain algebraic fit, giving results close to the geometric
// (orthogonal-distance) optimum in a single closed-form pass. It returns an
// error when fewer than three points are supplied or the points are collinear.
func FitCircleTaubin(pts []cv.Point2f) (Circle, error) {
	if len(pts) < 3 {
		return Circle{}, errors.New("shapefit: FitCircleTaubin needs at least 3 points")
	}
	c := Centroid(pts)
	n := float64(len(pts))
	var mxx, myy, mxy, mxz, myz, mzz float64
	for _, p := range pts {
		u := p.X - c.X
		v := p.Y - c.Y
		z := u*u + v*v
		mxx += u * u
		myy += v * v
		mxy += u * v
		mxz += u * z
		myz += v * z
		mzz += z * z
	}
	mxx /= n
	myy /= n
	mxy /= n
	mxz /= n
	myz /= n
	mzz /= n
	mz := mxx + myy
	covXY := mxx*myy - mxy*mxy
	varZ := mzz - mz*mz
	a3 := 4 * mz
	a2 := -3*mz*mz - mzz
	a1 := varZ*mz + 4*covXY*mz - mxz*mxz - myz*myz
	a0 := mxz*(mxz*myy-myz*mxy) + myz*(myz*mxx-mxz*mxy) - varZ*covXY
	a22 := a2 + a2
	a33 := a3 + a3 + a3
	// Newton's method from x = 0 for the smallest positive root.
	x := 0.0
	y := a0
	for iter := 0; iter < 99; iter++ {
		dy := a1 + x*(a22+x*a33)
		if math.Abs(dy) < shapefitEps {
			break
		}
		xnew := x - y/dy
		if xnew == x {
			break
		}
		ynew := a0 + xnew*(a1+xnew*(a2+xnew*a3))
		if math.Abs(ynew) >= math.Abs(y) && iter > 0 {
			break
		}
		x = xnew
		y = ynew
		if math.Abs(y) < shapefitEps {
			break
		}
	}
	det := x*x - x*mz + covXY
	if math.Abs(det) < shapefitEps {
		return Circle{}, errors.New("shapefit: FitCircleTaubin points are collinear")
	}
	xc := (mxz*(myy-x) - myz*mxy) / det / 2
	yc := (myz*(mxx-x) - mxz*mxy) / det / 2
	r := math.Sqrt(xc*xc + yc*yc + mz)
	return Circle{Center: cv.Point2f{X: xc + c.X, Y: yc + c.Y}, Radius: r}, nil
}

// CircleResiduals returns the radial residual of each point with respect to the
// circle, in input order.
func CircleResiduals(c Circle, pts []cv.Point2f) []float64 {
	out := make([]float64, len(pts))
	for i, p := range pts {
		out[i] = c.Distance(p)
	}
	return out
}

// CircleRMSE returns the root-mean-square radial residual of the points with
// respect to the circle. It returns 0 for an empty point set.
func CircleRMSE(c Circle, pts []cv.Point2f) float64 {
	if len(pts) == 0 {
		return 0
	}
	var s float64
	for _, p := range pts {
		d := c.Distance(p)
		s += d * d
	}
	return math.Sqrt(s / float64(len(pts)))
}
