package contours2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// MinAreaRect returns the minimum-area rotated rectangle enclosing the points,
// found by rotating calipers over the convex hull: for each hull edge the
// bounding box in that edge's frame is measured and the smallest-area result is
// kept. It panics on an empty point set. This mirrors OpenCV's minAreaRect.
func MinAreaRect(pts []cv.Point) RotatedRect {
	if len(pts) == 0 {
		panic("contours2: MinAreaRect on empty point set")
	}
	hull := ConvexHull(pts, false)
	if len(hull) == 1 {
		return RotatedRect{CenterX: float64(hull[0].X), CenterY: float64(hull[0].Y)}
	}
	if len(hull) == 2 {
		dx := float64(hull[1].X - hull[0].X)
		dy := float64(hull[1].Y - hull[0].Y)
		return RotatedRect{
			CenterX: float64(hull[0].X+hull[1].X) / 2,
			CenterY: float64(hull[0].Y+hull[1].Y) / 2,
			Width:   math.Hypot(dx, dy),
			Height:  0,
			Angle:   math.Atan2(dy, dx) * 180 / math.Pi,
		}
	}

	n := len(hull)
	bestArea := math.Inf(1)
	var best RotatedRect
	for i := 0; i < n; i++ {
		a := hull[i]
		b := hull[(i+1)%n]
		ex := float64(b.X - a.X)
		ey := float64(b.Y - a.Y)
		el := math.Hypot(ex, ey)
		if el == 0 {
			continue
		}
		ux, uy := ex/el, ey/el // edge unit direction
		nx, ny := -uy, ux      // perpendicular unit
		minU, maxU := math.Inf(1), math.Inf(-1)
		minV, maxV := math.Inf(1), math.Inf(-1)
		for _, p := range hull {
			px, py := float64(p.X), float64(p.Y)
			pu := px*ux + py*uy
			pv := px*nx + py*ny
			minU = math.Min(minU, pu)
			maxU = math.Max(maxU, pu)
			minV = math.Min(minV, pv)
			maxV = math.Max(maxV, pv)
		}
		w := maxU - minU
		h := maxV - minV
		area := w * h
		if area < bestArea {
			bestArea = area
			cu := (minU + maxU) / 2
			cvv := (minV + maxV) / 2
			cx := cu*ux + cvv*nx
			cy := cu*uy + cvv*ny
			best = RotatedRect{
				CenterX: cx,
				CenterY: cy,
				Width:   w,
				Height:  h,
				Angle:   math.Atan2(uy, ux) * 180 / math.Pi,
			}
		}
	}
	return best
}

// contours2circle is a helper record for the minimum enclosing circle search.
type contours2circle struct {
	cx, cy, r float64
}

func (c contours2circle) contains(p cv.Point2f) bool {
	return math.Hypot(p.X-c.cx, p.Y-c.cy) <= c.r+1e-9
}

// MinEnclosingCircle returns the smallest circle that encloses every input
// point, computed with Welzl's algorithm evaluated deterministically over the
// given order. It panics on an empty point set. This mirrors OpenCV's
// minEnclosingCircle.
func MinEnclosingCircle(pts []cv.Point) Circle {
	if len(pts) == 0 {
		panic("contours2: MinEnclosingCircle on empty point set")
	}
	ps := make([]cv.Point2f, len(pts))
	for i, p := range pts {
		ps[i] = cv.Point2f{X: float64(p.X), Y: float64(p.Y)}
	}
	c := contours2welzl(ps, nil)
	return Circle{Center: cv.Point2f{X: c.cx, Y: c.cy}, Radius: c.r}
}

// contours2welzl computes the minimum enclosing circle of points not yet placed
// on the boundary, given up to three boundary points.
func contours2welzl(pts []cv.Point2f, boundary []cv.Point2f) contours2circle {
	if len(pts) == 0 || len(boundary) == 3 {
		return contours2trivialCircle(boundary)
	}
	p := pts[len(pts)-1]
	rest := pts[:len(pts)-1]
	c := contours2welzl(rest, boundary)
	if c.contains(p) {
		return c
	}
	return contours2welzl(rest, append(append([]cv.Point2f{}, boundary...), p))
}

// contours2trivialCircle solves the enclosing circle for zero to three boundary
// points.
func contours2trivialCircle(b []cv.Point2f) contours2circle {
	switch len(b) {
	case 0:
		return contours2circle{}
	case 1:
		return contours2circle{cx: b[0].X, cy: b[0].Y}
	case 2:
		return contours2circleFrom2(b[0], b[1])
	default:
		c, ok := contours2circleFrom3(b[0], b[1], b[2])
		if !ok {
			best := contours2circleFrom2(b[0], b[1])
			for _, pair := range [][2]cv.Point2f{{b[0], b[2]}, {b[1], b[2]}} {
				cc := contours2circleFrom2(pair[0], pair[1])
				if cc.r > best.r {
					best = cc
				}
			}
			return best
		}
		return c
	}
}

func contours2circleFrom2(a, b cv.Point2f) contours2circle {
	cx := (a.X + b.X) / 2
	cy := (a.Y + b.Y) / 2
	return contours2circle{cx: cx, cy: cy, r: math.Hypot(a.X-cx, a.Y-cy)}
}

func contours2circleFrom3(a, b, c cv.Point2f) (contours2circle, bool) {
	d := 2 * (a.X*(b.Y-c.Y) + b.X*(c.Y-a.Y) + c.X*(a.Y-b.Y))
	if math.Abs(d) < 1e-12 {
		return contours2circle{}, false
	}
	a2 := a.X*a.X + a.Y*a.Y
	b2 := b.X*b.X + b.Y*b.Y
	c2 := c.X*c.X + c.Y*c.Y
	ux := (a2*(b.Y-c.Y) + b2*(c.Y-a.Y) + c2*(a.Y-b.Y)) / d
	uy := (a2*(c.X-b.X) + b2*(a.X-c.X) + c2*(b.X-a.X)) / d
	return contours2circle{cx: ux, cy: uy, r: math.Hypot(a.X-ux, a.Y-uy)}, true
}

// FitLine fits a line to a set of points by total least squares (minimising the
// orthogonal distance) and returns a unit direction (vx, vy) together with a
// point (x0, y0) on the line, matching the parameterisation of OpenCV's fitLine
// with DIST_L2. It panics on fewer than two points.
func FitLine(pts []cv.Point) (vx, vy, x0, y0 float64) {
	n := len(pts)
	if n < 2 {
		panic("contours2: FitLine requires at least two points")
	}
	var mx, my float64
	for _, p := range pts {
		mx += float64(p.X)
		my += float64(p.Y)
	}
	mx /= float64(n)
	my /= float64(n)
	var sxx, syy, sxy float64
	for _, p := range pts {
		dx := float64(p.X) - mx
		dy := float64(p.Y) - my
		sxx += dx * dx
		syy += dy * dy
		sxy += dx * dy
	}
	theta := 0.5 * math.Atan2(2*sxy, sxx-syy)
	vx = math.Cos(theta)
	vy = math.Sin(theta)
	return vx, vy, mx, my
}

// FitEllipse fits an ellipse to a set of points and returns it in OpenCV's
// [Ellipse] parameterisation (centre, full axis lengths and rotation in
// degrees). The fit uses the second-order central moments of the point set:
// the ellipse is centred at the centroid, oriented along the principal
// eigenvector of the covariance matrix, and scaled so that its second moments
// match those of a uniform elliptical disc. For points sampled from a genuine
// ellipse this recovers the centre and orientation exactly and the axis lengths
// up to the usual boundary-vs-area scaling. It panics on fewer than five points
// (an ellipse has five degrees of freedom). This is analogous to OpenCV's
// fitEllipse for well-conditioned inputs.
func FitEllipse(pts []cv.Point) Ellipse {
	n := len(pts)
	if n < 5 {
		panic("contours2: FitEllipse requires at least five points")
	}
	var mx, my float64
	for _, p := range pts {
		mx += float64(p.X)
		my += float64(p.Y)
	}
	mx /= float64(n)
	my /= float64(n)
	var cxx, cyy, cxy float64
	for _, p := range pts {
		dx := float64(p.X) - mx
		dy := float64(p.Y) - my
		cxx += dx * dx
		cyy += dy * dy
		cxy += dx * dy
	}
	cxx /= float64(n)
	cyy /= float64(n)
	cxy /= float64(n)

	// Eigenvalues/vectors of the symmetric 2x2 covariance matrix.
	tr := cxx + cyy
	det := cxx*cyy - cxy*cxy
	disc := math.Sqrt(math.Max(0, tr*tr/4-det))
	l1 := tr/2 + disc // larger eigenvalue
	l2 := tr/2 - disc // smaller eigenvalue

	// Orientation of the major axis (eigenvector of l1).
	var angle float64
	if cxy == 0 {
		if cxx >= cyy {
			angle = 0
		} else {
			angle = 90
		}
	} else {
		angle = math.Atan2(l1-cxx, cxy) * 180 / math.Pi
	}

	// For a uniform elliptical disc the second central moment along a semi-axis
	// of length s is s^2/4, so s = 2*sqrt(eigenvalue); the full axis is 2*s.
	major := 4 * math.Sqrt(math.Max(0, l1))
	minor := 4 * math.Sqrt(math.Max(0, l2))
	return Ellipse{
		Center: cv.Point2f{X: mx, Y: my},
		Width:  major,
		Height: minor,
		Angle:  angle,
	}
}
