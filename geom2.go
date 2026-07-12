package cv

import "math"

// Point2f is a floating-point image coordinate (x is the column, y is the
// row), the analogue of OpenCV's cv::Point2f.
type Point2f struct {
	X float64
	Y float64
}

// PointPolygonTest reports the relationship between a point and a polygon given
// by its vertices. When measureDist is false it returns +1 if the point is
// inside, -1 if outside and 0 if on an edge. When measureDist is true it returns
// the signed distance to the nearest edge (positive inside, negative outside).
// This mirrors OpenCV's cv::pointPolygonTest.
func PointPolygonTest(contour []Point, pt Point, measureDist bool) float64 {
	n := len(contour)
	if n < 3 {
		if measureDist {
			return -math.MaxFloat64
		}
		return -1
	}
	px, py := float64(pt.X), float64(pt.Y)
	inside := false
	minDist := math.MaxFloat64
	for i := 0; i < n; i++ {
		a := contour[i]
		b := contour[(i+1)%n]
		ax, ay := float64(a.X), float64(a.Y)
		bx, by := float64(b.X), float64(b.Y)
		// Ray-cast parity test for containment.
		if (ay > py) != (by > py) {
			xCross := ax + (py-ay)/(by-ay)*(bx-ax)
			if px < xCross {
				inside = !inside
			}
		}
		if measureDist || onSegment(px, py, ax, ay, bx, by) {
			d := segmentDist(px, py, ax, ay, bx, by)
			if d < minDist {
				minDist = d
			}
		}
	}
	if !measureDist {
		if minDist == 0 {
			return 0
		}
		if inside {
			return 1
		}
		return -1
	}
	if inside {
		return minDist
	}
	return -minDist
}

// onSegment reports whether (px,py) lies on the segment (ax,ay)-(bx,by).
func onSegment(px, py, ax, ay, bx, by float64) bool {
	return segmentDist(px, py, ax, ay, bx, by) < 1e-9
}

// segmentDist returns the Euclidean distance from (px,py) to the segment
// (ax,ay)-(bx,by).
func segmentDist(px, py, ax, ay, bx, by float64) float64 {
	dx, dy := bx-ax, by-ay
	if dx == 0 && dy == 0 {
		return math.Hypot(px-ax, py-ay)
	}
	t := ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)
	if t < 0 {
		t = 0
	} else if t > 1 {
		t = 1
	}
	return math.Hypot(px-(ax+t*dx), py-(ay+t*dy))
}

// IsContourConvex reports whether a polygon is convex, i.e. every turn between
// consecutive edges has the same orientation and there are no self-intersections
// implied by a sign change. Fewer than three vertices are treated as
// non-convex. This mirrors OpenCV's cv::isContourConvex.
func IsContourConvex(contour []Point) bool {
	n := len(contour)
	if n < 3 {
		return false
	}
	var sign int
	for i := 0; i < n; i++ {
		a := contour[i]
		b := contour[(i+1)%n]
		c := contour[(i+2)%n]
		cross := (b.X-a.X)*(c.Y-b.Y) - (b.Y-a.Y)*(c.X-b.X)
		if cross != 0 {
			s := 1
			if cross < 0 {
				s = -1
			}
			if sign == 0 {
				sign = s
			} else if s != sign {
				return false
			}
		}
	}
	return true
}

// MinEnclosingCircle returns the centre and radius of the smallest circle that
// encloses every input point, computed with Welzl's randomised algorithm (run
// deterministically over the given order). It panics on an empty point set.
func MinEnclosingCircle(pts []Point) (center Point2f, radius float64) {
	if len(pts) == 0 {
		panic("cv: MinEnclosingCircle on empty point set")
	}
	ps := make([]Point2f, len(pts))
	for i, p := range pts {
		ps[i] = Point2f{X: float64(p.X), Y: float64(p.Y)}
	}
	c := welzl(ps, nil)
	return Point2f{X: c.cx, Y: c.cy}, c.r
}

// circle is a helper record for the minimum enclosing circle search.
type circle struct {
	cx, cy, r float64
}

// contains reports whether p lies within the circle (with a small tolerance).
func (c circle) contains(p Point2f) bool {
	return math.Hypot(p.X-c.cx, p.Y-c.cy) <= c.r+1e-9
}

// welzl computes the minimum enclosing circle of points not yet placed on the
// boundary, given up to three boundary points.
func welzl(pts []Point2f, boundary []Point2f) circle {
	if len(pts) == 0 || len(boundary) == 3 {
		return trivialCircle(boundary)
	}
	p := pts[len(pts)-1]
	rest := pts[:len(pts)-1]
	c := welzl(rest, boundary)
	if c.contains(p) {
		return c
	}
	return welzl(rest, append(append([]Point2f{}, boundary...), p))
}

// trivialCircle solves the enclosing circle for zero to three boundary points.
func trivialCircle(b []Point2f) circle {
	switch len(b) {
	case 0:
		return circle{}
	case 1:
		return circle{cx: b[0].X, cy: b[0].Y}
	case 2:
		return circleFrom2(b[0], b[1])
	default:
		c, ok := circleFrom3(b[0], b[1], b[2])
		if !ok {
			// Collinear: fall back to the widest pair.
			best := circleFrom2(b[0], b[1])
			for _, pair := range [][2]Point2f{{b[0], b[2]}, {b[1], b[2]}} {
				cc := circleFrom2(pair[0], pair[1])
				if cc.r > best.r {
					best = cc
				}
			}
			return best
		}
		return c
	}
}

// circleFrom2 returns the circle with the two points as a diameter.
func circleFrom2(a, b Point2f) circle {
	cx := (a.X + b.X) / 2
	cy := (a.Y + b.Y) / 2
	return circle{cx: cx, cy: cy, r: math.Hypot(a.X-cx, a.Y-cy)}
}

// circleFrom3 returns the circumscribed circle of three points, reporting false
// when they are collinear.
func circleFrom3(a, b, c Point2f) (circle, bool) {
	d := 2 * (a.X*(b.Y-c.Y) + b.X*(c.Y-a.Y) + c.X*(a.Y-b.Y))
	if math.Abs(d) < 1e-12 {
		return circle{}, false
	}
	a2 := a.X*a.X + a.Y*a.Y
	b2 := b.X*b.X + b.Y*b.Y
	c2 := c.X*c.X + c.Y*c.Y
	ux := (a2*(b.Y-c.Y) + b2*(c.Y-a.Y) + c2*(a.Y-b.Y)) / d
	uy := (a2*(c.X-b.X) + b2*(a.X-c.X) + c2*(b.X-a.X)) / d
	return circle{cx: ux, cy: uy, r: math.Hypot(a.X-ux, a.Y-uy)}, true
}

// FitLine fits a line to a set of points by total least squares (minimising the
// orthogonal distance) and returns it as a unit direction (vx, vy) and a point
// (x0, y0) on the line, matching the parameterisation of OpenCV's cv::fitLine
// with DIST_L2. It panics on fewer than two points.
func FitLine(pts []Point) (vx, vy, x0, y0 float64) {
	n := len(pts)
	if n < 2 {
		panic("cv: FitLine requires at least two points")
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
	// Principal eigenvector of the 2×2 scatter matrix [[sxx,sxy],[sxy,syy]].
	theta := 0.5 * math.Atan2(2*sxy, sxx-syy)
	vx = math.Cos(theta)
	vy = math.Sin(theta)
	return vx, vy, mx, my
}

// HuMoments returns the seven Hu invariant moments computed from a set of
// normalised central moments, matching OpenCV's cv::HuMoments. They are
// invariant to translation, scale and rotation.
func HuMoments(m Moments) [7]float64 {
	var h [7]float64
	n20, n02, n11 := m.Nu20, m.Nu02, m.Nu11
	n30, n12, n21, n03 := m.Nu30, m.Nu12, m.Nu21, m.Nu03
	h[0] = n20 + n02
	h[1] = (n20-n02)*(n20-n02) + 4*n11*n11
	h[2] = (n30-3*n12)*(n30-3*n12) + (3*n21-n03)*(3*n21-n03)
	h[3] = (n30+n12)*(n30+n12) + (n21+n03)*(n21+n03)
	h[4] = (n30-3*n12)*(n30+n12)*((n30+n12)*(n30+n12)-3*(n21+n03)*(n21+n03)) +
		(3*n21-n03)*(n21+n03)*(3*(n30+n12)*(n30+n12)-(n21+n03)*(n21+n03))
	h[5] = (n20-n02)*((n30+n12)*(n30+n12)-(n21+n03)*(n21+n03)) +
		4*n11*(n30+n12)*(n21+n03)
	h[6] = (3*n21-n03)*(n30+n12)*((n30+n12)*(n30+n12)-3*(n21+n03)*(n21+n03)) -
		(n30-3*n12)*(n21+n03)*(3*(n30+n12)*(n30+n12)-(n21+n03)*(n21+n03))
	return h
}

// MatchShapes compares two shapes given by their moments using the log-scaled
// Hu-moment metric I1 (OpenCV's CONTOURS_MATCH_I1). A smaller value means a
// closer match; identical shapes score 0.
func MatchShapes(a, b Moments) float64 {
	ha := HuMoments(a)
	hb := HuMoments(b)
	var sum float64
	for i := 0; i < 7; i++ {
		ma := signedLog(ha[i])
		mb := signedLog(hb[i])
		if ma == 0 || mb == 0 {
			continue
		}
		sum += math.Abs(1/ma - 1/mb)
	}
	return sum
}

// signedLog returns sign(v)*log10(|v|), used by the Hu-moment shape metric.
func signedLog(v float64) float64 {
	if v == 0 {
		return 0
	}
	l := math.Log10(math.Abs(v))
	if v < 0 {
		return -l
	}
	return l
}

// PerspectiveTransform applies a 3×3 projective transform to a slice of
// floating-point points, dividing by the homogeneous coordinate, matching
// OpenCV's cv::perspectiveTransform for 2-D points.
func PerspectiveTransform(pts []Point2f, m PerspectiveMatrix) []Point2f {
	out := make([]Point2f, len(pts))
	for i, p := range pts {
		w := m[6]*p.X + m[7]*p.Y + m[8]
		if w == 0 {
			w = 1e-30
		}
		out[i] = Point2f{
			X: (m[0]*p.X + m[1]*p.Y + m[2]) / w,
			Y: (m[3]*p.X + m[4]*p.Y + m[5]) / w,
		}
	}
	return out
}
