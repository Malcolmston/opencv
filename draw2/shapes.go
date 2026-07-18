package draw2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Filled is a sentinel thickness value passed to outline routines to request a
// solid fill instead of an outline.
const Filled = -1

// Rectangle draws the outline of an axis-aligned rectangle spanning corners
// pt1 and pt2. thickness controls stroke width; pass [Filled] (or any negative
// value) to fill the interior instead.
func Rectangle(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar, thickness int) {
	if thickness < 0 {
		FilledRectangle(m, pt1, pt2, color)
		return
	}
	x0, x1 := draw2minInt(pt1.X, pt2.X), draw2maxInt(pt1.X, pt2.X)
	y0, y1 := draw2minInt(pt1.Y, pt2.Y), draw2maxInt(pt1.Y, pt2.Y)
	Line(m, cv.Point{X: x0, Y: y0}, cv.Point{X: x1, Y: y0}, color, thickness)
	Line(m, cv.Point{X: x0, Y: y1}, cv.Point{X: x1, Y: y1}, color, thickness)
	Line(m, cv.Point{X: x0, Y: y0}, cv.Point{X: x0, Y: y1}, color, thickness)
	Line(m, cv.Point{X: x1, Y: y0}, cv.Point{X: x1, Y: y1}, color, thickness)
}

// FilledRectangle fills the axis-aligned rectangle spanning corners pt1 and
// pt2 with a solid colour. The rectangle is clipped to the image bounds.
func FilledRectangle(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar) {
	x0 := draw2maxInt(0, draw2minInt(pt1.X, pt2.X))
	x1 := draw2minInt(m.Cols-1, draw2maxInt(pt1.X, pt2.X))
	y0 := draw2maxInt(0, draw2minInt(pt1.Y, pt2.Y))
	y1 := draw2minInt(m.Rows-1, draw2maxInt(pt1.Y, pt2.Y))
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			draw2set(m, x, y, color)
		}
	}
}

// RoundedRectangle draws the outline of a rectangle whose corners are rounded
// with the given radius. thickness controls stroke width. The corner radius is
// clamped so it never exceeds half the shorter side.
func RoundedRectangle(m *cv.Mat, pt1, pt2 cv.Point, radius int, color cv.Scalar, thickness int) {
	x0, x1 := draw2minInt(pt1.X, pt2.X), draw2maxInt(pt1.X, pt2.X)
	y0, y1 := draw2minInt(pt1.Y, pt2.Y), draw2maxInt(pt1.Y, pt2.Y)
	maxR := draw2minInt(x1-x0, y1-y0) / 2
	if radius > maxR {
		radius = maxR
	}
	if radius <= 0 {
		Rectangle(m, pt1, pt2, color, thickness)
		return
	}
	// Straight edges.
	Line(m, cv.Point{X: x0 + radius, Y: y0}, cv.Point{X: x1 - radius, Y: y0}, color, thickness)
	Line(m, cv.Point{X: x0 + radius, Y: y1}, cv.Point{X: x1 - radius, Y: y1}, color, thickness)
	Line(m, cv.Point{X: x0, Y: y0 + radius}, cv.Point{X: x0, Y: y1 - radius}, color, thickness)
	Line(m, cv.Point{X: x1, Y: y0 + radius}, cv.Point{X: x1, Y: y1 - radius}, color, thickness)
	// Corner arcs (quarter circles).
	EllipticArc(m, cv.Point{X: x0 + radius, Y: y0 + radius}, radius, radius, 180, 270, color, thickness)
	EllipticArc(m, cv.Point{X: x1 - radius, Y: y0 + radius}, radius, radius, 270, 360, color, thickness)
	EllipticArc(m, cv.Point{X: x1 - radius, Y: y1 - radius}, radius, radius, 0, 90, color, thickness)
	EllipticArc(m, cv.Point{X: x0 + radius, Y: y1 - radius}, radius, radius, 90, 180, color, thickness)
}

// Circle draws the outline of a circle of the given radius centred at center
// using the midpoint (Bresenham) circle algorithm. thickness controls stroke
// width; pass [Filled] to fill the disc instead.
func Circle(m *cv.Mat, center cv.Point, radius int, color cv.Scalar, thickness int) {
	if radius < 0 {
		return
	}
	if thickness < 0 {
		FilledCircle(m, center, radius, color)
		return
	}
	if thickness < 1 {
		thickness = 1
	}
	r := (thickness - 1) / 2
	x := radius
	y := 0
	e := 1 - x
	plot := func(px, py int) {
		if thickness == 1 {
			draw2set(m, px, py, color)
		} else {
			draw2disc(m, px, py, r, color)
		}
	}
	for x >= y {
		plot(center.X+x, center.Y+y)
		plot(center.X+y, center.Y+x)
		plot(center.X-y, center.Y+x)
		plot(center.X-x, center.Y+y)
		plot(center.X-x, center.Y-y)
		plot(center.X-y, center.Y-x)
		plot(center.X+y, center.Y-x)
		plot(center.X+x, center.Y-y)
		y++
		if e < 0 {
			e += 2*y + 1
		} else {
			x--
			e += 2*(y-x) + 1
		}
	}
}

// FilledCircle fills a disc of the given radius centred at center with a solid
// colour, using an exact per-scanline span so the boundary matches the
// discrete disc x*x + y*y <= r*r.
func FilledCircle(m *cv.Mat, center cv.Point, radius int, color cv.Scalar) {
	if radius < 0 {
		return
	}
	for dy := -radius; dy <= radius; dy++ {
		dxMax := int(math.Floor(math.Sqrt(float64(radius*radius - dy*dy))))
		y := center.Y + dy
		for dx := -dxMax; dx <= dxMax; dx++ {
			draw2set(m, center.X+dx, y, color)
		}
	}
}

// AACircle draws an anti-aliased circle outline of the given radius centred at
// center. Coverage is computed from the distance of each boundary pixel to the
// ideal circle and alpha-composited over the image.
func AACircle(m *cv.Mat, center cv.Point, radius float64, color cv.Scalar) {
	if radius < 0 {
		return
	}
	r := int(math.Ceil(radius)) + 1
	for dy := -r; dy <= r; dy++ {
		for dx := -r; dx <= r; dx++ {
			d := math.Hypot(float64(dx), float64(dy))
			cov := 1 - math.Abs(d-radius)
			if cov > 0 {
				draw2blend(m, center.X+dx, center.Y+dy, color, cov)
			}
		}
	}
}

// Ellipse draws the outline of an axis-aligned ellipse centred at center with
// semi-axes axesX and axesY, using the midpoint ellipse algorithm. thickness
// controls stroke width; pass [Filled] to fill it. For rotated ellipses use
// [EllipseRotated].
func Ellipse(m *cv.Mat, center cv.Point, axesX, axesY int, color cv.Scalar, thickness int) {
	if axesX <= 0 || axesY <= 0 {
		return
	}
	if thickness < 0 {
		FilledEllipse(m, center, axesX, axesY, color)
		return
	}
	if thickness < 1 {
		thickness = 1
	}
	r := (thickness - 1) / 2
	plot := func(px, py int) {
		if thickness == 1 {
			draw2set(m, px, py, color)
		} else {
			draw2disc(m, px, py, r, color)
		}
	}
	a := float64(axesX)
	b := float64(axesY)
	a2 := a * a
	b2 := b * b
	x := 0.0
	y := b
	// Region 1.
	d1 := b2 - a2*b + 0.25*a2
	dx := 2 * b2 * x
	dy := 2 * a2 * y
	for dx < dy {
		plot(center.X+int(x), center.Y+int(y))
		plot(center.X-int(x), center.Y+int(y))
		plot(center.X+int(x), center.Y-int(y))
		plot(center.X-int(x), center.Y-int(y))
		if d1 < 0 {
			x++
			dx += 2 * b2
			d1 += dx + b2
		} else {
			x++
			y--
			dx += 2 * b2
			dy -= 2 * a2
			d1 += dx - dy + b2
		}
	}
	// Region 2.
	d2 := b2*(x+0.5)*(x+0.5) + a2*(y-1)*(y-1) - a2*b2
	for y >= 0 {
		plot(center.X+int(x), center.Y+int(y))
		plot(center.X-int(x), center.Y+int(y))
		plot(center.X+int(x), center.Y-int(y))
		plot(center.X-int(x), center.Y-int(y))
		if d2 > 0 {
			y--
			dy -= 2 * a2
			d2 += a2 - dy
		} else {
			y--
			x++
			dx += 2 * b2
			dy -= 2 * a2
			d2 += dx - dy + a2
		}
	}
}

// FilledEllipse fills an axis-aligned ellipse centred at center with semi-axes
// axesX and axesY, using an exact per-scanline span derived from the ellipse
// equation.
func FilledEllipse(m *cv.Mat, center cv.Point, axesX, axesY int, color cv.Scalar) {
	if axesX <= 0 || axesY <= 0 {
		return
	}
	a2 := float64(axesX * axesX)
	for dy := -axesY; dy <= axesY; dy++ {
		frac := 1 - float64(dy*dy)/float64(axesY*axesY)
		if frac < 0 {
			continue
		}
		dxMax := int(math.Floor(math.Sqrt(a2 * frac)))
		y := center.Y + dy
		for dx := -dxMax; dx <= dxMax; dx++ {
			draw2set(m, center.X+dx, y, color)
		}
	}
}

// EllipseRotated draws a rotated ellipse centred at center with semi-axes
// axesX and axesY, rotated angle degrees clockwise. thickness controls stroke
// width; pass [Filled] to fill it. The curve is sampled as a closed polygon.
func EllipseRotated(m *cv.Mat, center cv.Point, axesX, axesY int, angle float64, color cv.Scalar, thickness int) {
	if axesX <= 0 || axesY <= 0 {
		return
	}
	rad := angle * math.Pi / 180
	ca, sa := math.Cos(rad), math.Sin(rad)
	const steps = 180
	pts := make([]cv.Point, steps)
	for i := 0; i < steps; i++ {
		t := 2 * math.Pi * float64(i) / float64(steps)
		ex := float64(axesX) * math.Cos(t)
		ey := float64(axesY) * math.Sin(t)
		pts[i] = cv.Point{
			X: center.X + draw2round(ex*ca-ey*sa),
			Y: center.Y + draw2round(ex*sa+ey*ca),
		}
	}
	if thickness < 0 {
		FillConvexPolygon(m, pts, color)
		return
	}
	Polyline(m, pts, true, color, thickness)
}

// EllipticArc draws an elliptical arc centred at center with semi-axes axesX
// and axesY, sweeping from startAngle to endAngle in degrees (measured
// clockwise from the positive x-axis). thickness controls stroke width. This
// is the arc primitive used by [Circle]-style APIs and rounded rectangles.
func EllipticArc(m *cv.Mat, center cv.Point, axesX, axesY int, startAngle, endAngle float64, color cv.Scalar, thickness int) {
	if axesX <= 0 || axesY <= 0 {
		return
	}
	if endAngle < startAngle {
		startAngle, endAngle = endAngle, startAngle
	}
	sweep := endAngle - startAngle
	steps := int(math.Ceil(sweep)) // ~1 degree resolution
	if steps < 1 {
		steps = 1
	}
	pts := make([]cv.Point, steps+1)
	for i := 0; i <= steps; i++ {
		ang := (startAngle + sweep*float64(i)/float64(steps)) * math.Pi / 180
		pts[i] = cv.Point{
			X: center.X + draw2round(float64(axesX)*math.Cos(ang)),
			Y: center.Y + draw2round(float64(axesY)*math.Sin(ang)),
		}
	}
	Polyline(m, pts, false, color, thickness)
}

// Polygon draws the closed outline of a polygon through pts. thickness
// controls stroke width; pass [Filled] to fill it with [FillPolygon].
func Polygon(m *cv.Mat, pts []cv.Point, color cv.Scalar, thickness int) {
	if thickness < 0 {
		FillPolygon(m, [][]cv.Point{pts}, color)
		return
	}
	Polyline(m, pts, true, color, thickness)
}

// RegularPolygon draws a regular polygon with n sides inscribed in a circle of
// the given radius centred at center. rotation rotates the shape in degrees.
// thickness controls stroke width; pass [Filled] to fill it.
func RegularPolygon(m *cv.Mat, center cv.Point, radius, n int, rotation float64, color cv.Scalar, thickness int) {
	if n < 3 || radius <= 0 {
		return
	}
	pts := make([]cv.Point, n)
	rot := rotation * math.Pi / 180
	for i := 0; i < n; i++ {
		ang := rot + 2*math.Pi*float64(i)/float64(n)
		pts[i] = cv.Point{
			X: center.X + draw2round(float64(radius)*math.Cos(ang)),
			Y: center.Y + draw2round(float64(radius)*math.Sin(ang)),
		}
	}
	if thickness < 0 {
		FillConvexPolygon(m, pts, color)
		return
	}
	Polyline(m, pts, true, color, thickness)
}

// FillPolygon fills one or more polygons with a solid colour using an even-odd
// scanline rule, so overlapping regions within a single call cancel. Each
// polygon is treated as implicitly closed.
func FillPolygon(m *cv.Mat, polys [][]cv.Point, color cv.Scalar) {
	minY, maxY := math.MaxInt, math.MinInt
	for _, poly := range polys {
		for _, p := range poly {
			if p.Y < minY {
				minY = p.Y
			}
			if p.Y > maxY {
				maxY = p.Y
			}
		}
	}
	if minY > maxY {
		return
	}
	if minY < 0 {
		minY = 0
	}
	if maxY >= m.Rows {
		maxY = m.Rows - 1
	}
	for y := minY; y <= maxY; y++ {
		var xs []float64
		for _, poly := range polys {
			n := len(poly)
			for i := 0; i < n; i++ {
				a := poly[i]
				b := poly[(i+1)%n]
				ay, by := a.Y, b.Y
				if ay == by {
					continue
				}
				if (y >= ay && y < by) || (y >= by && y < ay) {
					t := float64(y-ay) / float64(by-ay)
					xs = append(xs, float64(a.X)+t*float64(b.X-a.X))
				}
			}
		}
		if len(xs) < 2 {
			continue
		}
		draw2sortFloats(xs)
		for i := 0; i+1 < len(xs); i += 2 {
			xStart := int(math.Ceil(xs[i]))
			xEnd := int(math.Floor(xs[i+1]))
			for x := xStart; x <= xEnd; x++ {
				draw2set(m, x, y, color)
			}
		}
	}
}

// FillConvexPolygon fills a single convex polygon with a solid colour using a
// non-zero span per scanline. It is faster and more robust than [FillPolygon]
// for convex shapes because it fills the full extent between the leftmost and
// rightmost crossings on each row.
func FillConvexPolygon(m *cv.Mat, pts []cv.Point, color cv.Scalar) {
	if len(pts) < 3 {
		return
	}
	minY, maxY := math.MaxInt, math.MinInt
	for _, p := range pts {
		if p.Y < minY {
			minY = p.Y
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}
	if minY < 0 {
		minY = 0
	}
	if maxY >= m.Rows {
		maxY = m.Rows - 1
	}
	n := len(pts)
	for y := minY; y <= maxY; y++ {
		lo := math.Inf(1)
		hi := math.Inf(-1)
		for i := 0; i < n; i++ {
			a := pts[i]
			b := pts[(i+1)%n]
			ay, by := a.Y, b.Y
			if ay == by {
				continue
			}
			if (y >= ay && y < by) || (y >= by && y < ay) {
				t := float64(y-ay) / float64(by-ay)
				x := float64(a.X) + t*float64(b.X-a.X)
				if x < lo {
					lo = x
				}
				if x > hi {
					hi = x
				}
			}
		}
		if lo > hi {
			continue
		}
		for x := int(math.Ceil(lo)); x <= int(math.Floor(hi)); x++ {
			draw2set(m, x, y, color)
		}
	}
}

func draw2sortFloats(s []float64) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1] > s[j]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}
