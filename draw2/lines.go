package draw2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Line draws a straight line from pt1 to pt2 with the given colour and
// thickness in pixels (minimum 1) using Bresenham's algorithm. Thicknesses
// greater than one are produced by stamping a small disc at each step.
func Line(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar, thickness int) {
	if thickness < 1 {
		thickness = 1
	}
	r := (thickness - 1) / 2
	x0, y0 := pt1.X, pt1.Y
	x1, y1 := pt2.X, pt2.Y
	dx := draw2absInt(x1 - x0)
	dy := -draw2absInt(y1 - y0)
	sx := draw2signInt(x1 - x0)
	sy := draw2signInt(y1 - y0)
	err := dx + dy
	for {
		if thickness == 1 {
			draw2set(m, x0, y0, color)
		} else {
			draw2disc(m, x0, y0, r, color)
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

// ThickLine draws a filled rectangular line of the given width from pt1 to
// pt2. Unlike [Line], the join is a true rectangle of the requested width
// rather than a chain of discs, giving crisper wide strokes. width is clamped
// to a minimum of 1.
func ThickLine(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar, width int) {
	if width < 1 {
		width = 1
	}
	if width == 1 {
		Line(m, pt1, pt2, color, 1)
		return
	}
	dx := float64(pt2.X - pt1.X)
	dy := float64(pt2.Y - pt1.Y)
	length := math.Hypot(dx, dy)
	if length == 0 {
		draw2disc(m, pt1.X, pt1.Y, width/2, color)
		return
	}
	// Unit normal to the segment.
	nx := -dy / length
	ny := dx / length
	h := float64(width) / 2
	corners := []cv.Point{
		{X: draw2round(float64(pt1.X) + nx*h), Y: draw2round(float64(pt1.Y) + ny*h)},
		{X: draw2round(float64(pt2.X) + nx*h), Y: draw2round(float64(pt2.Y) + ny*h)},
		{X: draw2round(float64(pt2.X) - nx*h), Y: draw2round(float64(pt2.Y) - ny*h)},
		{X: draw2round(float64(pt1.X) - nx*h), Y: draw2round(float64(pt1.Y) - ny*h)},
	}
	FillConvexPolygon(m, corners, color)
}

// WuLine draws an anti-aliased line from pt1 to pt2 using Xiaolin Wu's
// algorithm. Endpoints are given in floating point so sub-pixel positions
// render smoothly; coverage is alpha-composited over the existing pixels.
func WuLine(m *cv.Mat, x0, y0, x1, y1 float64, color cv.Scalar) {
	steep := math.Abs(y1-y0) > math.Abs(x1-x0)
	if steep {
		x0, y0 = y0, x0
		x1, y1 = y1, x1
	}
	if x0 > x1 {
		x0, x1 = x1, x0
		y0, y1 = y1, y0
	}
	dx := x1 - x0
	dy := y1 - y0
	gradient := 1.0
	if dx != 0 {
		gradient = dy / dx
	}

	plot := func(x, y int, c float64) {
		if steep {
			draw2blend(m, y, x, color, c)
		} else {
			draw2blend(m, x, y, color, c)
		}
	}

	// First endpoint.
	xend := math.Floor(x0 + 0.5)
	yend := y0 + gradient*(xend-x0)
	xgap := draw2rfpart(x0 + 0.5)
	xpxl1 := int(xend)
	ypxl1 := int(math.Floor(yend))
	plot(xpxl1, ypxl1, draw2rfpart(yend)*xgap)
	plot(xpxl1, ypxl1+1, draw2fpart(yend)*xgap)
	intery := yend + gradient

	// Second endpoint.
	xend = math.Floor(x1 + 0.5)
	yend = y1 + gradient*(xend-x1)
	xgap = draw2fpart(x1 + 0.5)
	xpxl2 := int(xend)
	ypxl2 := int(math.Floor(yend))
	plot(xpxl2, ypxl2, draw2rfpart(yend)*xgap)
	plot(xpxl2, ypxl2+1, draw2fpart(yend)*xgap)

	for x := xpxl1 + 1; x < xpxl2; x++ {
		yb := int(math.Floor(intery))
		plot(x, yb, draw2rfpart(intery))
		plot(x, yb+1, draw2fpart(intery))
		intery += gradient
	}
}

// DashedLine draws a dashed straight line from pt1 to pt2. dashLen and gapLen
// give the on and off run lengths in pixels; both are clamped to a minimum of
// 1. thickness controls stroke width.
func DashedLine(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar, thickness, dashLen, gapLen int) {
	if dashLen < 1 {
		dashLen = 1
	}
	if gapLen < 1 {
		gapLen = 1
	}
	dx := float64(pt2.X - pt1.X)
	dy := float64(pt2.Y - pt1.Y)
	length := math.Hypot(dx, dy)
	if length == 0 {
		Line(m, pt1, pt2, color, thickness)
		return
	}
	ux := dx / length
	uy := dy / length
	period := float64(dashLen + gapLen)
	for s := 0.0; s < length; s += period {
		e := s + float64(dashLen)
		if e > length {
			e = length
		}
		a := cv.Point{X: pt1.X + draw2round(ux*s), Y: pt1.Y + draw2round(uy*s)}
		b := cv.Point{X: pt1.X + draw2round(ux*e), Y: pt1.Y + draw2round(uy*e)}
		Line(m, a, b, color, thickness)
	}
}

// Polyline draws a connected sequence of straight segments through pts. When
// closed is true the last point is joined back to the first. thickness
// controls stroke width.
func Polyline(m *cv.Mat, pts []cv.Point, closed bool, color cv.Scalar, thickness int) {
	if len(pts) == 0 {
		return
	}
	for i := 0; i+1 < len(pts); i++ {
		Line(m, pts[i], pts[i+1], color, thickness)
	}
	if closed && len(pts) > 1 {
		Line(m, pts[len(pts)-1], pts[0], color, thickness)
	}
}

// AAPolyline draws a connected sequence of anti-aliased segments through pts
// using [WuLine]. When closed is true the last point is joined back to the
// first.
func AAPolyline(m *cv.Mat, pts []cv.Point, closed bool, color cv.Scalar) {
	if len(pts) == 0 {
		return
	}
	for i := 0; i+1 < len(pts); i++ {
		WuLine(m, float64(pts[i].X), float64(pts[i].Y), float64(pts[i+1].X), float64(pts[i+1].Y), color)
	}
	if closed && len(pts) > 1 {
		last := pts[len(pts)-1]
		WuLine(m, float64(last.X), float64(last.Y), float64(pts[0].X), float64(pts[0].Y), color)
	}
}

// ArrowedLine draws a line from pt1 to pt2 with an arrow head at pt2. tipLength
// is the head length as a fraction of the segment length (a typical value is
// 0.1). thickness controls stroke width.
func ArrowedLine(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar, thickness int, tipLength float64) {
	Line(m, pt1, pt2, color, thickness)
	if tipLength <= 0 {
		tipLength = 0.1
	}
	dx := float64(pt1.X - pt2.X)
	dy := float64(pt1.Y - pt2.Y)
	length := math.Hypot(dx, dy)
	if length == 0 {
		return
	}
	angle := math.Atan2(dy, dx)
	headLen := tipLength * length
	const spread = math.Pi / 6 // 30 degrees each side
	p1 := cv.Point{
		X: pt2.X + draw2round(headLen*math.Cos(angle+spread)),
		Y: pt2.Y + draw2round(headLen*math.Sin(angle+spread)),
	}
	p2 := cv.Point{
		X: pt2.X + draw2round(headLen*math.Cos(angle-spread)),
		Y: pt2.Y + draw2round(headLen*math.Sin(angle-spread)),
	}
	Line(m, pt2, p1, color, thickness)
	Line(m, pt2, p2, color, thickness)
}
