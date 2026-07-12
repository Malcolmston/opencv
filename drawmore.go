package cv

import "math"

// MarkerType selects the glyph drawn by [DrawMarker].
type MarkerType int

const (
	// MarkerCross draws a upright plus sign (+).
	MarkerCross MarkerType = iota
	// MarkerTiltedCross draws a diagonal cross (×).
	MarkerTiltedCross
	// MarkerStar draws an eight-pointed star (combined + and ×).
	MarkerStar
	// MarkerDiamond draws a diamond outline.
	MarkerDiamond
	// MarkerSquare draws an axis-aligned square outline.
	MarkerSquare
	// MarkerTriangleUp draws an upward-pointing triangle outline.
	MarkerTriangleUp
	// MarkerTriangleDown draws a downward-pointing triangle outline.
	MarkerTriangleDown
)

// DrawMarker draws a marker of the given type centred at position, with a
// bounding size of markerSize pixels and the given line thickness. It mirrors
// OpenCV's cv::drawMarker.
func DrawMarker(m *Mat, position Point, color Scalar, markerType MarkerType, markerSize, thickness int) {
	if markerSize <= 0 {
		markerSize = 20
	}
	r := markerSize / 2
	cx, cy := position.X, position.Y
	switch markerType {
	case MarkerCross:
		Line(m, Point{cx - r, cy}, Point{cx + r, cy}, color, thickness)
		Line(m, Point{cx, cy - r}, Point{cx, cy + r}, color, thickness)
	case MarkerTiltedCross:
		Line(m, Point{cx - r, cy - r}, Point{cx + r, cy + r}, color, thickness)
		Line(m, Point{cx - r, cy + r}, Point{cx + r, cy - r}, color, thickness)
	case MarkerStar:
		Line(m, Point{cx - r, cy}, Point{cx + r, cy}, color, thickness)
		Line(m, Point{cx, cy - r}, Point{cx, cy + r}, color, thickness)
		Line(m, Point{cx - r, cy - r}, Point{cx + r, cy + r}, color, thickness)
		Line(m, Point{cx - r, cy + r}, Point{cx + r, cy - r}, color, thickness)
	case MarkerDiamond:
		pts := []Point{{cx, cy - r}, {cx + r, cy}, {cx, cy + r}, {cx - r, cy}}
		Polylines(m, [][]Point{pts}, true, color, thickness)
	case MarkerSquare:
		pts := []Point{{cx - r, cy - r}, {cx + r, cy - r}, {cx + r, cy + r}, {cx - r, cy + r}}
		Polylines(m, [][]Point{pts}, true, color, thickness)
	case MarkerTriangleUp:
		pts := []Point{{cx, cy - r}, {cx + r, cy + r}, {cx - r, cy + r}}
		Polylines(m, [][]Point{pts}, true, color, thickness)
	case MarkerTriangleDown:
		pts := []Point{{cx, cy + r}, {cx + r, cy - r}, {cx - r, cy - r}}
		Polylines(m, [][]Point{pts}, true, color, thickness)
	}
}

// ArrowedLine draws a line segment from pt1 to pt2 with an arrow head at pt2.
// tipLength is the length of the arrow head as a fraction of the segment length
// (OpenCV's default is 0.1). It mirrors cv::arrowedLine.
func ArrowedLine(m *Mat, pt1, pt2 Point, color Scalar, thickness int, tipLength float64) {
	Line(m, pt1, pt2, color, thickness)
	if tipLength <= 0 {
		tipLength = 0.1
	}
	dx := float64(pt1.X - pt2.X)
	dy := float64(pt1.Y - pt2.Y)
	length := math.Hypot(dx, dy)
	if length < 1e-9 {
		return
	}
	angle := math.Atan2(dy, dx)
	tip := tipLength * length
	const spread = math.Pi / 4 // 45° half-opening between shaft and each barb.
	for _, a := range []float64{angle + spread/2, angle - spread/2} {
		bx := pt2.X + int(math.Round(tip*math.Cos(a)))
		by := pt2.Y + int(math.Round(tip*math.Sin(a)))
		Line(m, Point{bx, by}, pt2, color, thickness)
	}
}

// GetTextSize returns the bounding box (width, height) of text rendered with
// [PutText] at the given integer scale, together with the baseline offset below
// the box. The width and height are exact for the built-in 5×7 bitmap font. It
// mirrors cv::getTextSize.
func GetTextSize(text string, scale int) (size Point, baseline int) {
	if scale < 1 {
		scale = 1
	}
	n := len([]rune(text))
	width := 0
	if n > 0 {
		width = n*(fontWidth+1)*scale - scale
	}
	return Point{X: width, Y: fontHeight * scale}, scale
}

// ClipLine clips the segment pt1-pt2 against the rectangle rect using the
// Liang-Barsky algorithm. It returns the clipped endpoints and reports whether
// any part of the segment lies inside the rectangle. It mirrors cv::clipLine.
func ClipLine(rect Rect, pt1, pt2 Point) (Point, Point, bool) {
	x0 := float64(pt1.X)
	y0 := float64(pt1.Y)
	x1 := float64(pt2.X)
	y1 := float64(pt2.Y)
	dx := x1 - x0
	dy := y1 - y0
	tMin, tMax := 0.0, 1.0
	left := float64(rect.X)
	right := float64(rect.X + rect.Width - 1)
	top := float64(rect.Y)
	bottom := float64(rect.Y + rect.Height - 1)
	edges := [4][2]float64{{-dx, x0 - left}, {dx, right - x0}, {-dy, y0 - top}, {dy, bottom - y0}}
	for _, e := range edges {
		p, q := e[0], e[1]
		if p == 0 {
			if q < 0 {
				return pt1, pt2, false
			}
			continue
		}
		t := q / p
		if p < 0 {
			if t > tMax {
				return pt1, pt2, false
			}
			if t > tMin {
				tMin = t
			}
		} else {
			if t < tMin {
				return pt1, pt2, false
			}
			if t < tMax {
				tMax = t
			}
		}
	}
	c1 := Point{X: int(math.Round(x0 + tMin*dx)), Y: int(math.Round(y0 + tMin*dy))}
	c2 := Point{X: int(math.Round(x0 + tMax*dx)), Y: int(math.Round(y0 + tMax*dy))}
	return c1, c2, true
}

// Ellipse2Poly approximates an elliptical arc with a polyline. The ellipse is
// centred at center with semi-axes axesX and axesY, rotated by angle degrees;
// the arc spans from arcStart to arcEnd degrees and is sampled every delta
// degrees. It mirrors cv::ellipse2Poly.
func Ellipse2Poly(center Point, axesX, axesY int, angle, arcStart, arcEnd float64, delta int) []Point {
	if delta <= 0 {
		delta = 1
	}
	if arcEnd < arcStart {
		arcStart, arcEnd = arcEnd, arcStart
	}
	rad := angle * math.Pi / 180
	ca, sa := math.Cos(rad), math.Sin(rad)
	var pts []Point
	for a := arcStart; a < arcEnd+float64(delta)/2; a += float64(delta) {
		if a > arcEnd {
			a = arcEnd
		}
		t := a * math.Pi / 180
		x := float64(axesX) * math.Cos(t)
		y := float64(axesY) * math.Sin(t)
		px := center.X + int(math.Round(x*ca-y*sa))
		py := center.Y + int(math.Round(x*sa+y*ca))
		pts = append(pts, Point{X: px, Y: py})
		if a == arcEnd {
			break
		}
	}
	return pts
}

// FillConvexPoly fills a convex polygon with a solid colour using a scanline
// fill that is faster and cleaner than the general [FillPoly] for convex
// shapes. The vertices may be given in either winding order. It mirrors
// cv::fillConvexPoly.
func FillConvexPoly(m *Mat, pts []Point, color Scalar) {
	if len(pts) < 3 {
		return
	}
	minY, maxY := pts[0].Y, pts[0].Y
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
		xMin, xMax := math.MaxInt, math.MinInt
		for i := 0; i < n; i++ {
			a := pts[i]
			b := pts[(i+1)%n]
			ay, by := a.Y, b.Y
			if ay == by {
				continue
			}
			if (y >= ay && y < by) || (y >= by && y < ay) {
				t := float64(y-ay) / float64(by-ay)
				x := int(math.Round(float64(a.X) + t*float64(b.X-a.X)))
				if x < xMin {
					xMin = x
				}
				if x > xMax {
					xMax = x
				}
			}
		}
		if xMin > xMax {
			continue
		}
		for x := xMin; x <= xMax; x++ {
			m.setPixelScalar(x, y, color)
		}
	}
}

// BoxPoints returns the four corner points of a rotated rectangle as
// floating-point coordinates, in the order bottom-left, top-left, top-right,
// bottom-right (matching OpenCV's cv::boxPoints convention where y grows
// downward).
func BoxPoints(r RotatedRect) [4]Point2f {
	rad := r.Angle * math.Pi / 180
	c, s := math.Cos(rad), math.Sin(rad)
	hw, hh := r.Width/2, r.Height/2
	local := [4][2]float64{{-hw, hh}, {-hw, -hh}, {hw, -hh}, {hw, hh}}
	var out [4]Point2f
	for i, p := range local {
		out[i] = Point2f{
			X: r.CenterX + p[0]*c - p[1]*s,
			Y: r.CenterY + p[0]*s + p[1]*c,
		}
	}
	return out
}
