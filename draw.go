package cv

import "math"

// Scalar is a colour with up to four components, matching cv2's Scalar. When
// drawing on a Mat only the first Channels components are used; RGB images
// interpret the components as (R, G, B).
type Scalar [4]float64

// NewScalar builds a Scalar from the given components (missing entries are
// zero, extra entries are ignored).
func NewScalar(v ...float64) Scalar {
	var s Scalar
	for i := 0; i < len(v) && i < 4; i++ {
		s[i] = v[i]
	}
	return s
}

// Point is an integer image coordinate (x is the column, y is the row).
type Point struct {
	X int
	Y int
}

// setPixelScalar writes color into pixel (x, y) of m, ignoring out-of-range
// coordinates so drawing primitives can clip silently.
func (m *Mat) setPixelScalar(x, y int, color Scalar) {
	if !m.inBounds(y, x) {
		return
	}
	i := m.index(y, x)
	for c := 0; c < m.Channels; c++ {
		m.Data[i+c] = clampToUint8(color[c] + 0.5)
	}
}

// fillDisc paints a filled disc of the given radius centred on (cx, cy), used
// to give lines and outlines their thickness.
func (m *Mat) fillDisc(cx, cy, radius int, color Scalar) {
	if radius <= 0 {
		m.setPixelScalar(cx, cy, color)
		return
	}
	r2 := radius * radius
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			if dx*dx+dy*dy <= r2 {
				m.setPixelScalar(cx+dx, cy+dy, color)
			}
		}
	}
}

// Line draws a straight line from pt1 to pt2 with the given colour and
// thickness (in pixels, minimum 1) using Bresenham's algorithm.
func Line(m *Mat, pt1, pt2 Point, color Scalar, thickness int) {
	if thickness < 1 {
		thickness = 1
	}
	r := (thickness - 1) / 2
	x0, y0 := pt1.X, pt1.Y
	x1, y1 := pt2.X, pt2.Y
	dx := abs(x1 - x0)
	dy := -abs(y1 - y0)
	sx := sign(x1 - x0)
	sy := sign(y1 - y0)
	err := dx + dy
	for {
		if thickness == 1 {
			m.setPixelScalar(x0, y0, color)
		} else {
			m.fillDisc(x0, y0, r, color)
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

// Rectangle draws an axis-aligned rectangle spanning corners pt1 and pt2. A
// positive thickness draws the outline; a negative thickness (or [Filled])
// fills the interior.
func Rectangle(m *Mat, pt1, pt2 Point, color Scalar, thickness int) {
	x0, x1 := minInt(pt1.X, pt2.X), maxInt(pt1.X, pt2.X)
	y0, y1 := minInt(pt1.Y, pt2.Y), maxInt(pt1.Y, pt2.Y)
	if thickness < 0 {
		for y := y0; y <= y1; y++ {
			for x := x0; x <= x1; x++ {
				m.setPixelScalar(x, y, color)
			}
		}
		return
	}
	Line(m, Point{x0, y0}, Point{x1, y0}, color, thickness)
	Line(m, Point{x0, y1}, Point{x1, y1}, color, thickness)
	Line(m, Point{x0, y0}, Point{x0, y1}, color, thickness)
	Line(m, Point{x1, y0}, Point{x1, y1}, color, thickness)
}

// Filled is a sentinel thickness that fills a shape instead of outlining it.
const Filled = -1

// Circle draws a circle of the given radius centred at center. A positive
// thickness draws the outline; a negative thickness fills the disc.
func Circle(m *Mat, center Point, radius int, color Scalar, thickness int) {
	if radius < 0 {
		return
	}
	if thickness < 0 {
		m.fillDisc(center.X, center.Y, radius, color)
		return
	}
	r := (thickness - 1) / 2
	// Midpoint circle algorithm, thickened by a small disc per plotted point.
	x := radius
	y := 0
	e := 1 - x
	plot := func(px, py int) {
		if thickness == 1 {
			m.setPixelScalar(px, py, color)
		} else {
			m.fillDisc(px, py, r, color)
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

// Ellipse draws an axis-aligned or rotated ellipse centred at center with the
// given semi-axes (axesX, axesY). angle rotates the ellipse in degrees. A
// positive thickness draws the outline; a negative thickness fills it. The
// ellipse is rendered as a closed polygon sampled around its perimeter.
func Ellipse(m *Mat, center Point, axesX, axesY int, angle float64, color Scalar, thickness int) {
	if axesX <= 0 || axesY <= 0 {
		return
	}
	rad := angle * math.Pi / 180
	ca, sa := math.Cos(rad), math.Sin(rad)
	const steps = 180
	pts := make([]Point, steps)
	for i := 0; i < steps; i++ {
		t := 2 * math.Pi * float64(i) / float64(steps)
		ex := float64(axesX) * math.Cos(t)
		ey := float64(axesY) * math.Sin(t)
		pts[i] = Point{
			X: center.X + int(math.Round(ex*ca-ey*sa)),
			Y: center.Y + int(math.Round(ex*sa+ey*ca)),
		}
	}
	if thickness < 0 {
		FillPoly(m, [][]Point{pts}, color)
		return
	}
	Polylines(m, [][]Point{pts}, true, color, thickness)
}

// Polylines draws one or more polylines. When closed is true each polyline's
// last point is joined back to its first. thickness is the line width.
func Polylines(m *Mat, polys [][]Point, closed bool, color Scalar, thickness int) {
	for _, poly := range polys {
		if len(poly) == 0 {
			continue
		}
		for i := 0; i < len(poly)-1; i++ {
			Line(m, poly[i], poly[i+1], color, thickness)
		}
		if closed && len(poly) > 1 {
			Line(m, poly[len(poly)-1], poly[0], color, thickness)
		}
	}
}

// FillPoly fills one or more polygons with a solid colour using an even-odd
// scanline algorithm, so overlapping regions of a single call cancel.
func FillPoly(m *Mat, polys [][]Point, color Scalar) {
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
		sortFloats(xs)
		for i := 0; i+1 < len(xs); i += 2 {
			xStart := int(math.Ceil(xs[i]))
			xEnd := int(math.Floor(xs[i+1]))
			for x := xStart; x <= xEnd; x++ {
				m.setPixelScalar(x, y, color)
			}
		}
	}
}

// PutText renders text at org (the bottom-left of the first glyph) using a
// built-in 5×7 bitmap font. scale integer-magnifies each glyph pixel and
// color/thickness control the appearance. Only ASCII 32–126 are drawn; unknown
// runes render as blanks.
func PutText(m *Mat, text string, org Point, scale int, color Scalar) {
	if scale < 1 {
		scale = 1
	}
	cursorX := org.X
	top := org.Y - fontHeight*scale
	for _, r := range text {
		glyph, ok := font5x7[r]
		if !ok {
			cursorX += (fontWidth + 1) * scale
			continue
		}
		for row := 0; row < fontHeight; row++ {
			bits := glyph[row]
			for col := 0; col < fontWidth; col++ {
				if bits&(1<<(fontWidth-1-col)) != 0 {
					for sy := 0; sy < scale; sy++ {
						for sx := 0; sx < scale; sx++ {
							m.setPixelScalar(cursorX+col*scale+sx, top+row*scale+sy, color)
						}
					}
				}
			}
		}
		cursorX += (fontWidth + 1) * scale
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func sign(v int) int {
	switch {
	case v > 0:
		return 1
	case v < 0:
		return -1
	default:
		return 0
	}
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func sortFloats(s []float64) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j-1] > s[j]; j-- {
			s[j-1], s[j] = s[j], s[j-1]
		}
	}
}
