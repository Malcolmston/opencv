package draw2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// DrawContours draws each contour in contours as a closed polyline. contourIdx
// selects a single contour to draw; pass a negative value to draw them all.
// thickness controls stroke width; pass [Filled] to fill the contours with
// [FillContours] instead.
func DrawContours(m *cv.Mat, contours [][]cv.Point, contourIdx int, color cv.Scalar, thickness int) {
	if thickness < 0 {
		if contourIdx >= 0 && contourIdx < len(contours) {
			FillContours(m, [][]cv.Point{contours[contourIdx]}, color)
		} else {
			FillContours(m, contours, color)
		}
		return
	}
	for i, c := range contours {
		if contourIdx >= 0 && i != contourIdx {
			continue
		}
		Polyline(m, c, true, color, thickness)
	}
}

// FillContours fills every contour with a solid colour using the even-odd
// rule, so nested contours (holes) cancel just as OpenCV's drawContours does
// with a negative thickness.
func FillContours(m *cv.Mat, contours [][]cv.Point, color cv.Scalar) {
	FillPolygon(m, contours, color)
}

// OverlayContours fills every contour with color at the given opacity,
// alpha-compositing the fill over the image. It is useful for highlighting
// detected regions without hiding the underlying pixels. alpha is clamped to
// [0,1].
func OverlayContours(m *cv.Mat, contours [][]cv.Point, color cv.Scalar, alpha float64) {
	if alpha <= 0 {
		return
	}
	mask := cv.NewMat(m.Rows, m.Cols, 1)
	FillPolygon(mask, contours, cv.Scalar{255, 255, 255, 255})
	if alpha > 1 {
		alpha = 1
	}
	for p := 0; p < m.Rows*m.Cols; p++ {
		if mask.Data[p] == 0 {
			continue
		}
		base := p * m.Channels
		for c := 0; c < m.Channels; c++ {
			m.Data[base+c] = draw2clamp8(float64(m.Data[base+c])*(1-alpha) + color[c]*alpha)
		}
	}
}

// BoundingRect returns the top-left and bottom-right corners of the smallest
// axis-aligned rectangle enclosing pts. It returns two zero points for an
// empty input.
func BoundingRect(pts []cv.Point) (tl, br cv.Point) {
	if len(pts) == 0 {
		return cv.Point{}, cv.Point{}
	}
	minX, minY := math.MaxInt, math.MaxInt
	maxX, maxY := math.MinInt, math.MinInt
	for _, p := range pts {
		if p.X < minX {
			minX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}
	return cv.Point{X: minX, Y: minY}, cv.Point{X: maxX, Y: maxY}
}

// DrawBoundingBoxes draws the axis-aligned bounding rectangle of each contour.
// thickness controls stroke width.
func DrawBoundingBoxes(m *cv.Mat, contours [][]cv.Point, color cv.Scalar, thickness int) {
	for _, c := range contours {
		if len(c) == 0 {
			continue
		}
		tl, br := BoundingRect(c)
		Rectangle(m, tl, br, color, thickness)
	}
}

// Centroid returns the arithmetic mean of the points as an integer coordinate.
// It returns the zero point for an empty input.
func Centroid(pts []cv.Point) cv.Point {
	if len(pts) == 0 {
		return cv.Point{}
	}
	var sx, sy int
	for _, p := range pts {
		sx += p.X
		sy += p.Y
	}
	return cv.Point{X: sx / len(pts), Y: sy / len(pts)}
}

// PolygonArea returns the absolute area enclosed by the polygon pts using the
// shoelace formula. The polygon is treated as implicitly closed.
func PolygonArea(pts []cv.Point) float64 {
	n := len(pts)
	if n < 3 {
		return 0
	}
	var sum int
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		sum += pts[i].X*pts[j].Y - pts[j].X*pts[i].Y
	}
	return math.Abs(float64(sum)) / 2
}

// PolygonPerimeter returns the total edge length of the polygon pts. When
// closed is true the closing edge from the last point back to the first is
// included.
func PolygonPerimeter(pts []cv.Point, closed bool) float64 {
	n := len(pts)
	if n < 2 {
		return 0
	}
	var total float64
	for i := 0; i+1 < n; i++ {
		total += math.Hypot(float64(pts[i+1].X-pts[i].X), float64(pts[i+1].Y-pts[i].Y))
	}
	if closed {
		total += math.Hypot(float64(pts[0].X-pts[n-1].X), float64(pts[0].Y-pts[n-1].Y))
	}
	return total
}
