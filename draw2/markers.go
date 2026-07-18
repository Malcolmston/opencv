package draw2

import (
	cv "github.com/malcolmston/opencv"
)

// MarkerType selects the glyph drawn by [DrawMarker].
type MarkerType int

const (
	// MarkerCross draws an upright plus-shaped cross (+).
	MarkerCross MarkerType = iota
	// MarkerTiltedCross draws a diagonal cross (x).
	MarkerTiltedCross
	// MarkerStar draws a combined upright and diagonal cross (*).
	MarkerStar
	// MarkerDiamond draws a diamond outline.
	MarkerDiamond
	// MarkerSquare draws a square outline.
	MarkerSquare
	// MarkerTriangleUp draws an upward-pointing triangle outline.
	MarkerTriangleUp
	// MarkerTriangleDown draws a downward-pointing triangle outline.
	MarkerTriangleDown
	// MarkerCircle draws a circle outline.
	MarkerCircle
)

// DrawMarker draws a marker glyph of the given type centred at center. size is
// the marker's half-extent in pixels and thickness controls stroke width.
func DrawMarker(m *cv.Mat, center cv.Point, marker MarkerType, size int, color cv.Scalar, thickness int) {
	if size < 1 {
		size = 1
	}
	if thickness < 1 {
		thickness = 1
	}
	cx, cy := center.X, center.Y
	switch marker {
	case MarkerCross:
		Line(m, cv.Point{X: cx - size, Y: cy}, cv.Point{X: cx + size, Y: cy}, color, thickness)
		Line(m, cv.Point{X: cx, Y: cy - size}, cv.Point{X: cx, Y: cy + size}, color, thickness)
	case MarkerTiltedCross:
		Line(m, cv.Point{X: cx - size, Y: cy - size}, cv.Point{X: cx + size, Y: cy + size}, color, thickness)
		Line(m, cv.Point{X: cx - size, Y: cy + size}, cv.Point{X: cx + size, Y: cy - size}, color, thickness)
	case MarkerStar:
		Line(m, cv.Point{X: cx - size, Y: cy}, cv.Point{X: cx + size, Y: cy}, color, thickness)
		Line(m, cv.Point{X: cx, Y: cy - size}, cv.Point{X: cx, Y: cy + size}, color, thickness)
		Line(m, cv.Point{X: cx - size, Y: cy - size}, cv.Point{X: cx + size, Y: cy + size}, color, thickness)
		Line(m, cv.Point{X: cx - size, Y: cy + size}, cv.Point{X: cx + size, Y: cy - size}, color, thickness)
	case MarkerDiamond:
		pts := []cv.Point{
			{X: cx, Y: cy - size}, {X: cx + size, Y: cy},
			{X: cx, Y: cy + size}, {X: cx - size, Y: cy},
		}
		Polyline(m, pts, true, color, thickness)
	case MarkerSquare:
		Rectangle(m, cv.Point{X: cx - size, Y: cy - size}, cv.Point{X: cx + size, Y: cy + size}, color, thickness)
	case MarkerTriangleUp:
		pts := []cv.Point{
			{X: cx, Y: cy - size}, {X: cx + size, Y: cy + size}, {X: cx - size, Y: cy + size},
		}
		Polyline(m, pts, true, color, thickness)
	case MarkerTriangleDown:
		pts := []cv.Point{
			{X: cx, Y: cy + size}, {X: cx + size, Y: cy - size}, {X: cx - size, Y: cy - size},
		}
		Polyline(m, pts, true, color, thickness)
	case MarkerCircle:
		Circle(m, center, size, color, thickness)
	}
}
