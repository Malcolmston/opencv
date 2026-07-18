package contours2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ArcLength returns the perimeter (or curve length) of a set of points. When
// closed is true the segment from the last point back to the first is included;
// otherwise the curve is treated as an open polyline. This mirrors OpenCV's
// arcLength.
func ArcLength(curve []cv.Point, closed bool) float64 {
	n := len(curve)
	if n < 2 {
		return 0
	}
	var total float64
	for i := 0; i < n-1; i++ {
		total += math.Hypot(float64(curve[i+1].X-curve[i].X), float64(curve[i+1].Y-curve[i].Y))
	}
	if closed {
		total += math.Hypot(float64(curve[0].X-curve[n-1].X), float64(curve[0].Y-curve[n-1].Y))
	}
	return total
}

// ContourAreaSigned returns the signed polygon area of a contour using the
// shoelace formula. In image coordinates (y increasing downward) a clockwise
// contour yields a positive value and a counter-clockwise contour a negative
// one. Fewer than three points give 0.
func ContourAreaSigned(contour []cv.Point) float64 {
	n := len(contour)
	if n < 3 {
		return 0
	}
	var s float64
	for i := 0; i < n; i++ {
		a := contour[i]
		b := contour[(i+1)%n]
		s += float64(a.X)*float64(b.Y) - float64(b.X)*float64(a.Y)
	}
	return s / 2
}

// ContourArea returns the non-negative area enclosed by a contour, computed via
// the shoelace formula. This mirrors OpenCV's contourArea (which returns the
// magnitude). Fewer than three points give 0.
func ContourArea(contour []cv.Point) float64 {
	return math.Abs(ContourAreaSigned(contour))
}

// BoundingRect returns the smallest axis-aligned integer rectangle that
// contains all the given points. It panics on an empty point set. This mirrors
// OpenCV's boundingRect.
func BoundingRect(pts []cv.Point) Rect {
	if len(pts) == 0 {
		panic("contours2: BoundingRect on empty point set")
	}
	minX, minY := pts[0].X, pts[0].Y
	maxX, maxY := pts[0].X, pts[0].Y
	for _, p := range pts[1:] {
		if p.X < minX {
			minX = p.X
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}
	return Rect{X: minX, Y: minY, Width: maxX - minX + 1, Height: maxY - minY + 1}
}

// ExtremePoints returns the leftmost, rightmost, topmost and bottommost points
// of a contour. When several points share an extreme coordinate the first one
// encountered in contour order is returned. It panics on an empty point set.
func ExtremePoints(contour []cv.Point) (left, right, top, bottom cv.Point) {
	if len(contour) == 0 {
		panic("contours2: ExtremePoints on empty point set")
	}
	left, right, top, bottom = contour[0], contour[0], contour[0], contour[0]
	for _, p := range contour[1:] {
		if p.X < left.X {
			left = p
		}
		if p.X > right.X {
			right = p
		}
		if p.Y < top.Y {
			top = p
		}
		if p.Y > bottom.Y {
			bottom = p
		}
	}
	return left, right, top, bottom
}

// PointPolygonTest reports the relationship between a point and a polygon given
// by its vertices. When measureDist is false it returns +1 if the point is
// inside, -1 if outside and 0 if on an edge. When measureDist is true it returns
// the signed distance to the nearest edge (positive inside, negative outside).
// This mirrors OpenCV's pointPolygonTest.
func PointPolygonTest(contour []cv.Point, pt cv.Point, measureDist bool) float64 {
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
	onEdge := false
	for i := 0; i < n; i++ {
		a := contour[i]
		b := contour[(i+1)%n]
		ax, ay := float64(a.X), float64(a.Y)
		bx, by := float64(b.X), float64(b.Y)
		if (ay > py) != (by > py) {
			xCross := ax + (py-ay)/(by-ay)*(bx-ax)
			if px < xCross {
				inside = !inside
			}
		}
		d := contours2segDist(px, py, ax, ay, bx, by)
		if d < minDist {
			minDist = d
		}
		if d < 1e-9 {
			onEdge = true
		}
	}
	if !measureDist {
		if onEdge {
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
