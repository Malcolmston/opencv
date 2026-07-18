package geom_cv

import (
	"sort"

	cv "github.com/malcolmston/opencv"
)

// geom_cvScanFill visits every pixel (x, y) whose center lies inside the simple
// polygon, using an even-odd scanline rule that matches [PointInPolygon]. Pixels
// are clipped to the rectangle [0,width) × [0,height).
func geom_cvScanFill(width, height int, poly []cv.Point2f, visit func(x, y int)) {
	n := len(poly)
	if n < 3 {
		return
	}
	for y := 0; y < height; y++ {
		yc := float64(y) + 0.5
		var xs []float64
		for i := 0; i < n; i++ {
			a := poly[i]
			b := poly[(i+1)%n]
			if (a.Y > yc) != (b.Y > yc) {
				x := a.X + (yc-a.Y)/(b.Y-a.Y)*(b.X-a.X)
				xs = append(xs, x)
			}
		}
		if len(xs) < 2 {
			continue
		}
		sort.Float64s(xs)
		for i := 0; i+1 < len(xs); i += 2 {
			xL, xR := xs[i], xs[i+1]
			for x := 0; x < width; x++ {
				xc := float64(x) + 0.5
				if xc > xL && xc < xR {
					visit(x, y)
				}
			}
		}
	}
}

// PolygonMask rasterizes the simple polygon into a new single-channel
// [github.com/malcolmston/opencv.Mat] of the given size, setting every pixel
// whose center is inside the polygon to 255 and the rest to 0. It panics if
// width or height is not positive.
func PolygonMask(width, height int, poly []cv.Point2f) *cv.Mat {
	if width <= 0 || height <= 0 {
		panic("geom_cv: PolygonMask requires positive width and height")
	}
	m := cv.NewMat(height, width, 1)
	geom_cvScanFill(width, height, poly, func(x, y int) {
		m.Set(y, x, 0, 255)
	})
	return m
}

// FillPolygon fills the interior of the simple polygon in the existing image m,
// writing value into every channel of each covered pixel. Pixels outside the
// image bounds are ignored. It panics if m is nil or empty.
func FillPolygon(m *cv.Mat, poly []cv.Point2f, value uint8) {
	if m.Empty() {
		panic("geom_cv: FillPolygon on empty Mat")
	}
	geom_cvScanFill(m.Cols, m.Rows, poly, func(x, y int) {
		for c := 0; c < m.Channels; c++ {
			m.Set(y, x, c, value)
		}
	})
}

// DrawPolygonOutline draws the closed polygon boundary in the image m using
// Bresenham line segments between consecutive vertices, writing value into every
// channel of each pixel on the outline. Pixels outside the image bounds are
// clipped. It panics if m is nil or empty.
func DrawPolygonOutline(m *cv.Mat, poly []cv.Point2f, value uint8) {
	if m.Empty() {
		panic("geom_cv: DrawPolygonOutline on empty Mat")
	}
	n := len(poly)
	if n < 2 {
		return
	}
	for i := 0; i < n; i++ {
		a := ToPoint(poly[i])
		b := ToPoint(poly[(i+1)%n])
		geom_cvBresenham(a, b, func(x, y int) {
			if x >= 0 && x < m.Cols && y >= 0 && y < m.Rows {
				for c := 0; c < m.Channels; c++ {
					m.Set(y, x, c, value)
				}
			}
		})
	}
}

// geom_cvBresenham walks the integer pixels of the segment from a to b with
// Bresenham's algorithm, calling visit for each.
func geom_cvBresenham(a, b cv.Point, visit func(x, y int)) {
	dx := b.X - a.X
	if dx < 0 {
		dx = -dx
	}
	dy := b.Y - a.Y
	if dy < 0 {
		dy = -dy
	}
	sx := 1
	if a.X > b.X {
		sx = -1
	}
	sy := 1
	if a.Y > b.Y {
		sy = -1
	}
	err := dx - dy
	x, y := a.X, a.Y
	for {
		visit(x, y)
		if x == b.X && y == b.Y {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x += sx
		}
		if e2 < dx {
			err += dx
			y += sy
		}
	}
}
