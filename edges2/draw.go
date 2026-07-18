package edges2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// edges2SetIfInside writes value at (y,x) when the coordinate is inside dst.
func edges2SetIfInside(dst *cv.Mat, y, x int, value uint8) {
	if y >= 0 && y < dst.Rows && x >= 0 && x < dst.Cols {
		dst.Data[y*dst.Cols+x] = value
	}
}

// edges2DrawSegment rasterises a segment into dst with Bresenham's algorithm.
func edges2DrawSegment(dst *cv.Mat, s Segment, value uint8) {
	x0 := int(math.Round(s.X1))
	y0 := int(math.Round(s.Y1))
	x1 := int(math.Round(s.X2))
	y1 := int(math.Round(s.Y2))
	dx := int(math.Abs(float64(x1 - x0)))
	dy := -int(math.Abs(float64(y1 - y0)))
	sx := 1
	if x0 > x1 {
		sx = -1
	}
	sy := 1
	if y0 > y1 {
		sy = -1
	}
	err := dx + dy
	for {
		edges2SetIfInside(dst, y0, x0, value)
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

// DrawSegments rasterises every segment into a copy of img with the given
// intensity value and returns the result, leaving the input untouched. It
// panics on multi-channel input.
func DrawSegments(img *cv.Mat, segs []Segment, value uint8) *cv.Mat {
	edges2RequireGray(img, "DrawSegments")
	dst := img.Clone()
	for _, s := range segs {
		edges2DrawSegment(dst, s, value)
	}
	return dst
}

// DrawLines clips each infinite line to the image, rasterises the visible
// portion into a copy of img with the given intensity value and returns the
// result, leaving the input untouched. It panics on multi-channel input.
func DrawLines(img *cv.Mat, lines []Line, value uint8) *cv.Mat {
	edges2RequireGray(img, "DrawLines")
	dst := img.Clone()
	for _, l := range lines {
		if seg, ok := LineToSegment(l, img.Rows, img.Cols); ok {
			edges2DrawSegment(dst, seg, value)
		}
	}
	return dst
}

// DrawCircles rasterises the outline of every circle into a copy of img with
// the given intensity value (using the midpoint circle algorithm) and returns
// the result, leaving the input untouched. It panics on multi-channel input.
func DrawCircles(img *cv.Mat, circles []Circle, value uint8) *cv.Mat {
	edges2RequireGray(img, "DrawCircles")
	dst := img.Clone()
	for _, c := range circles {
		cx := int(math.Round(c.X))
		cy := int(math.Round(c.Y))
		r := int(math.Round(c.Radius))
		x := r
		y := 0
		e := 1 - x
		for x >= y {
			for _, p := range [][2]int{
				{cy + y, cx + x}, {cy + y, cx - x}, {cy - y, cx + x}, {cy - y, cx - x},
				{cy + x, cx + y}, {cy + x, cx - y}, {cy - x, cx + y}, {cy - x, cx - y},
			} {
				edges2SetIfInside(dst, p[0], p[1], value)
			}
			y++
			if e < 0 {
				e += 2*y + 1
			} else {
				x--
				e += 2*(y-x) + 1
			}
		}
	}
	return dst
}
