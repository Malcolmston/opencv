package contours2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// contours2requireGray panics if src is nil, empty or not single-channel.
func contours2requireGray(src *cv.Mat, name string) {
	if src == nil || src.Empty() {
		panic("contours2: " + name + " requires a non-empty image")
	}
	if src.Channels != 1 {
		panic("contours2: " + name + " requires a single-channel image")
	}
}

// contours2abs returns the absolute value of an int.
func contours2abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// contours2sign returns -1, 0 or +1 according to the sign of v.
func contours2sign(v int) int {
	switch {
	case v > 0:
		return 1
	case v < 0:
		return -1
	default:
		return 0
	}
}

// contours2crossi returns the z component of the cross product of (b-a) and
// (c-a) for integer points; its sign gives the turn direction a->b->c.
func contours2crossi(a, b, c cv.Point) int {
	return (b.X-a.X)*(c.Y-a.Y) - (b.Y-a.Y)*(c.X-a.X)
}

// contours2distToLine returns the perpendicular distance from point p to the
// infinite line through a and b (integer points). If a == b it returns the
// point-to-point distance.
func contours2distToLine(p, a, b cv.Point) float64 {
	dx := float64(b.X - a.X)
	dy := float64(b.Y - a.Y)
	if dx == 0 && dy == 0 {
		return math.Hypot(float64(p.X-a.X), float64(p.Y-a.Y))
	}
	num := math.Abs(dy*float64(p.X-a.X) - dx*float64(p.Y-a.Y))
	return num / math.Hypot(dx, dy)
}

// contours2segDist returns the Euclidean distance from (px,py) to the segment
// (ax,ay)-(bx,by).
func contours2segDist(px, py, ax, ay, bx, by float64) float64 {
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
