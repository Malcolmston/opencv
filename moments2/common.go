package moments2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// moments2requireGray panics if src is nil, empty or not single-channel. All
// raster routines in this package require a single-channel image.
func moments2requireGray(src *cv.Mat, name string) {
	if src == nil || src.Empty() {
		panic("moments2: " + name + " requires a non-empty image")
	}
	if src.Channels != 1 {
		panic("moments2: " + name + " requires a single-channel image")
	}
}

// moments2sample returns the sample value at (y, x) of a single-channel Mat as
// a float64. The caller guarantees the coordinates are in range.
func moments2sample(src *cv.Mat, y, x int) float64 {
	return float64(src.Data[y*src.Cols+x])
}

// moments2hypot is math.Hypot, named locally for brevity in inner loops.
func moments2hypot(a, b float64) float64 { return math.Hypot(a, b) }

// moments2factorial returns n! as a float64 for small non-negative n.
func moments2factorial(n int) float64 {
	f := 1.0
	for i := 2; i <= n; i++ {
		f *= float64(i)
	}
	return f
}

// moments2closePolygon returns a copy of pts with the first vertex appended if
// the polygon is not already closed and has at least one point.
func moments2closePolygon(pts []cv.Point) []cv.Point {
	n := len(pts)
	if n == 0 {
		return pts
	}
	if pts[0] == pts[n-1] {
		out := make([]cv.Point, n)
		copy(out, pts)
		return out
	}
	out := make([]cv.Point, 0, n+1)
	out = append(out, pts...)
	out = append(out, pts[0])
	return out
}

// moments2sign returns -1, 0 or +1 according to the sign of v.
func moments2sign(v float64) float64 {
	switch {
	case v > 0:
		return 1
	case v < 0:
		return -1
	default:
		return 0
	}
}
