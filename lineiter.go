package cv

// LineIterator walks the integer pixels along the segment from one point to
// another using Bresenham's algorithm, mirroring OpenCV's cv::LineIterator. It
// yields every pixel from the start point through the end point inclusive.
type LineIterator struct {
	x, y   int
	x1, y1 int
	sx, sy int
	dx, dy int
	err    int
	count  int
	total  int
	done   bool
}

// NewLineIterator creates an 8-connected iterator over the segment (x0, y0) to
// (x1, y1). The iterator starts positioned on the first pixel; call Pos to read
// it and Next to advance.
func NewLineIterator(x0, y0, x1, y1 int) *LineIterator {
	dx := absCV(x1 - x0)
	dy := absCV(y1 - y0)
	sx := 1
	if x1 < x0 {
		sx = -1
	}
	sy := 1
	if y1 < y0 {
		sy = -1
	}
	total := dx
	if dy > dx {
		total = dy
	}
	total++
	return &LineIterator{
		x: x0, y: y0, x1: x1, y1: y1,
		sx: sx, sy: sy, dx: dx, dy: dy,
		err: dx - dy, total: total,
	}
}

// Pos returns the pixel the iterator currently points at.
func (it *LineIterator) Pos() Point { return Point{X: it.x, Y: it.y} }

// Count returns the total number of pixels the iterator will visit.
func (it *LineIterator) Count() int { return it.total }

// Valid reports whether the iterator still points at a pixel of the segment.
func (it *LineIterator) Valid() bool { return !it.done && it.count < it.total }

// Next advances to the following pixel and reports whether the iterator is
// still valid afterwards.
func (it *LineIterator) Next() bool {
	if it.done {
		return false
	}
	it.count++
	if it.count >= it.total {
		it.done = true
		return false
	}
	e2 := 2 * it.err
	if e2 > -it.dy {
		it.err -= it.dy
		it.x += it.sx
	}
	if e2 < it.dx {
		it.err += it.dx
		it.y += it.sy
	}
	return true
}

// Points returns every pixel along the segment as a slice, from start to end
// inclusive. It does not consume the receiver's current position.
func (it *LineIterator) Points() []Point {
	clone := *it
	pts := make([]Point, 0, clone.total)
	for clone.Valid() {
		pts = append(pts, clone.Pos())
		clone.Next()
	}
	return pts
}

// absCV returns the absolute value of x.
func absCV(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
