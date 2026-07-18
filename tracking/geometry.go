package tracking

import (
	"fmt"
	"math"
)

// Point2f is a floating-point image coordinate where X is the column and Y is
// the row. It is the sub-pixel value type used by the optical-flow solvers and
// the centroid tracker.
type Point2f struct {
	// X is the horizontal (column) coordinate.
	X float64
	// Y is the vertical (row) coordinate.
	Y float64
}

// Pt2f returns the Point2f with the given coordinates.
func Pt2f(x, y float64) Point2f { return Point2f{X: x, Y: y} }

// Add returns the component-wise sum of p and o.
func (p Point2f) Add(o Point2f) Point2f { return Point2f{p.X + o.X, p.Y + o.Y} }

// Sub returns the component-wise difference p-o.
func (p Point2f) Sub(o Point2f) Point2f { return Point2f{p.X - o.X, p.Y - o.Y} }

// Scale returns p with both components multiplied by s.
func (p Point2f) Scale(s float64) Point2f { return Point2f{p.X * s, p.Y * s} }

// Norm returns the Euclidean length of p treated as a vector from the origin.
func (p Point2f) Norm() float64 { return math.Hypot(p.X, p.Y) }

// Distance returns the Euclidean distance between p and o.
func (p Point2f) Distance(o Point2f) float64 { return math.Hypot(p.X-o.X, p.Y-o.Y) }

// String renders the point as "(x, y)".
func (p Point2f) String() string { return fmt.Sprintf("(%.4g, %.4g)", p.X, p.Y) }

// Rect is an axis-aligned integer bounding box with its top-left corner at
// (X, Y) and the given Width and Height. It is the window and detection-box type
// used throughout the package.
type Rect struct {
	// X is the left edge (column of the top-left corner).
	X int
	// Y is the top edge (row of the top-left corner).
	Y int
	// Width is the box width in pixels.
	Width int
	// Height is the box height in pixels.
	Height int
}

// NewRect returns the Rect with the given position and size.
func NewRect(x, y, width, height int) Rect {
	return Rect{X: x, Y: y, Width: width, Height: height}
}

// Area returns the number of pixels covered by the rectangle, clamped to zero
// for degenerate boxes.
func (r Rect) Area() int {
	if r.Width <= 0 || r.Height <= 0 {
		return 0
	}
	return r.Width * r.Height
}

// Empty reports whether the rectangle has no area.
func (r Rect) Empty() bool { return r.Width <= 0 || r.Height <= 0 }

// Right returns the exclusive right edge X+Width.
func (r Rect) Right() int { return r.X + r.Width }

// Bottom returns the exclusive bottom edge Y+Height.
func (r Rect) Bottom() int { return r.Y + r.Height }

// Center returns the geometric center of the rectangle as a sub-pixel point.
func (r Rect) Center() Point2f {
	return Point2f{X: float64(r.X) + float64(r.Width)/2, Y: float64(r.Y) + float64(r.Height)/2}
}

// Contains reports whether the integer point (x, y) lies inside the rectangle,
// treating the right and bottom edges as exclusive.
func (r Rect) Contains(x, y int) bool {
	return x >= r.X && x < r.Right() && y >= r.Y && y < r.Bottom()
}

// Intersect returns the overlapping rectangle of r and o. When the two boxes do
// not overlap the result is a zero-area rectangle (Empty reports true).
func (r Rect) Intersect(o Rect) Rect {
	x0 := max(r.X, o.X)
	y0 := max(r.Y, o.Y)
	x1 := min(r.Right(), o.Right())
	y1 := min(r.Bottom(), o.Bottom())
	if x1 <= x0 || y1 <= y0 {
		return Rect{}
	}
	return Rect{X: x0, Y: y0, Width: x1 - x0, Height: y1 - y0}
}

// Union returns the smallest rectangle that contains both r and o. If either
// box is empty the other is returned unchanged.
func (r Rect) Union(o Rect) Rect {
	if r.Empty() {
		return o
	}
	if o.Empty() {
		return r
	}
	x0 := min(r.X, o.X)
	y0 := min(r.Y, o.Y)
	x1 := max(r.Right(), o.Right())
	y1 := max(r.Bottom(), o.Bottom())
	return Rect{X: x0, Y: y0, Width: x1 - x0, Height: y1 - y0}
}

// IoU returns the intersection-over-union overlap of r and o, a value in [0, 1].
// It is zero when the boxes are disjoint or either is empty.
func (r Rect) IoU(o Rect) float64 {
	inter := r.Intersect(o).Area()
	if inter == 0 {
		return 0
	}
	union := r.Area() + o.Area() - inter
	if union <= 0 {
		return 0
	}
	return float64(inter) / float64(union)
}

// String renders the rectangle as "[WxH from (X, Y)]".
func (r Rect) String() string {
	return fmt.Sprintf("[%dx%d from (%d, %d)]", r.Width, r.Height, r.X, r.Y)
}

// clampTo shifts and shrinks the rectangle so it lies fully inside a rows*cols
// image, preserving as much of the original extent as possible.
func (r Rect) clampTo(rows, cols int) Rect {
	x0 := max(0, r.X)
	y0 := max(0, r.Y)
	x1 := min(cols, r.Right())
	y1 := min(rows, r.Bottom())
	if x1 <= x0 || y1 <= y0 {
		return Rect{}
	}
	return Rect{X: x0, Y: y0, Width: x1 - x0, Height: y1 - y0}
}

// IoU returns the intersection-over-union overlap of two rectangles, a value in
// [0, 1]. It is a free-function convenience equivalent to a.IoU(b).
func IoU(a, b Rect) float64 { return a.IoU(b) }

// RotatedRect is an oriented rectangle described by its center, size and
// rotation. Angle is measured in degrees, counter-clockwise positive. It is the
// result type of [CamShift].
type RotatedRect struct {
	// Center is the sub-pixel center of the box.
	Center Point2f
	// Width is the box extent along its (rotated) local X axis.
	Width float64
	// Height is the box extent along its (rotated) local Y axis.
	Height float64
	// Angle is the rotation in degrees, counter-clockwise positive.
	Angle float64
}

// BoundingRect returns the smallest axis-aligned integer rectangle that fully
// contains the rotated rectangle.
func (r RotatedRect) BoundingRect() Rect {
	rad := r.Angle * math.Pi / 180
	c := math.Cos(rad)
	s := math.Sin(rad)
	hw := r.Width / 2
	hh := r.Height / 2
	// Half-extents of the axis-aligned bound of a rotated box.
	ex := math.Abs(hw*c) + math.Abs(hh*s)
	ey := math.Abs(hw*s) + math.Abs(hh*c)
	x0 := int(math.Floor(r.Center.X - ex))
	y0 := int(math.Floor(r.Center.Y - ey))
	x1 := int(math.Ceil(r.Center.X + ex))
	y1 := int(math.Ceil(r.Center.Y + ey))
	return Rect{X: x0, Y: y0, Width: x1 - x0, Height: y1 - y0}
}

// TermCriteria describes when an iterative algorithm should stop. Iteration
// halts once MaxCount iterations have run or the per-iteration change falls at
// or below Epsilon, whichever happens first. A non-positive field disables that
// test; at least one field should be positive for a bounded algorithm.
type TermCriteria struct {
	// MaxCount is the maximum number of iterations (<= 0 disables the count test).
	MaxCount int
	// Epsilon is the convergence threshold on the iteration step (<= 0 disables
	// the accuracy test).
	Epsilon float64
}

// NewTermCriteria returns a TermCriteria with the given iteration cap and
// accuracy threshold.
func NewTermCriteria(maxCount int, epsilon float64) TermCriteria {
	return TermCriteria{MaxCount: maxCount, Epsilon: epsilon}
}

// reached reports whether iteration should stop after completing the zero-based
// iteration iter with the given step magnitude.
func (t TermCriteria) reached(iter int, step float64) bool {
	if t.MaxCount > 0 && iter+1 >= t.MaxCount {
		return true
	}
	if t.Epsilon > 0 && step <= t.Epsilon {
		return true
	}
	return false
}

// iterCap returns a usable upper bound on iterations, defaulting to fallback
// when MaxCount is non-positive.
func (t TermCriteria) iterCap(fallback int) int {
	if t.MaxCount > 0 {
		return t.MaxCount
	}
	return fallback
}
