package core

import "fmt"

// Rect2i is an axis-aligned rectangle with an integer top-left corner and
// integer dimensions, mirroring cv::Rect. The region covers the columns
// [X, X+Width) and rows [Y, Y+Height).
type Rect2i struct {
	X      int
	Y      int
	Width  int
	Height int
}

// Rc2i builds a Rect2i from its corner and size.
func Rc2i(x, y, w, h int) Rect2i { return Rect2i{x, y, w, h} }

// TL returns the top-left corner.
func (r Rect2i) TL() Point2i { return Point2i{r.X, r.Y} }

// BR returns the bottom-right corner (exclusive: X+Width, Y+Height).
func (r Rect2i) BR() Point2i { return Point2i{r.X + r.Width, r.Y + r.Height} }

// Size returns the rectangle's size.
func (r Rect2i) Size() Size2i { return Size2i{r.Width, r.Height} }

// Area returns Width*Height.
func (r Rect2i) Area() int { return r.Width * r.Height }

// Empty reports whether the rectangle has non-positive area.
func (r Rect2i) Empty() bool { return r.Width <= 0 || r.Height <= 0 }

// Contains reports whether point p lies inside the rectangle.
func (r Rect2i) Contains(p Point2i) bool {
	return p.X >= r.X && p.X < r.X+r.Width && p.Y >= r.Y && p.Y < r.Y+r.Height
}

// And returns the intersection of r and o; an empty rectangle when they do not
// overlap. This mirrors OpenCV's Rect & operator.
func (r Rect2i) And(o Rect2i) Rect2i {
	x1 := maxInt(r.X, o.X)
	y1 := maxInt(r.Y, o.Y)
	x2 := minInt(r.X+r.Width, o.X+o.Width)
	y2 := minInt(r.Y+r.Height, o.Y+o.Height)
	if x2 <= x1 || y2 <= y1 {
		return Rect2i{}
	}
	return Rect2i{x1, y1, x2 - x1, y2 - y1}
}

// Or returns the smallest rectangle containing both r and o. When one operand
// is empty the other is returned, mirroring OpenCV's Rect | operator.
func (r Rect2i) Or(o Rect2i) Rect2i {
	if r.Empty() {
		return o
	}
	if o.Empty() {
		return r
	}
	x1 := minInt(r.X, o.X)
	y1 := minInt(r.Y, o.Y)
	x2 := maxInt(r.X+r.Width, o.X+o.Width)
	y2 := maxInt(r.Y+r.Height, o.Y+o.Height)
	return Rect2i{x1, y1, x2 - x1, y2 - y1}
}

// Shift returns the rectangle translated by point p.
func (r Rect2i) Shift(p Point2i) Rect2i { return Rect2i{r.X + p.X, r.Y + p.Y, r.Width, r.Height} }

// Equals reports whether two rectangles are identical.
func (r Rect2i) Equals(o Rect2i) bool {
	return r.X == o.X && r.Y == o.Y && r.Width == o.Width && r.Height == o.Height
}

// String renders the rectangle as [WxH from (X, Y)].
func (r Rect2i) String() string {
	return fmt.Sprintf("[%dx%d from (%d, %d)]", r.Width, r.Height, r.X, r.Y)
}

// Rect2f is an axis-aligned rectangle with float32 fields, mirroring cv::Rect2f.
type Rect2f struct {
	X      float32
	Y      float32
	Width  float32
	Height float32
}

// Rc2f builds a Rect2f from its corner and size.
func Rc2f(x, y, w, h float32) Rect2f { return Rect2f{x, y, w, h} }

// TL returns the top-left corner.
func (r Rect2f) TL() Point2f { return Point2f{r.X, r.Y} }

// BR returns the bottom-right corner.
func (r Rect2f) BR() Point2f { return Point2f{r.X + r.Width, r.Y + r.Height} }

// Size returns the rectangle's size.
func (r Rect2f) Size() Size2f { return Size2f{r.Width, r.Height} }

// Area returns Width*Height.
func (r Rect2f) Area() float32 { return r.Width * r.Height }

// Empty reports whether the rectangle has non-positive area.
func (r Rect2f) Empty() bool { return r.Width <= 0 || r.Height <= 0 }

// Contains reports whether point p lies inside the rectangle.
func (r Rect2f) Contains(p Point2f) bool {
	return p.X >= r.X && p.X < r.X+r.Width && p.Y >= r.Y && p.Y < r.Y+r.Height
}

// And returns the intersection of r and o; an empty rectangle when disjoint.
func (r Rect2f) And(o Rect2f) Rect2f {
	x1 := maxF32(r.X, o.X)
	y1 := maxF32(r.Y, o.Y)
	x2 := minF32(r.X+r.Width, o.X+o.Width)
	y2 := minF32(r.Y+r.Height, o.Y+o.Height)
	if x2 <= x1 || y2 <= y1 {
		return Rect2f{}
	}
	return Rect2f{x1, y1, x2 - x1, y2 - y1}
}

// Or returns the smallest rectangle containing both r and o.
func (r Rect2f) Or(o Rect2f) Rect2f {
	if r.Empty() {
		return o
	}
	if o.Empty() {
		return r
	}
	x1 := minF32(r.X, o.X)
	y1 := minF32(r.Y, o.Y)
	x2 := maxF32(r.X+r.Width, o.X+o.Width)
	y2 := maxF32(r.Y+r.Height, o.Y+o.Height)
	return Rect2f{x1, y1, x2 - x1, y2 - y1}
}

// Shift returns the rectangle translated by point p.
func (r Rect2f) Shift(p Point2f) Rect2f { return Rect2f{r.X + p.X, r.Y + p.Y, r.Width, r.Height} }

// Equals reports whether two rectangles are identical.
func (r Rect2f) Equals(o Rect2f) bool {
	return r.X == o.X && r.Y == o.Y && r.Width == o.Width && r.Height == o.Height
}

// String renders the rectangle as [WxH from (X, Y)].
func (r Rect2f) String() string {
	return fmt.Sprintf("[%gx%g from (%g, %g)]", r.Width, r.Height, r.X, r.Y)
}

// Rect2d is an axis-aligned rectangle with float64 fields, mirroring cv::Rect2d.
type Rect2d struct {
	X      float64
	Y      float64
	Width  float64
	Height float64
}

// Rc2d builds a Rect2d from its corner and size.
func Rc2d(x, y, w, h float64) Rect2d { return Rect2d{x, y, w, h} }

// TL returns the top-left corner.
func (r Rect2d) TL() Point2d { return Point2d{r.X, r.Y} }

// BR returns the bottom-right corner.
func (r Rect2d) BR() Point2d { return Point2d{r.X + r.Width, r.Y + r.Height} }

// Size returns the rectangle's size.
func (r Rect2d) Size() Size2d { return Size2d{r.Width, r.Height} }

// Area returns Width*Height.
func (r Rect2d) Area() float64 { return r.Width * r.Height }

// Empty reports whether the rectangle has non-positive area.
func (r Rect2d) Empty() bool { return r.Width <= 0 || r.Height <= 0 }

// Contains reports whether point p lies inside the rectangle.
func (r Rect2d) Contains(p Point2d) bool {
	return p.X >= r.X && p.X < r.X+r.Width && p.Y >= r.Y && p.Y < r.Y+r.Height
}

// And returns the intersection of r and o; an empty rectangle when disjoint.
func (r Rect2d) And(o Rect2d) Rect2d {
	x1 := maxF64(r.X, o.X)
	y1 := maxF64(r.Y, o.Y)
	x2 := minF64(r.X+r.Width, o.X+o.Width)
	y2 := minF64(r.Y+r.Height, o.Y+o.Height)
	if x2 <= x1 || y2 <= y1 {
		return Rect2d{}
	}
	return Rect2d{x1, y1, x2 - x1, y2 - y1}
}

// Or returns the smallest rectangle containing both r and o.
func (r Rect2d) Or(o Rect2d) Rect2d {
	if r.Empty() {
		return o
	}
	if o.Empty() {
		return r
	}
	x1 := minF64(r.X, o.X)
	y1 := minF64(r.Y, o.Y)
	x2 := maxF64(r.X+r.Width, o.X+o.Width)
	y2 := maxF64(r.Y+r.Height, o.Y+o.Height)
	return Rect2d{x1, y1, x2 - x1, y2 - y1}
}

// Shift returns the rectangle translated by point p.
func (r Rect2d) Shift(p Point2d) Rect2d { return Rect2d{r.X + p.X, r.Y + p.Y, r.Width, r.Height} }

// Equals reports whether two rectangles are identical.
func (r Rect2d) Equals(o Rect2d) bool {
	return r.X == o.X && r.Y == o.Y && r.Width == o.Width && r.Height == o.Height
}

// String renders the rectangle as [WxH from (X, Y)].
func (r Rect2d) String() string {
	return fmt.Sprintf("[%gx%g from (%g, %g)]", r.Width, r.Height, r.X, r.Y)
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxF32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func minF32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

func maxF64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func minF64(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}
