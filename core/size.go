package core

import "fmt"

// Size2i is a width/height pair with integer dimensions, mirroring cv::Size.
type Size2i struct {
	Width  int
	Height int
}

// Sz2i builds a Size2i from width and height.
func Sz2i(w, h int) Size2i { return Size2i{w, h} }

// Area returns Width*Height.
func (s Size2i) Area() int { return s.Width * s.Height }

// Empty reports whether either dimension is non-positive.
func (s Size2i) Empty() bool { return s.Width <= 0 || s.Height <= 0 }

// Add returns the element-wise sum of two sizes.
func (s Size2i) Add(o Size2i) Size2i { return Size2i{s.Width + o.Width, s.Height + o.Height} }

// Sub returns the element-wise difference of two sizes.
func (s Size2i) Sub(o Size2i) Size2i { return Size2i{s.Width - o.Width, s.Height - o.Height} }

// Scale returns the size with both dimensions multiplied by k (truncated).
func (s Size2i) Scale(k float64) Size2i {
	return Size2i{int(float64(s.Width) * k), int(float64(s.Height) * k)}
}

// Aspect returns Width/Height as a float64 (0 when Height is 0).
func (s Size2i) Aspect() float64 {
	if s.Height == 0 {
		return 0
	}
	return float64(s.Width) / float64(s.Height)
}

// Equals reports whether two sizes are identical.
func (s Size2i) Equals(o Size2i) bool { return s.Width == o.Width && s.Height == o.Height }

// String renders the size as WxH.
func (s Size2i) String() string { return fmt.Sprintf("%dx%d", s.Width, s.Height) }

// Size2f is a width/height pair with float32 dimensions, mirroring cv::Size2f.
type Size2f struct {
	Width  float32
	Height float32
}

// Sz2f builds a Size2f from width and height.
func Sz2f(w, h float32) Size2f { return Size2f{w, h} }

// Area returns Width*Height.
func (s Size2f) Area() float32 { return s.Width * s.Height }

// Empty reports whether either dimension is non-positive.
func (s Size2f) Empty() bool { return s.Width <= 0 || s.Height <= 0 }

// Add returns the element-wise sum of two sizes.
func (s Size2f) Add(o Size2f) Size2f { return Size2f{s.Width + o.Width, s.Height + o.Height} }

// Sub returns the element-wise difference of two sizes.
func (s Size2f) Sub(o Size2f) Size2f { return Size2f{s.Width - o.Width, s.Height - o.Height} }

// Scale returns the size with both dimensions multiplied by k.
func (s Size2f) Scale(k float32) Size2f { return Size2f{s.Width * k, s.Height * k} }

// Aspect returns Width/Height as a float64 (0 when Height is 0).
func (s Size2f) Aspect() float64 {
	if s.Height == 0 {
		return 0
	}
	return float64(s.Width) / float64(s.Height)
}

// Equals reports whether two sizes are identical.
func (s Size2f) Equals(o Size2f) bool { return s.Width == o.Width && s.Height == o.Height }

// String renders the size as WxH.
func (s Size2f) String() string { return fmt.Sprintf("%gx%g", s.Width, s.Height) }

// Size2d is a width/height pair with float64 dimensions, mirroring cv::Size2d.
type Size2d struct {
	Width  float64
	Height float64
}

// Sz2d builds a Size2d from width and height.
func Sz2d(w, h float64) Size2d { return Size2d{w, h} }

// Area returns Width*Height.
func (s Size2d) Area() float64 { return s.Width * s.Height }

// Empty reports whether either dimension is non-positive.
func (s Size2d) Empty() bool { return s.Width <= 0 || s.Height <= 0 }

// Add returns the element-wise sum of two sizes.
func (s Size2d) Add(o Size2d) Size2d { return Size2d{s.Width + o.Width, s.Height + o.Height} }

// Sub returns the element-wise difference of two sizes.
func (s Size2d) Sub(o Size2d) Size2d { return Size2d{s.Width - o.Width, s.Height - o.Height} }

// Scale returns the size with both dimensions multiplied by k.
func (s Size2d) Scale(k float64) Size2d { return Size2d{s.Width * k, s.Height * k} }

// Aspect returns Width/Height as a float64 (0 when Height is 0).
func (s Size2d) Aspect() float64 {
	if s.Height == 0 {
		return 0
	}
	return s.Width / s.Height
}

// Equals reports whether two sizes are identical.
func (s Size2d) Equals(o Size2d) bool { return s.Width == o.Width && s.Height == o.Height }

// String renders the size as WxH.
func (s Size2d) String() string { return fmt.Sprintf("%gx%g", s.Width, s.Height) }
