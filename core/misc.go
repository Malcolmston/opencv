package core

import (
	"fmt"
	"math"
)

// Scalar is a 4-element double vector, mirroring cv::Scalar. It is commonly
// used to carry per-channel values such as colours or means.
type Scalar [4]float64

// NewScalar builds a Scalar from up to four components; missing entries are
// zero and extra entries are ignored.
func NewScalar(v ...float64) Scalar {
	var s Scalar
	for i := 0; i < len(v) && i < 4; i++ {
		s[i] = v[i]
	}
	return s
}

// ScalarAll returns a Scalar with every component set to v.
func ScalarAll(v float64) Scalar { return Scalar{v, v, v, v} }

// Add returns the element-wise sum s+o.
func (s Scalar) Add(o Scalar) Scalar {
	return Scalar{s[0] + o[0], s[1] + o[1], s[2] + o[2], s[3] + o[3]}
}

// Sub returns the element-wise difference s-o.
func (s Scalar) Sub(o Scalar) Scalar {
	return Scalar{s[0] - o[0], s[1] - o[1], s[2] - o[2], s[3] - o[3]}
}

// Mul returns the scalar scaled by k.
func (s Scalar) Mul(k float64) Scalar { return Scalar{s[0] * k, s[1] * k, s[2] * k, s[3] * k} }

// MulElem returns the element-wise product s*o.
func (s Scalar) MulElem(o Scalar) Scalar {
	return Scalar{s[0] * o[0], s[1] * o[1], s[2] * o[2], s[3] * o[3]}
}

// Conj returns the quaternion-style conjugate (v0, -v1, -v2, -v3).
func (s Scalar) Conj() Scalar { return Scalar{s[0], -s[1], -s[2], -s[3]} }

// IsReal reports whether the last three components are all zero.
func (s Scalar) IsReal() bool { return s[1] == 0 && s[2] == 0 && s[3] == 0 }

// Equals reports whether two scalars are identical.
func (s Scalar) Equals(o Scalar) bool {
	return s[0] == o[0] && s[1] == o[1] && s[2] == o[2] && s[3] == o[3]
}

// String renders the scalar as [a, b, c, d].
func (s Scalar) String() string { return fmt.Sprintf("[%g, %g, %g, %g]", s[0], s[1], s[2], s[3]) }

// Complexd is a double-precision complex number, mirroring cv::Complexd.
type Complexd struct {
	Re float64
	Im float64
}

// NewComplexd builds a Complexd from real and imaginary parts.
func NewComplexd(re, im float64) Complexd { return Complexd{re, im} }

// Add returns c+o.
func (c Complexd) Add(o Complexd) Complexd { return Complexd{c.Re + o.Re, c.Im + o.Im} }

// Sub returns c-o.
func (c Complexd) Sub(o Complexd) Complexd { return Complexd{c.Re - o.Re, c.Im - o.Im} }

// Mul returns the complex product c*o.
func (c Complexd) Mul(o Complexd) Complexd {
	return Complexd{c.Re*o.Re - c.Im*o.Im, c.Re*o.Im + c.Im*o.Re}
}

// Div returns the complex quotient c/o.
func (c Complexd) Div(o Complexd) Complexd {
	d := o.Re*o.Re + o.Im*o.Im
	return Complexd{(c.Re*o.Re + c.Im*o.Im) / d, (c.Im*o.Re - c.Re*o.Im) / d}
}

// Conj returns the complex conjugate.
func (c Complexd) Conj() Complexd { return Complexd{c.Re, -c.Im} }

// Abs returns the modulus |c|.
func (c Complexd) Abs() float64 { return math.Hypot(c.Re, c.Im) }

// Arg returns the argument (phase angle) of c in radians.
func (c Complexd) Arg() float64 { return math.Atan2(c.Im, c.Re) }

// String renders the number as re+imi.
func (c Complexd) String() string { return fmt.Sprintf("(%g%+gi)", c.Re, c.Im) }

// Complexf is a single-precision complex number, mirroring cv::Complexf.
type Complexf struct {
	Re float32
	Im float32
}

// NewComplexf builds a Complexf from real and imaginary parts.
func NewComplexf(re, im float32) Complexf { return Complexf{re, im} }

// Add returns c+o.
func (c Complexf) Add(o Complexf) Complexf { return Complexf{c.Re + o.Re, c.Im + o.Im} }

// Sub returns c-o.
func (c Complexf) Sub(o Complexf) Complexf { return Complexf{c.Re - o.Re, c.Im - o.Im} }

// Mul returns the complex product c*o.
func (c Complexf) Mul(o Complexf) Complexf {
	return Complexf{c.Re*o.Re - c.Im*o.Im, c.Re*o.Im + c.Im*o.Re}
}

// Conj returns the complex conjugate.
func (c Complexf) Conj() Complexf { return Complexf{c.Re, -c.Im} }

// Abs returns the modulus |c|.
func (c Complexf) Abs() float64 { return math.Hypot(float64(c.Re), float64(c.Im)) }

// Arg returns the argument (phase angle) of c in radians.
func (c Complexf) Arg() float64 { return math.Atan2(float64(c.Im), float64(c.Re)) }

// String renders the number as re+imi.
func (c Complexf) String() string { return fmt.Sprintf("(%g%+gi)", c.Re, c.Im) }

// Range is a half-open integer interval [Start, End), mirroring cv::Range.
type Range struct {
	Start int
	End   int
}

// NewRange builds a Range from start and end.
func NewRange(start, end int) Range { return Range{start, end} }

// AllRange returns the special range covering an entire dimension, encoded as
// [MinInt, MaxInt) the way OpenCV's Range::all does.
func AllRange() Range { return Range{math.MinInt32, math.MaxInt32} }

// Size returns End-Start.
func (r Range) Size() int { return r.End - r.Start }

// Empty reports whether the range contains no elements.
func (r Range) Empty() bool { return r.Start >= r.End }

// Contains reports whether i lies in [Start, End).
func (r Range) Contains(i int) bool { return i >= r.Start && i < r.End }

// Shift returns the range translated by delta.
func (r Range) Shift(delta int) Range { return Range{r.Start + delta, r.End + delta} }

// Intersect returns the overlap of r and o; an empty range when disjoint.
func (r Range) Intersect(o Range) Range {
	s := r.Start
	if o.Start > s {
		s = o.Start
	}
	e := r.End
	if o.End < e {
		e = o.End
	}
	if e < s {
		e = s
	}
	return Range{s, e}
}

// Equals reports whether two ranges are identical.
func (r Range) Equals(o Range) bool { return r.Start == o.Start && r.End == o.End }

// String renders the range as [Start, End).
func (r Range) String() string { return fmt.Sprintf("[%d, %d)", r.Start, r.End) }

// RotatedRect is a rectangle rotated about its centre, mirroring
// cv::RotatedRect. Center is in fractional pixels, Size holds the side lengths
// and Angle is the clockwise rotation in degrees.
type RotatedRect struct {
	Center Point2f
	Size   Size2f
	Angle  float64
}

// NewRotatedRect builds a RotatedRect from its centre, size and angle.
func NewRotatedRect(center Point2f, size Size2f, angle float64) RotatedRect {
	return RotatedRect{center, size, angle}
}

// Points returns the four corners of the rotated rectangle in order
// (bottom-left, top-left, top-right, bottom-right in the rectangle's frame).
func (r RotatedRect) Points() [4]Point2f {
	rad := r.Angle * math.Pi / 180
	c, s := math.Cos(rad), math.Sin(rad)
	hw := float64(r.Size.Width) / 2
	hh := float64(r.Size.Height) / 2
	local := [4][2]float64{{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}}
	var out [4]Point2f
	for i, p := range local {
		x := float64(r.Center.X) + p[0]*c - p[1]*s
		y := float64(r.Center.Y) + p[0]*s + p[1]*c
		out[i] = Point2f{float32(x), float32(y)}
	}
	return out
}

// Area returns the rectangle's area (Width*Height).
func (r RotatedRect) Area() float64 { return float64(r.Size.Width) * float64(r.Size.Height) }

// BoundingRect returns the smallest upright integer rectangle enclosing the
// rotated rectangle.
func (r RotatedRect) BoundingRect() Rect2i {
	pts := r.Points()
	minX, minY := math.Inf(1), math.Inf(1)
	maxX, maxY := math.Inf(-1), math.Inf(-1)
	for _, p := range pts {
		minX = math.Min(minX, float64(p.X))
		minY = math.Min(minY, float64(p.Y))
		maxX = math.Max(maxX, float64(p.X))
		maxY = math.Max(maxY, float64(p.Y))
	}
	x0 := int(math.Floor(minX))
	y0 := int(math.Floor(minY))
	x1 := int(math.Ceil(maxX))
	y1 := int(math.Ceil(maxY))
	return Rect2i{x0, y0, x1 - x0, y1 - y0}
}

// TermCriteriaType is a bit flag selecting the stopping conditions of an
// iterative algorithm, mirroring cv::TermCriteria::Type.
type TermCriteriaType int

const (
	// TermCount stops after a maximum iteration count.
	TermCount TermCriteriaType = 1
	// TermEps stops when the achieved accuracy falls below Epsilon.
	TermEps TermCriteriaType = 2
)

// TermCriteria describes when an iterative algorithm should stop, mirroring
// cv::TermCriteria.
type TermCriteria struct {
	Type     TermCriteriaType
	MaxCount int
	Epsilon  float64
}

// NewTermCriteria builds a TermCriteria from its type flags, iteration cap and
// epsilon.
func NewTermCriteria(t TermCriteriaType, maxCount int, eps float64) TermCriteria {
	return TermCriteria{t, maxCount, eps}
}

// IsValid reports whether at least one stopping condition is enabled with a
// usable bound.
func (t TermCriteria) IsValid() bool {
	countOK := t.Type&TermCount != 0 && t.MaxCount > 0
	epsOK := t.Type&TermEps != 0 && t.Epsilon >= 0
	return countOK || epsOK
}

// KeyPoint is a salient image point produced by a feature detector, mirroring
// cv::KeyPoint.
type KeyPoint struct {
	Pt       Point2f
	Size     float64
	Angle    float64
	Response float64
	Octave   int
	ClassID  int
}

// NewKeyPoint builds a KeyPoint at (x, y) with the given diameter, angle,
// response, octave and class id. Angle is -1 when undefined.
func NewKeyPoint(x, y, size, angle, response float64, octave, classID int) KeyPoint {
	return KeyPoint{Point2f{float32(x), float32(y)}, size, angle, response, octave, classID}
}

// Overlap returns the intersection-over-union of the two keypoint regions,
// treated as discs of radius Size/2, matching cv::KeyPoint::overlap.
func (k KeyPoint) Overlap(o KeyPoint) float64 {
	a := float64(k.Size) / 2
	b := float64(o.Size) / 2
	dx := float64(k.Pt.X) - float64(o.Pt.X)
	dy := float64(k.Pt.Y) - float64(o.Pt.Y)
	d := math.Hypot(dx, dy)
	if d >= a+b {
		return 0
	}
	if d <= math.Abs(a-b) {
		rMin := math.Min(a, b)
		rMax := math.Max(a, b)
		return (rMin * rMin) / (rMax * rMax)
	}
	a2, b2, d2 := a*a, b*b, d*d
	alpha := math.Acos((d2 + a2 - b2) / (2 * d * a))
	beta := math.Acos((d2 + b2 - a2) / (2 * d * b))
	inter := a2*(alpha-0.5*math.Sin(2*alpha)) + b2*(beta-0.5*math.Sin(2*beta))
	union := math.Pi*a2 + math.Pi*b2 - inter
	if union == 0 {
		return 0
	}
	return inter / union
}

// DMatch links a query descriptor to a train descriptor with a distance,
// mirroring cv::DMatch.
type DMatch struct {
	QueryIdx int
	TrainIdx int
	ImgIdx   int
	Distance float64
}

// NewDMatch builds a DMatch from the query and train indices and a distance.
func NewDMatch(queryIdx, trainIdx, imgIdx int, distance float64) DMatch {
	return DMatch{queryIdx, trainIdx, imgIdx, distance}
}

// Less reports whether m sorts before o, ordering by ascending distance the way
// cv::DMatch::operator< does.
func (m DMatch) Less(o DMatch) bool { return m.Distance < o.Distance }
