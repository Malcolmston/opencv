package geom_cv

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Segment is a finite line segment between two endpoints A and B.
type Segment struct {
	// A is the first endpoint.
	A cv.Point2f
	// B is the second endpoint.
	B cv.Point2f
}

// Length returns the Euclidean length of the segment.
func (s Segment) Length() float64 {
	return Distance(s.A, s.B)
}

// Direction returns the unit vector pointing from A to B, or the zero vector if
// the endpoints coincide.
func (s Segment) Direction() cv.Point2f {
	return Normalize(Sub(s.B, s.A))
}

// Midpoint returns the point halfway between the segment endpoints.
func (s Segment) Midpoint() cv.Point2f {
	return Midpoint(s.A, s.B)
}

// Line is an infinite line represented by a point P on the line and a direction
// vector Dir. Dir need not be normalized; a zero Dir denotes a degenerate line.
type Line struct {
	// P is an arbitrary point lying on the line.
	P cv.Point2f
	// Dir is the direction of the line; it need not be unit length.
	Dir cv.Point2f
}

// LineThrough returns the infinite line passing through the two points a and b,
// with its direction pointing from a toward b.
func LineThrough(a, b cv.Point2f) Line {
	return Line{P: a, Dir: Sub(b, a)}
}

// PointAt returns the point P + t*Dir on the line.
func (l Line) PointAt(t float64) cv.Point2f {
	return Add(l.P, Scale(l.Dir, t))
}

// Circle is a circle with a center and a radius.
type Circle struct {
	// Center is the circle center.
	Center cv.Point2f
	// Radius is the circle radius; a negative radius denotes an empty circle.
	Radius float64
}

// Contains reports whether the point p lies inside or on the circle, within the
// package tolerance.
func (c Circle) Contains(p cv.Point2f) bool {
	return Distance(c.Center, p) <= c.Radius+geom_cvEps
}

// Area returns the area π·r² enclosed by the circle.
func (c Circle) Area() float64 {
	return math.Pi * c.Radius * c.Radius
}

// Triangle is a triangle given by its three corners.
type Triangle struct {
	// A, B and C are the triangle corners.
	A, B, C cv.Point2f
}

// Area returns the unsigned area of the triangle.
func (t Triangle) Area() float64 {
	return math.Abs(Cross(Sub(t.B, t.A), Sub(t.C, t.A))) / 2
}

// SignedArea returns the signed area of the triangle: positive when A→B→C winds
// counter-clockwise in a right-handed frame and negative otherwise.
func (t Triangle) SignedArea() float64 {
	return Cross(Sub(t.B, t.A), Sub(t.C, t.A)) / 2
}

// Centroid returns the arithmetic mean of the three corners, which is the
// triangle's center of mass.
func (t Triangle) Centroid() cv.Point2f {
	return cv.Point2f{
		X: (t.A.X + t.B.X + t.C.X) / 3,
		Y: (t.A.Y + t.B.Y + t.C.Y) / 3,
	}
}

// Contains reports whether the point p lies inside or on the boundary of the
// triangle. Degenerate (zero-area) triangles contain only points on their
// supporting segment.
func (t Triangle) Contains(p cv.Point2f) bool {
	d1 := Cross(Sub(t.B, t.A), Sub(p, t.A))
	d2 := Cross(Sub(t.C, t.B), Sub(p, t.B))
	d3 := Cross(Sub(t.A, t.C), Sub(p, t.C))
	hasNeg := d1 < -geom_cvEps || d2 < -geom_cvEps || d3 < -geom_cvEps
	hasPos := d1 > geom_cvEps || d2 > geom_cvEps || d3 > geom_cvEps
	return !(hasNeg && hasPos)
}

// BoundingBox is an axis-aligned rectangle described by its minimum and maximum
// corners. It is the floating-point analogue of the parent library's Rect.
type BoundingBox struct {
	// Min holds the smallest X and Y of the box.
	Min cv.Point2f
	// Max holds the largest X and Y of the box.
	Max cv.Point2f
}

// Width returns the horizontal extent Max.X-Min.X of the box.
func (b BoundingBox) Width() float64 {
	return b.Max.X - b.Min.X
}

// Height returns the vertical extent Max.Y-Min.Y of the box.
func (b BoundingBox) Height() float64 {
	return b.Max.Y - b.Min.Y
}

// Area returns Width times Height.
func (b BoundingBox) Area() float64 {
	return b.Width() * b.Height()
}

// Center returns the midpoint of the box.
func (b BoundingBox) Center() cv.Point2f {
	return Midpoint(b.Min, b.Max)
}

// Contains reports whether the point p lies inside or on the box, within the
// package tolerance.
func (b BoundingBox) Contains(p cv.Point2f) bool {
	return p.X >= b.Min.X-geom_cvEps && p.X <= b.Max.X+geom_cvEps &&
		p.Y >= b.Min.Y-geom_cvEps && p.Y <= b.Max.Y+geom_cvEps
}

// Corners returns the four corners of the box in counter-clockwise order
// starting from Min: (Min.X,Min.Y), (Max.X,Min.Y), (Max.X,Max.Y),
// (Min.X,Max.Y).
func (b BoundingBox) Corners() []cv.Point2f {
	return []cv.Point2f{
		{X: b.Min.X, Y: b.Min.Y},
		{X: b.Max.X, Y: b.Min.Y},
		{X: b.Max.X, Y: b.Max.Y},
		{X: b.Min.X, Y: b.Max.Y},
	}
}

// Rect converts the box to the parent library's integer
// [github.com/malcolmston/opencv.Rect], rounding the corners outward so the
// rectangle fully covers the box.
func (b BoundingBox) Rect() cv.Rect {
	minX := int(math.Floor(b.Min.X))
	minY := int(math.Floor(b.Min.Y))
	maxX := int(math.Ceil(b.Max.X))
	maxY := int(math.Ceil(b.Max.Y))
	return cv.Rect{X: minX, Y: minY, Width: maxX - minX, Height: maxY - minY}
}
