package core

import (
	"fmt"
	"math"
)

// Point2i is a 2D point with integer coordinates, mirroring cv::Point2i
// (a.k.a. cv::Point). X is the column and Y is the row.
type Point2i struct {
	X int
	Y int
}

// Pt2i builds a Point2i from x and y.
func Pt2i(x, y int) Point2i { return Point2i{x, y} }

// Add returns the vector sum p+o.
func (p Point2i) Add(o Point2i) Point2i { return Point2i{p.X + o.X, p.Y + o.Y} }

// Sub returns the vector difference p-o.
func (p Point2i) Sub(o Point2i) Point2i { return Point2i{p.X - o.X, p.Y - o.Y} }

// Mul returns the point scaled by s.
func (p Point2i) Mul(s int) Point2i { return Point2i{p.X * s, p.Y * s} }

// Dot returns the integer dot product p·o.
func (p Point2i) Dot(o Point2i) int { return p.X*o.X + p.Y*o.Y }

// Ddot returns the dot product accumulated in float64.
func (p Point2i) Ddot(o Point2i) float64 {
	return float64(p.X)*float64(o.X) + float64(p.Y)*float64(o.Y)
}

// Cross returns the 2D cross product (z-component) p×o.
func (p Point2i) Cross(o Point2i) float64 {
	return float64(p.X)*float64(o.Y) - float64(p.Y)*float64(o.X)
}

// Norm returns the Euclidean length of the position vector.
func (p Point2i) Norm() float64 { return math.Hypot(float64(p.X), float64(p.Y)) }

// Inside reports whether the point lies within rectangle r (edges inclusive on
// the top-left, exclusive on the bottom-right, matching OpenCV).
func (p Point2i) Inside(r Rect2i) bool {
	return p.X >= r.X && p.X < r.X+r.Width && p.Y >= r.Y && p.Y < r.Y+r.Height
}

// ToPoint2f converts to a float32 point.
func (p Point2i) ToPoint2f() Point2f { return Point2f{float32(p.X), float32(p.Y)} }

// ToPoint2d converts to a float64 point.
func (p Point2i) ToPoint2d() Point2d { return Point2d{float64(p.X), float64(p.Y)} }

// Equals reports whether p and o are the same point.
func (p Point2i) Equals(o Point2i) bool { return p.X == o.X && p.Y == o.Y }

// String renders the point as (x, y).
func (p Point2i) String() string { return fmt.Sprintf("(%d, %d)", p.X, p.Y) }

// Point2f is a 2D point with float32 coordinates, mirroring cv::Point2f.
type Point2f struct {
	X float32
	Y float32
}

// Pt2f builds a Point2f from x and y.
func Pt2f(x, y float32) Point2f { return Point2f{x, y} }

// Add returns the vector sum p+o.
func (p Point2f) Add(o Point2f) Point2f { return Point2f{p.X + o.X, p.Y + o.Y} }

// Sub returns the vector difference p-o.
func (p Point2f) Sub(o Point2f) Point2f { return Point2f{p.X - o.X, p.Y - o.Y} }

// Mul returns the point scaled by s.
func (p Point2f) Mul(s float32) Point2f { return Point2f{p.X * s, p.Y * s} }

// Dot returns the dot product p·o.
func (p Point2f) Dot(o Point2f) float32 { return p.X*o.X + p.Y*o.Y }

// Ddot returns the dot product accumulated in float64.
func (p Point2f) Ddot(o Point2f) float64 {
	return float64(p.X)*float64(o.X) + float64(p.Y)*float64(o.Y)
}

// Cross returns the 2D cross product (z-component) p×o.
func (p Point2f) Cross(o Point2f) float64 {
	return float64(p.X)*float64(o.Y) - float64(p.Y)*float64(o.X)
}

// Norm returns the Euclidean length of the position vector.
func (p Point2f) Norm() float64 { return math.Hypot(float64(p.X), float64(p.Y)) }

// Inside reports whether the point lies within rectangle r.
func (p Point2f) Inside(r Rect2f) bool {
	return p.X >= r.X && p.X < r.X+r.Width && p.Y >= r.Y && p.Y < r.Y+r.Height
}

// ToPoint2d converts to a float64 point.
func (p Point2f) ToPoint2d() Point2d { return Point2d{float64(p.X), float64(p.Y)} }

// Round returns the nearest integer point.
func (p Point2f) Round() Point2i {
	return Point2i{int(math.Round(float64(p.X))), int(math.Round(float64(p.Y)))}
}

// Equals reports whether p and o are the same point.
func (p Point2f) Equals(o Point2f) bool { return p.X == o.X && p.Y == o.Y }

// String renders the point as (x, y).
func (p Point2f) String() string { return fmt.Sprintf("(%g, %g)", p.X, p.Y) }

// Point2d is a 2D point with float64 coordinates, mirroring cv::Point2d.
type Point2d struct {
	X float64
	Y float64
}

// Pt2d builds a Point2d from x and y.
func Pt2d(x, y float64) Point2d { return Point2d{x, y} }

// Add returns the vector sum p+o.
func (p Point2d) Add(o Point2d) Point2d { return Point2d{p.X + o.X, p.Y + o.Y} }

// Sub returns the vector difference p-o.
func (p Point2d) Sub(o Point2d) Point2d { return Point2d{p.X - o.X, p.Y - o.Y} }

// Mul returns the point scaled by s.
func (p Point2d) Mul(s float64) Point2d { return Point2d{p.X * s, p.Y * s} }

// Dot returns the dot product p·o.
func (p Point2d) Dot(o Point2d) float64 { return p.X*o.X + p.Y*o.Y }

// Ddot is an alias for Dot kept for symmetry with the integer types.
func (p Point2d) Ddot(o Point2d) float64 { return p.Dot(o) }

// Cross returns the 2D cross product (z-component) p×o.
func (p Point2d) Cross(o Point2d) float64 { return p.X*o.Y - p.Y*o.X }

// Norm returns the Euclidean length of the position vector.
func (p Point2d) Norm() float64 { return math.Hypot(p.X, p.Y) }

// Inside reports whether the point lies within rectangle r.
func (p Point2d) Inside(r Rect2d) bool {
	return p.X >= r.X && p.X < r.X+r.Width && p.Y >= r.Y && p.Y < r.Y+r.Height
}

// Round returns the nearest integer point.
func (p Point2d) Round() Point2i { return Point2i{int(math.Round(p.X)), int(math.Round(p.Y))} }

// Equals reports whether p and o are the same point.
func (p Point2d) Equals(o Point2d) bool { return p.X == o.X && p.Y == o.Y }

// String renders the point as (x, y).
func (p Point2d) String() string { return fmt.Sprintf("(%g, %g)", p.X, p.Y) }

// Point3i is a 3D point with integer coordinates, mirroring cv::Point3i.
type Point3i struct {
	X int
	Y int
	Z int
}

// Pt3i builds a Point3i from x, y and z.
func Pt3i(x, y, z int) Point3i { return Point3i{x, y, z} }

// Add returns the vector sum p+o.
func (p Point3i) Add(o Point3i) Point3i { return Point3i{p.X + o.X, p.Y + o.Y, p.Z + o.Z} }

// Sub returns the vector difference p-o.
func (p Point3i) Sub(o Point3i) Point3i { return Point3i{p.X - o.X, p.Y - o.Y, p.Z - o.Z} }

// Mul returns the point scaled by s.
func (p Point3i) Mul(s int) Point3i { return Point3i{p.X * s, p.Y * s, p.Z * s} }

// Dot returns the integer dot product p·o.
func (p Point3i) Dot(o Point3i) int { return p.X*o.X + p.Y*o.Y + p.Z*o.Z }

// Cross returns the 3D cross product p×o.
func (p Point3i) Cross(o Point3i) Point3i {
	return Point3i{p.Y*o.Z - p.Z*o.Y, p.Z*o.X - p.X*o.Z, p.X*o.Y - p.Y*o.X}
}

// Norm returns the Euclidean length of the position vector.
func (p Point3i) Norm() float64 {
	return math.Sqrt(float64(p.X*p.X + p.Y*p.Y + p.Z*p.Z))
}

// Equals reports whether p and o are the same point.
func (p Point3i) Equals(o Point3i) bool { return p.X == o.X && p.Y == o.Y && p.Z == o.Z }

// String renders the point as (x, y, z).
func (p Point3i) String() string { return fmt.Sprintf("(%d, %d, %d)", p.X, p.Y, p.Z) }

// Point3f is a 3D point with float32 coordinates, mirroring cv::Point3f.
type Point3f struct {
	X float32
	Y float32
	Z float32
}

// Pt3f builds a Point3f from x, y and z.
func Pt3f(x, y, z float32) Point3f { return Point3f{x, y, z} }

// Add returns the vector sum p+o.
func (p Point3f) Add(o Point3f) Point3f { return Point3f{p.X + o.X, p.Y + o.Y, p.Z + o.Z} }

// Sub returns the vector difference p-o.
func (p Point3f) Sub(o Point3f) Point3f { return Point3f{p.X - o.X, p.Y - o.Y, p.Z - o.Z} }

// Mul returns the point scaled by s.
func (p Point3f) Mul(s float32) Point3f { return Point3f{p.X * s, p.Y * s, p.Z * s} }

// Dot returns the dot product p·o.
func (p Point3f) Dot(o Point3f) float32 { return p.X*o.X + p.Y*o.Y + p.Z*o.Z }

// Cross returns the 3D cross product p×o.
func (p Point3f) Cross(o Point3f) Point3f {
	return Point3f{p.Y*o.Z - p.Z*o.Y, p.Z*o.X - p.X*o.Z, p.X*o.Y - p.Y*o.X}
}

// Norm returns the Euclidean length of the position vector.
func (p Point3f) Norm() float64 {
	return math.Sqrt(float64(p.X*p.X + p.Y*p.Y + p.Z*p.Z))
}

// Normalize returns the unit vector along p; a zero vector is returned as-is.
func (p Point3f) Normalize() Point3f {
	n := p.Norm()
	if n == 0 {
		return p
	}
	return Point3f{float32(float64(p.X) / n), float32(float64(p.Y) / n), float32(float64(p.Z) / n)}
}

// Equals reports whether p and o are the same point.
func (p Point3f) Equals(o Point3f) bool { return p.X == o.X && p.Y == o.Y && p.Z == o.Z }

// String renders the point as (x, y, z).
func (p Point3f) String() string { return fmt.Sprintf("(%g, %g, %g)", p.X, p.Y, p.Z) }

// Point3d is a 3D point with float64 coordinates, mirroring cv::Point3d.
type Point3d struct {
	X float64
	Y float64
	Z float64
}

// Pt3d builds a Point3d from x, y and z.
func Pt3d(x, y, z float64) Point3d { return Point3d{x, y, z} }

// Add returns the vector sum p+o.
func (p Point3d) Add(o Point3d) Point3d { return Point3d{p.X + o.X, p.Y + o.Y, p.Z + o.Z} }

// Sub returns the vector difference p-o.
func (p Point3d) Sub(o Point3d) Point3d { return Point3d{p.X - o.X, p.Y - o.Y, p.Z - o.Z} }

// Mul returns the point scaled by s.
func (p Point3d) Mul(s float64) Point3d { return Point3d{p.X * s, p.Y * s, p.Z * s} }

// Dot returns the dot product p·o.
func (p Point3d) Dot(o Point3d) float64 { return p.X*o.X + p.Y*o.Y + p.Z*o.Z }

// Cross returns the 3D cross product p×o.
func (p Point3d) Cross(o Point3d) Point3d {
	return Point3d{p.Y*o.Z - p.Z*o.Y, p.Z*o.X - p.X*o.Z, p.X*o.Y - p.Y*o.X}
}

// Norm returns the Euclidean length of the position vector.
func (p Point3d) Norm() float64 { return math.Sqrt(p.X*p.X + p.Y*p.Y + p.Z*p.Z) }

// Normalize returns the unit vector along p; a zero vector is returned as-is.
func (p Point3d) Normalize() Point3d {
	n := p.Norm()
	if n == 0 {
		return p
	}
	return Point3d{p.X / n, p.Y / n, p.Z / n}
}

// Equals reports whether p and o are the same point.
func (p Point3d) Equals(o Point3d) bool { return p.X == o.X && p.Y == o.Y && p.Z == o.Z }

// String renders the point as (x, y, z).
func (p Point3d) String() string { return fmt.Sprintf("(%g, %g, %g)", p.X, p.Y, p.Z) }
