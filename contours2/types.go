package contours2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Contour is an ordered list of boundary points describing a connected shape,
// as produced by [FindContours]. Points are stored in traversal order around
// the border and are treated as a closed polygon by the analysis routines.
type Contour []cv.Point

// RetrievalMode selects which contours [FindContours] returns and how their
// hierarchy is built, mirroring OpenCV's RETR_* modes.
type RetrievalMode int

const (
	// RetrExternal returns only the outermost contours, discarding any holes
	// nested inside them. Every returned entry has Parent == -1.
	RetrExternal RetrievalMode = iota
	// RetrList returns every contour (outer borders and holes) as a flat list
	// with no parent/child relationships recorded.
	RetrList
	// RetrCComp returns every contour organised into a two-level hierarchy:
	// the outer boundaries of connected components sit at the top level and the
	// boundaries of their holes are their children.
	RetrCComp
	// RetrTree returns every contour and reconstructs the full nesting tree in
	// the hierarchy (parent, child and sibling links to arbitrary depth).
	RetrTree
)

// ChainApproxMethod selects how [FindContours] stores each border's points.
type ChainApproxMethod int

const (
	// ChainApproxNone stores every point along the border.
	ChainApproxNone ChainApproxMethod = iota
	// ChainApproxSimple collapses straight horizontal, vertical and diagonal
	// runs to their end points, so a rectangle keeps only its four corners.
	ChainApproxSimple
)

// HierarchyNode mirrors OpenCV's 4-element hierarchy entry. Each field is an
// index into the slice of contours returned by [FindContours], or -1 when
// absent. Next and Prev link siblings at the same nesting level; FirstChild is
// the first contour nested one level deeper; Parent is the enclosing contour.
type HierarchyNode struct {
	// Next is the index of the next sibling contour, or -1.
	Next int
	// Prev is the index of the previous sibling contour, or -1.
	Prev int
	// FirstChild is the index of the first nested (child) contour, or -1.
	FirstChild int
	// Parent is the index of the enclosing (parent) contour, or -1.
	Parent int
}

// Rect is an axis-aligned rectangle in integer pixel coordinates, matching
// OpenCV's cv::Rect. X and Y are the top-left corner; Width and Height are the
// extents.
type Rect struct {
	// X is the column of the top-left corner.
	X int
	// Y is the row of the top-left corner.
	Y int
	// Width is the horizontal extent in pixels.
	Width int
	// Height is the vertical extent in pixels.
	Height int
}

// Area returns the rectangle area (Width*Height).
func (r Rect) Area() int { return r.Width * r.Height }

// Contains reports whether the point lies inside the rectangle, treating the
// top and left edges as inside and the bottom and right edges as outside
// (the half-open convention used by OpenCV).
func (r Rect) Contains(p cv.Point) bool {
	return p.X >= r.X && p.X < r.X+r.Width && p.Y >= r.Y && p.Y < r.Y+r.Height
}

// TopLeft returns the top-left corner of the rectangle.
func (r Rect) TopLeft() cv.Point { return cv.Point{X: r.X, Y: r.Y} }

// BottomRight returns the bottom-right corner (exclusive) of the rectangle.
func (r Rect) BottomRight() cv.Point { return cv.Point{X: r.X + r.Width, Y: r.Y + r.Height} }

// RotatedRect is a rectangle that may be rotated about its centre, matching
// OpenCV's cv::RotatedRect. The centre is in fractional pixel coordinates,
// Width and Height are the side lengths, and Angle is the rotation in degrees
// measured clockwise (because the image y axis points downward).
type RotatedRect struct {
	// CenterX is the x coordinate of the centre.
	CenterX float64
	// CenterY is the y coordinate of the centre.
	CenterY float64
	// Width is the length of the first pair of sides.
	Width float64
	// Height is the length of the second pair of sides.
	Height float64
	// Angle is the rotation of the rectangle in degrees.
	Angle float64
}

// Area returns the area of the rotated rectangle (Width*Height).
func (r RotatedRect) Area() float64 { return r.Width * r.Height }

// Points returns the four corner points of the rotated rectangle in order,
// rounded to the nearest pixel. This mirrors OpenCV's boxPoints.
func (r RotatedRect) Points() [4]cv.Point {
	rad := r.Angle * math.Pi / 180
	c, s := math.Cos(rad), math.Sin(rad)
	hw, hh := r.Width/2, r.Height/2
	local := [4][2]float64{{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}}
	var out [4]cv.Point
	for i, p := range local {
		x := r.CenterX + p[0]*c - p[1]*s
		y := r.CenterY + p[0]*s + p[1]*c
		out[i] = cv.Point{X: int(math.Round(x)), Y: int(math.Round(y))}
	}
	return out
}

// PointsF returns the four corner points of the rotated rectangle in order as
// floating-point coordinates, without rounding.
func (r RotatedRect) PointsF() [4]cv.Point2f {
	rad := r.Angle * math.Pi / 180
	c, s := math.Cos(rad), math.Sin(rad)
	hw, hh := r.Width/2, r.Height/2
	local := [4][2]float64{{-hw, -hh}, {hw, -hh}, {hw, hh}, {-hw, hh}}
	var out [4]cv.Point2f
	for i, p := range local {
		out[i] = cv.Point2f{
			X: r.CenterX + p[0]*c - p[1]*s,
			Y: r.CenterY + p[0]*s + p[1]*c,
		}
	}
	return out
}

// BoundingRect returns the smallest axis-aligned integer rectangle that
// contains all four corners of the rotated rectangle.
func (r RotatedRect) BoundingRect() Rect {
	pts := r.PointsF()
	minX, minY := math.Inf(1), math.Inf(1)
	maxX, maxY := math.Inf(-1), math.Inf(-1)
	for _, p := range pts {
		minX = math.Min(minX, p.X)
		minY = math.Min(minY, p.Y)
		maxX = math.Max(maxX, p.X)
		maxY = math.Max(maxY, p.Y)
	}
	x0 := int(math.Floor(minX))
	y0 := int(math.Floor(minY))
	x1 := int(math.Ceil(maxX))
	y1 := int(math.Ceil(maxY))
	return Rect{X: x0, Y: y0, Width: x1 - x0, Height: y1 - y0}
}

// Circle is a circle in fractional pixel coordinates, as returned by
// [MinEnclosingCircle].
type Circle struct {
	// Center is the centre of the circle.
	Center cv.Point2f
	// Radius is the circle radius in pixels.
	Radius float64
}

// Contains reports whether the point lies within the circle (inclusive of the
// boundary, within a small tolerance).
func (c Circle) Contains(p cv.Point2f) bool {
	return math.Hypot(p.X-c.Center.X, p.Y-c.Center.Y) <= c.Radius+1e-9
}

// Area returns the area of the circle (pi*r^2).
func (c Circle) Area() float64 { return math.Pi * c.Radius * c.Radius }

// Ellipse describes an ellipse by its centre, full axis lengths and rotation,
// matching the parameterisation of OpenCV's fitEllipse output. Width and Height
// are the full lengths of the two axes (not the semi-axes), and Angle is the
// rotation of the Width axis in degrees.
type Ellipse struct {
	// Center is the ellipse centre.
	Center cv.Point2f
	// Width is the full length of the first axis.
	Width float64
	// Height is the full length of the second axis.
	Height float64
	// Angle is the rotation of the Width axis in degrees.
	Angle float64
}

// Area returns the area enclosed by the ellipse (pi*a*b for semi-axes a, b).
func (e Ellipse) Area() float64 { return math.Pi * (e.Width / 2) * (e.Height / 2) }

// Moments holds the spatial, central and normalised central moments of a shape
// up to third order, mirroring OpenCV's cv::Moments. Spatial moments are Mpq,
// central moments (translation invariant) are Mupq and normalised central
// moments (scale invariant) are Nupq.
type Moments struct {
	M00, M10, M01, M20, M11, M02, M30, M21, M12, M03 float64
	Mu20, Mu11, Mu02, Mu30, Mu21, Mu12, Mu03         float64
	Nu20, Nu11, Nu02, Nu30, Nu21, Nu12, Nu03         float64
}

// Centroid returns the centre of mass (M10/M00, M01/M00). It returns (0, 0) for
// a shape of zero total mass.
func (m Moments) Centroid() cv.Point2f {
	if m.M00 == 0 {
		return cv.Point2f{}
	}
	return cv.Point2f{X: m.M10 / m.M00, Y: m.M01 / m.M00}
}

// ConvexityDefect describes a concavity of a contour relative to its convex
// hull, matching OpenCV's convexityDefects output. StartIndex and EndIndex are
// indices into the contour of the two hull points bounding the defect,
// FarthestPointIndex is the index of the contour point deepest inside the
// defect, and Depth is that point's distance from the hull edge in pixels.
type ConvexityDefect struct {
	// StartIndex is the contour index of the defect's start hull point.
	StartIndex int
	// EndIndex is the contour index of the defect's end hull point.
	EndIndex int
	// FarthestPointIndex is the contour index of the deepest contour point.
	FarthestPointIndex int
	// Depth is the distance from the farthest point to the hull edge.
	Depth float64
}

// ShapeMatchMethod selects the metric used by [MatchShapes], mirroring OpenCV's
// CONTOURS_MATCH_I1/I2/I3.
type ShapeMatchMethod int

const (
	// ContoursMatchI1 is the sum of absolute differences of reciprocal
	// log-scaled Hu moments.
	ContoursMatchI1 ShapeMatchMethod = iota + 1
	// ContoursMatchI2 is the sum of absolute differences of log-scaled Hu
	// moments.
	ContoursMatchI2
	// ContoursMatchI3 is the maximum relative difference of log-scaled Hu
	// moments.
	ContoursMatchI3
)
