package moments2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// AxisLengths returns the lengths of the major and minor axes of the ellipse
// that has the same second central moments as the region described by m. The
// values are full axis lengths (not semi-axes). It returns (0, 0) for a shape of
// zero mass.
func AxisLengths(m Moments) (major, minor float64) {
	if m.M00 == 0 {
		return 0, 0
	}
	a := m.Mu20 / m.M00
	b := m.Mu11 / m.M00
	c := m.Mu02 / m.M00
	common := math.Sqrt((a-c)*(a-c) + 4*b*b)
	l1 := (a + c + common) / 2
	l2 := (a + c - common) / 2
	if l2 < 0 {
		l2 = 0
	}
	return 4 * math.Sqrt(l1), 4 * math.Sqrt(l2)
}

// Eccentricity returns the eccentricity of the equivalent ellipse of m, a value
// in [0, 1) where 0 is a perfect circle and values approaching 1 are highly
// elongated. It returns 0 for a shape of zero mass.
func Eccentricity(m Moments) float64 {
	if m.M00 == 0 {
		return 0
	}
	a := m.Mu20 / m.M00
	b := m.Mu11 / m.M00
	c := m.Mu02 / m.M00
	common := math.Sqrt((a-c)*(a-c) + 4*b*b)
	l1 := (a + c + common) / 2
	l2 := (a + c - common) / 2
	if l1 <= 0 {
		return 0
	}
	if l2 < 0 {
		l2 = 0
	}
	return math.Sqrt(1 - l2/l1)
}

// Elongation returns the ratio of the major to the minor axis length of the
// equivalent ellipse of m, always at least 1. A circle yields 1; an infinitely
// thin shape yields +Inf. It returns 0 for a shape of zero mass.
func Elongation(m Moments) float64 {
	major, minor := AxisLengths(m)
	if major == 0 {
		return 0
	}
	if minor == 0 {
		return math.Inf(1)
	}
	return major / minor
}

// Orientation returns the angle, in radians in the range (-pi/2, pi/2], of the
// major axis of the equivalent ellipse of m relative to the positive x axis. It
// returns 0 for a shape of zero mass or a rotationally symmetric one.
func Orientation(m Moments) float64 {
	if m.M00 == 0 {
		return 0
	}
	a := m.Mu20 / m.M00
	b := m.Mu11 / m.M00
	c := m.Mu02 / m.M00
	return 0.5 * math.Atan2(2*b, a-c)
}

// PolygonArea returns the absolute area enclosed by a polygon using the shoelace
// formula. Vertices are given in order and the polygon is treated as closed.
func PolygonArea(pts []cv.Point) float64 {
	n := len(pts)
	if n < 3 {
		return 0
	}
	var s float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		s += float64(pts[i].X)*float64(pts[j].Y) - float64(pts[j].X)*float64(pts[i].Y)
	}
	return math.Abs(s) / 2
}

// PolygonPerimeter returns the closed perimeter length of a polygon, the sum of
// the Euclidean edge lengths including the closing edge.
func PolygonPerimeter(pts []cv.Point) float64 {
	n := len(pts)
	if n < 2 {
		return 0
	}
	var p float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		dx := float64(pts[j].X - pts[i].X)
		dy := float64(pts[j].Y - pts[i].Y)
		p += math.Hypot(dx, dy)
	}
	return p
}

// PolygonCentroid returns the area centroid of a polygon computed from its first
// moments. It returns the origin for a degenerate polygon of zero area.
func PolygonCentroid(pts []cv.Point) cv.Point2f {
	n := len(pts)
	if n < 3 {
		return cv.Point2f{}
	}
	var a, cx, cy float64
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		xi, yi := float64(pts[i].X), float64(pts[i].Y)
		xj, yj := float64(pts[j].X), float64(pts[j].Y)
		cross := xi*yj - xj*yi
		a += cross
		cx += (xi + xj) * cross
		cy += (yi + yj) * cross
	}
	if a == 0 {
		return cv.Point2f{}
	}
	return cv.Point2f{X: cx / (3 * a), Y: cy / (3 * a)}
}

// Solidity returns the ratio of a contour's area to the area of its convex hull,
// a value in (0, 1] where 1 means the shape is convex. It returns 0 when the
// hull area is zero.
func Solidity(contour []cv.Point) float64 {
	area := cv.ContourArea(cv.Contour(contour))
	hull := cv.ConvexHull(contour)
	hullArea := cv.ContourArea(cv.Contour(hull))
	if hullArea == 0 {
		return 0
	}
	return area / hullArea
}

// Extent returns the ratio of a contour's area to the area of its upright
// bounding rectangle, a value in (0, 1]. It returns 0 for an empty contour.
func Extent(contour []cv.Point) float64 {
	if len(contour) == 0 {
		return 0
	}
	area := cv.ContourArea(cv.Contour(contour))
	r := cv.BoundingRect(contour)
	box := float64(r.Width) * float64(r.Height)
	if box == 0 {
		return 0
	}
	return area / box
}

// EquivalentDiameter returns the diameter of the circle whose area equals the
// given region area, sqrt(4*area/pi).
func EquivalentDiameter(area float64) float64 {
	if area <= 0 {
		return 0
	}
	return math.Sqrt(4 * area / math.Pi)
}

// Compactness returns the isoperimetric compactness perimeter^2/area of a
// region. The minimum value, achieved by a circle, is 4*pi. It returns 0 when
// area is zero.
func Compactness(area, perimeter float64) float64 {
	if area == 0 {
		return 0
	}
	return perimeter * perimeter / area
}

// Circularity returns the normalized circularity 4*pi*area/perimeter^2 of a
// region, a value in (0, 1] where 1 is a perfect circle. It returns 0 when the
// perimeter is zero.
func Circularity(area, perimeter float64) float64 {
	if perimeter == 0 {
		return 0
	}
	return 4 * math.Pi * area / (perimeter * perimeter)
}

// FormFactor is an alias of [Circularity]: 4*pi*area/perimeter^2.
func FormFactor(area, perimeter float64) float64 { return Circularity(area, perimeter) }

// Roundness returns 4*area/(pi*maxDiameter^2), a compactness measure that, unlike
// circularity, is insensitive to boundary roughness because it uses the longest
// diameter rather than the perimeter. It returns 0 when maxDiameter is zero.
func Roundness(area, maxDiameter float64) float64 {
	if maxDiameter == 0 {
		return 0
	}
	return 4 * area / (math.Pi * maxDiameter * maxDiameter)
}

// AspectRatio returns the ratio of the width to the height of a contour's
// upright bounding rectangle. It returns 0 for an empty contour.
func AspectRatio(contour []cv.Point) float64 {
	if len(contour) == 0 {
		return 0
	}
	r := cv.BoundingRect(contour)
	if r.Height == 0 {
		return 0
	}
	return float64(r.Width) / float64(r.Height)
}

// Rectangularity returns the ratio of a contour's area to the area of its
// minimum-area rotated bounding rectangle, a value in (0, 1] measuring how well
// the shape fills its tightest rectangle. It returns 0 when that rectangle has
// zero area.
func Rectangularity(contour []cv.Point) float64 {
	if len(contour) == 0 {
		return 0
	}
	area := cv.ContourArea(cv.Contour(contour))
	rr := cv.MinAreaRect(contour)
	box := rr.Width * rr.Height
	if box == 0 {
		return 0
	}
	return area / box
}

// Convexity returns the ratio of the convex-hull perimeter to the contour
// perimeter, a value in (0, 1] where 1 means the contour is convex. Boundary
// concavities lengthen the contour perimeter and lower the ratio. It returns 0
// when the contour perimeter is zero.
func Convexity(contour []cv.Point) float64 {
	perim := PolygonPerimeter(contour)
	if perim == 0 {
		return 0
	}
	hull := cv.ConvexHull(contour)
	return PolygonPerimeter(hull) / perim
}

// MaxDiameter returns the greatest distance between any two vertices of a
// contour (its Feret diameter). It returns 0 for fewer than two points.
func MaxDiameter(contour []cv.Point) float64 {
	n := len(contour)
	var best float64
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			dx := float64(contour[i].X - contour[j].X)
			dy := float64(contour[i].Y - contour[j].Y)
			if d := math.Hypot(dx, dy); d > best {
				best = d
			}
		}
	}
	return best
}

// EllipseParams describes an ellipse by its centre, full axis lengths and the
// orientation of its major axis in radians.
type EllipseParams struct {
	// Center is the ellipse centre in image coordinates.
	Center cv.Point2f
	// Major is the full length of the major axis.
	Major float64
	// Minor is the full length of the minor axis.
	Minor float64
	// Angle is the orientation of the major axis in radians.
	Angle float64
}

// FitEllipse returns the ellipse that has the same centroid and second central
// moments as the region described by m. This is the classic moment-based
// ellipse fit; it does not perform an algebraic least-squares fit to boundary
// points. It returns the zero EllipseParams for a shape of zero mass.
func FitEllipse(m Moments) EllipseParams {
	if m.M00 == 0 {
		return EllipseParams{}
	}
	major, minor := AxisLengths(m)
	return EllipseParams{
		Center: m.Centroid(),
		Major:  major,
		Minor:  minor,
		Angle:  Orientation(m),
	}
}
