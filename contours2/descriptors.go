package contours2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ContourCentroid returns the centroid of a contour computed from its moments.
// It returns (0, 0) for a degenerate contour of zero area.
func ContourCentroid(contour []cv.Point) cv.Point2f {
	return ContourMoments(contour).Centroid()
}

// AspectRatio returns the ratio of width to height of the upright bounding box
// of a contour. It panics on an empty contour.
func AspectRatio(contour []cv.Point) float64 {
	r := BoundingRect(contour)
	if r.Height == 0 {
		return 0
	}
	return float64(r.Width) / float64(r.Height)
}

// Extent returns the ratio of the contour area to the area of its upright
// bounding box, a measure of how fully the shape fills its box. It panics on an
// empty contour and returns 0 for a zero-area box.
func Extent(contour []cv.Point) float64 {
	r := BoundingRect(contour)
	boxArea := float64(r.Area())
	if boxArea == 0 {
		return 0
	}
	return ContourArea(contour) / boxArea
}

// Solidity returns the ratio of the contour area to its convex-hull area, a
// measure of convexity in the range (0, 1]. It panics on an empty contour and
// returns 0 when the hull has zero area.
func Solidity(contour []cv.Point) float64 {
	if len(contour) == 0 {
		panic("contours2: Solidity on empty contour")
	}
	hull := ConvexHull(contour, false)
	hullArea := ContourArea(hull)
	if hullArea == 0 {
		return 0
	}
	return ContourArea(contour) / hullArea
}

// EquivalentDiameter returns the diameter of the circle whose area equals the
// contour area, sqrt(4*area/pi).
func EquivalentDiameter(contour []cv.Point) float64 {
	return math.Sqrt(4 * ContourArea(contour) / math.Pi)
}

// Orientation returns the angle, in degrees, of the major axis of the ellipse
// with the same second-order central moments as the contour, in the range
// (-90, 90]. It returns 0 for a shape with no second-moment anisotropy.
func Orientation(contour []cv.Point) float64 {
	m := ContourMoments(contour)
	if m.M00 == 0 {
		return 0
	}
	return 0.5 * math.Atan2(2*m.Mu11, m.Mu20-m.Mu02) * 180 / math.Pi
}

// Eccentricity returns the eccentricity of the ellipse with the same
// second-order central moments as the contour, in the range [0, 1): 0 for a
// circle, approaching 1 for an elongated shape. It returns 0 for a degenerate
// contour.
func Eccentricity(contour []cv.Point) float64 {
	m := ContourMoments(contour)
	if m.M00 == 0 {
		return 0
	}
	a := m.Mu20 / m.M00
	b := m.Mu11 / m.M00
	c := m.Mu02 / m.M00
	common := math.Sqrt(math.Max(0, (a-c)*(a-c)+4*b*b))
	l1 := (a + c + common) / 2
	l2 := (a + c - common) / 2
	if l1 <= 0 {
		return 0
	}
	return math.Sqrt(math.Max(0, 1-l2/l1))
}
