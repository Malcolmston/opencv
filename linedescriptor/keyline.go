package linedescriptor

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// KeyLine is a detected straight line segment, mirroring OpenCV's
// cv::line_descriptor::KeyLine. A segment is oriented: it runs from StartPoint
// to EndPoint, and [KeyLine.Angle] encodes that direction. Because a line has
// no intrinsic head or tail, callers comparing two segments' orientations
// should treat their angles as equal modulo π.
type KeyLine struct {
	// StartPoint is the first endpoint of the segment (X is the column, Y the
	// row).
	StartPoint cv.Point
	// EndPoint is the second endpoint of the segment.
	EndPoint cv.Point
	// Angle is the orientation of the segment in radians, computed as
	// math.Atan2(EndPoint.Y-StartPoint.Y, EndPoint.X-StartPoint.X); it lies in
	// the range (-π, π].
	Angle float64
	// Length is the Euclidean distance between the two endpoints in pixels.
	Length float64
	// Response ranks segments; larger is stronger. The detector sets it to the
	// segment length, so sorting by descending Response yields the longest,
	// most prominent segments first.
	Response float64
	// Octave is the scale-pyramid layer the segment was detected in. This port
	// works at a single scale, so it is always 0.
	Octave int
}

// newKeyLine builds a KeyLine from two floating-point endpoints, filling in the
// derived Angle, Length and Response fields. The endpoints are rounded to the
// nearest pixel for storage.
func newKeyLine(x1, y1, x2, y2 float64) KeyLine {
	dx := x2 - x1
	dy := y2 - y1
	length := math.Hypot(dx, dy)
	return KeyLine{
		StartPoint: cv.Point{X: int(math.Round(x1)), Y: int(math.Round(y1))},
		EndPoint:   cv.Point{X: int(math.Round(x2)), Y: int(math.Round(y2))},
		Angle:      math.Atan2(dy, dx),
		Length:     length,
		Response:   length,
		Octave:     0,
	}
}

// toGray returns a single-channel Mat view of img: a 1-channel input is cloned,
// a 3-channel input is converted to grayscale with the BT.601 luma weights via
// [cv.CvtColor]. It panics for other channel counts.
func toGray(img *cv.Mat) *cv.Mat {
	switch img.Channels {
	case 1:
		return img.Clone()
	case 3:
		return cv.CvtColor(img, cv.ColorRGB2Gray)
	default:
		panic("linedescriptor: expected a 1- or 3-channel image")
	}
}

// gradients computes the signed Sobel derivatives of a single-channel image and
// returns the per-pixel gx, gy (row-major) plus the image dimensions.
func gradients(gray *cv.Mat) (gx, gy []float64, rows, cols int) {
	gx = cv.SobelFloat(gray, 1, 0, 3)[0]
	gy = cv.SobelFloat(gray, 0, 1, 3)[0]
	return gx, gy, gray.Rows, gray.Cols
}

// angleDiff returns the absolute difference between two angles wrapped into
// [0, π]. Angles that differ by π (opposite directions) return π.
func angleDiff(a, b float64) float64 {
	d := math.Mod(math.Abs(a-b), 2*math.Pi)
	if d > math.Pi {
		d = 2*math.Pi - d
	}
	return d
}
