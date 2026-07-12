package features2d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// DrawFlags controls how [DrawKeypointsFlags] renders keypoints, mirroring
// OpenCV's cv::DrawMatchesFlags. Combine values with bitwise OR.
type DrawFlags int

const (
	// DrawDefault draws each keypoint as a small fixed-radius circle at its
	// centre, ignoring Size and Angle.
	DrawDefault DrawFlags = 0
	// DrawRichKeypoints draws each keypoint as a circle sized by its Size and,
	// when an orientation is set, a radius line indicating its Angle. This
	// corresponds to OpenCV's DRAW_RICH_KEYPOINTS.
	DrawRichKeypoints DrawFlags = 1 << 2
)

// DrawKeypointsFlags returns a 3-channel copy of img with the keypoints drawn in
// the given colour (RGB). With [DrawRichKeypoints] the circle radius reflects
// each keypoint's Size and a line shows its Angle; otherwise a fixed small
// marker is drawn. Pass a zero Scalar to use the package default green. The
// input may be single- or three-channel and is not modified. This complements
// the simpler [DrawKeypoints], which always draws rich markers in green.
func DrawKeypointsFlags(img *cv.Mat, kps []KeyPoint, color cv.Scalar, flags DrawFlags) *cv.Mat {
	out := toColor(img)
	c := color
	if c == (cv.Scalar{}) {
		c = keypointColor
	}
	rich := flags&DrawRichKeypoints != 0
	for _, kp := range kps {
		cx, cy := kp.Pt.X, kp.Pt.Y
		center := cv.Point{X: cx, Y: cy}
		if !rich {
			cv.Circle(out, center, 3, c, 1)
			continue
		}
		radius := int(kp.Size / 2)
		if radius < 2 {
			radius = 2
		}
		cv.Circle(out, center, radius, c, 1)
		if kp.Angle >= 0 {
			rad := kp.Angle * math.Pi / 180
			end := cv.Point{
				X: cx + int(math.Round(float64(radius)*math.Cos(rad))),
				Y: cy + int(math.Round(float64(radius)*math.Sin(rad))),
			}
			cv.Line(out, center, end, c, 1)
		}
	}
	return out
}
