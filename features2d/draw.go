package features2d

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// keypointColor is the default colour (green, RGB) used to draw keypoints.
var keypointColor = cv.NewScalar(0, 255, 0)

// matchColor is the default colour (yellow, RGB) used to draw match lines.
var matchColor = cv.NewScalar(255, 255, 0)

// DrawKeypoints returns a 3-channel copy of img with each keypoint drawn as a
// circle whose radius reflects its Size and, when an orientation is set
// (Angle >= 0), a radial line indicating the angle. The input may be single- or
// three-channel; the original is not modified.
func DrawKeypoints(img *cv.Mat, kps []KeyPoint) *cv.Mat {
	out := toColor(img)
	drawKeypointsOn(out, kps, cv.Point{X: 0, Y: 0})
	return out
}

// drawKeypointsOn draws keypoints onto dst, offsetting every position by off so
// the same routine serves both DrawKeypoints and the two panels of DrawMatches.
func drawKeypointsOn(dst *cv.Mat, kps []KeyPoint, off cv.Point) {
	for _, kp := range kps {
		cx := kp.Pt.X + off.X
		cy := kp.Pt.Y + off.Y
		radius := int(kp.Size / 2)
		if radius < 2 {
			radius = 2
		}
		center := cv.Point{X: cx, Y: cy}
		cv.Circle(dst, center, radius, keypointColor, 1)
		if kp.Angle >= 0 {
			rad := kp.Angle * math.Pi / 180
			end := cv.Point{
				X: cx + int(math.Round(float64(radius)*math.Cos(rad))),
				Y: cy + int(math.Round(float64(radius)*math.Sin(rad))),
			}
			cv.Line(dst, center, end, keypointColor, 1)
		}
	}
}

// DrawMatches renders img1 and img2 side by side (img1 on the left) in a new
// 3-channel Mat, draws every keypoint, and connects each matched pair with a
// line from kp1[m.QueryIdx] to kp2[m.TrainIdx]. Both images may be single- or
// three-channel; the originals are not modified. Matches whose indices fall
// outside the keypoint slices are skipped.
func DrawMatches(img1 *cv.Mat, kp1 []KeyPoint, img2 *cv.Mat, kp2 []KeyPoint, matches []DMatch) *cv.Mat {
	left := toColor(img1)
	right := toColor(img2)
	h := left.Rows
	if right.Rows > h {
		h = right.Rows
	}
	w := left.Cols + right.Cols
	canvas := cv.NewMat(h, w, 3)
	left.CopyTo(canvas, 0, 0)
	right.CopyTo(canvas, 0, left.Cols)

	off := cv.Point{X: left.Cols, Y: 0}
	drawKeypointsOn(canvas, kp1, cv.Point{X: 0, Y: 0})
	drawKeypointsOn(canvas, kp2, off)

	for _, mt := range matches {
		if mt.QueryIdx < 0 || mt.QueryIdx >= len(kp1) || mt.TrainIdx < 0 || mt.TrainIdx >= len(kp2) {
			continue
		}
		p1 := kp1[mt.QueryIdx].Pt
		p2 := kp2[mt.TrainIdx].Pt
		cv.Line(canvas, p1, cv.Point{X: p2.X + off.X, Y: p2.Y + off.Y}, matchColor, 1)
	}
	return canvas
}
