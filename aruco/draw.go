package aruco

import (
	"strconv"

	cv "github.com/malcolmston/opencv"
)

// outlineColor is the colour (green, RGB) used for marker outlines.
var outlineColor = cv.NewScalar(0, 255, 0)

// cornerColor is the colour (red, RGB) marking the top-left corner.
var cornerColor = cv.NewScalar(255, 0, 0)

// textColor is the colour (magenta, RGB) used for identifier labels.
var textColor = cv.NewScalar(255, 0, 255)

// DrawDetectedMarkers returns a three-channel copy of img with every detection
// overlaid: each marker's four edges are drawn as a green quadrilateral, its
// top-left corner (corners[i][0]) is dotted in red, and its identifier is
// printed near the marker centre. corners and ids are the parallel slices
// returned by [DetectMarkers]; detections whose id index is missing are labelled
// by position only. The input is not modified.
func DrawDetectedMarkers(img *cv.Mat, corners [][4]cv.Point, ids []int) *cv.Mat {
	out := toColor(img)
	for i, quad := range corners {
		poly := []cv.Point{quad[0], quad[1], quad[2], quad[3]}
		cv.Polylines(out, [][]cv.Point{poly}, true, outlineColor, 2)
		cv.Circle(out, quad[0], 4, cornerColor, cv.Filled)

		if i < len(ids) {
			cx, cy := quadCenter(quad)
			org := cv.Point{X: int(cx) - 6, Y: int(cy) + 3}
			cv.PutText(out, strconv.Itoa(ids[i]), org, 1, textColor)
		}
	}
	return out
}

// toColor returns a three-channel copy of img, promoting grayscale to RGB.
func toColor(img *cv.Mat) *cv.Mat {
	if img.Channels == 3 {
		return img.Clone()
	}
	return cv.CvtColor(img, cv.ColorGray2RGB)
}
