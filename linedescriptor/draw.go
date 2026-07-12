package linedescriptor

import cv "github.com/malcolmston/opencv"

// DrawKeylines renders lines onto a copy of img and returns the result, echoing
// cv::line_descriptor::drawKeylines. The output is always a 3-channel image so
// that coloured segments are visible even over a grayscale source: a 1-channel
// img is promoted to RGB first, a 3-channel img is copied as-is. Each segment
// is drawn as a straight line from its start point to its end point in the
// given colour with the given thickness (a thickness below 1 is treated as 1).
func DrawKeylines(img *cv.Mat, lines []KeyLine, color cv.Scalar, thickness int) *cv.Mat {
	var canvas *cv.Mat
	switch img.Channels {
	case 1:
		canvas = cv.CvtColor(img, cv.ColorGray2RGB)
	case 3:
		canvas = img.Clone()
	default:
		panic("linedescriptor: DrawKeylines expects a 1- or 3-channel image")
	}
	if thickness < 1 {
		thickness = 1
	}
	for _, kl := range lines {
		cv.Line(canvas, kl.StartPoint, kl.EndPoint, color, thickness)
	}
	return canvas
}
