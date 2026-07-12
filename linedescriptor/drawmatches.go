package linedescriptor

import cv "github.com/malcolmston/opencv"

// promoteRGB returns a 3-channel copy of img so segments can be drawn in colour
// over any source. A 1-channel image is expanded to RGB, a 3-channel image is
// cloned, and any other channel count panics.
func promoteRGB(img *cv.Mat) *cv.Mat {
	switch img.Channels {
	case 1:
		return cv.CvtColor(img, cv.ColorGray2RGB)
	case 3:
		return img.Clone()
	default:
		panic("linedescriptor: expected a 1- or 3-channel image")
	}
}

// DrawLineMatches renders two images side by side, draws the detected segments
// of each and connects matched pairs with a coloured line, mirroring
// cv::line_descriptor::drawLineMatches. img1's segments (lines1) are drawn on
// the left half of the canvas and img2's segments (lines2) on the right half,
// offset by img1's width. For every [DMatch] the query segment, the train
// segment and a line joining their midpoints are drawn in matchColor; segments
// that take part in no match are drawn in singleColor.
//
// The output is always a 3-channel image whose height is the taller of the two
// inputs and whose width is their combined widths. A thickness below 1 is
// treated as 1. Indices in matches that fall outside their respective line
// slice are skipped, so partial or filtered match lists are safe to pass.
func DrawLineMatches(img1 *cv.Mat, lines1 []KeyLine, img2 *cv.Mat, lines2 []KeyLine, matches []DMatch, matchColor, singleColor cv.Scalar, thickness int) *cv.Mat {
	if thickness < 1 {
		thickness = 1
	}
	left := promoteRGB(img1)
	right := promoteRGB(img2)

	h := left.Rows
	if right.Rows > h {
		h = right.Rows
	}
	w := left.Cols + right.Cols
	canvas := cv.NewMat(h, w, 3)
	left.CopyTo(canvas, 0, 0)
	right.CopyTo(canvas, 0, left.Cols)

	offset := left.Cols

	// Track which segments are matched so the rest can be drawn as singletons.
	matched1 := make([]bool, len(lines1))
	matched2 := make([]bool, len(lines2))
	for _, mt := range matches {
		if mt.QueryIdx < 0 || mt.QueryIdx >= len(lines1) {
			continue
		}
		if mt.TrainIdx < 0 || mt.TrainIdx >= len(lines2) {
			continue
		}
		matched1[mt.QueryIdx] = true
		matched2[mt.TrainIdx] = true
	}

	// Draw unmatched (single) segments first so matched ones render on top.
	for i, kl := range lines1 {
		if !matched1[i] {
			cv.Line(canvas, kl.StartPoint, kl.EndPoint, singleColor, thickness)
		}
	}
	for i, kl := range lines2 {
		if !matched2[i] {
			cv.Line(canvas, shiftX(kl.StartPoint, offset), shiftX(kl.EndPoint, offset), singleColor, thickness)
		}
	}

	// Draw matched pairs and their connecting lines.
	for _, mt := range matches {
		if mt.QueryIdx < 0 || mt.QueryIdx >= len(lines1) {
			continue
		}
		if mt.TrainIdx < 0 || mt.TrainIdx >= len(lines2) {
			continue
		}
		a := lines1[mt.QueryIdx]
		b := lines2[mt.TrainIdx]
		cv.Line(canvas, a.StartPoint, a.EndPoint, matchColor, thickness)
		cv.Line(canvas, shiftX(b.StartPoint, offset), shiftX(b.EndPoint, offset), matchColor, thickness)
		cv.Line(canvas, midpoint(a), shiftX(midpoint(b), offset), matchColor, thickness)
	}
	return canvas
}

// shiftX returns p translated right by dx columns.
func shiftX(p cv.Point, dx int) cv.Point {
	return cv.Point{X: p.X + dx, Y: p.Y}
}

// midpoint returns the integer midpoint of a segment's endpoints.
func midpoint(kl KeyLine) cv.Point {
	return cv.Point{
		X: (kl.StartPoint.X + kl.EndPoint.X) / 2,
		Y: (kl.StartPoint.Y + kl.EndPoint.Y) / 2,
	}
}
