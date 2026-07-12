package features2d

import cv "github.com/malcolmston/opencv"

// buildBlobs renders a light background with a fixed set of dark filled discs of
// varying radius, for the blob detector. It returns the image and the disc
// centres.
func buildBlobs(size int) (*cv.Mat, []cv.Point) {
	m := cv.NewMat(size, size, 1)
	m.SetTo(255)
	centers := []cv.Point{
		{X: 30, Y: 30}, {X: 90, Y: 40}, {X: 50, Y: 95}, {X: 110, Y: 110},
	}
	radii := []int{8, 12, 6, 10}
	for i, c := range centers {
		cv.Circle(m, c, radii[i], cv.NewScalar(0), cv.Filled)
	}
	return m, centers
}
