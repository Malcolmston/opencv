package linedescriptor_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/linedescriptor"
)

// ExampleLSDDetector_DetectPyramid detects a line across a three-octave scale
// pyramid, tagging each recovered segment with the octave it was found in.
func ExampleLSDDetector_DetectPyramid() {
	img := cv.NewMat(200, 200, 1)
	cv.Line(img, cv.Point{X: 20, Y: 100}, cv.Point{X: 180, Y: 100}, cv.NewScalar(255), 5)

	ex := linedescriptor.NewLSDDetector().DetectPyramid(img, 3, 2.0)
	seen := map[int]bool{}
	for _, l := range ex {
		seen[l.Octave] = true
	}
	fmt.Println("octave 0:", seen[0])
	fmt.Println("octave 1:", seen[1])
	fmt.Println("octave 2:", seen[2])
	// Output:
	// octave 0: true
	// octave 1: true
	// octave 2: true
}

// ExampleEDLinesDetector_Detect recovers the two edges of a drawn bar with the
// edge-drawing detector.
func ExampleEDLinesDetector_Detect() {
	img := cv.NewMat(130, 130, 1)
	cv.Line(img, cv.Point{X: 20, Y: 60}, cv.Point{X: 110, Y: 60}, cv.NewScalar(255), 3)

	lines := linedescriptor.NewEDLinesDetector().Detect(img)
	fmt.Println("found segments:", len(lines) > 0)
	fmt.Printf("orientation: %.2f rad\n", lines[0].Angle)
	// Output:
	// found segments: true
	// orientation: 0.00 rad
}

// ExampleLSHIndex demonstrates multi-index LSH retrieval: an identical
// descriptor is found at Hamming distance 0.
func ExampleLSHIndex() {
	train := [][]byte{
		{0xFF, 0x00, 0xAA, 0x55},
		{0x00, 0xFF, 0x55, 0xAA},
	}
	idx := linedescriptor.NewLSHIndex(4, 8)
	idx.Add(train)

	knn := idx.KnnMatch([][]byte{{0xFF, 0x00, 0xAA, 0x55}}, 1)
	fmt.Println("train index:", knn[0][0].TrainIdx)
	fmt.Println("distance:", knn[0][0].Distance)
	// Output:
	// train index: 0
	// distance: 0
}

// ExampleBinaryDescriptorMatcher_RadiusMatch returns all train codes within a
// Hamming radius of the query.
func ExampleBinaryDescriptorMatcher_RadiusMatch() {
	m := linedescriptor.NewBinaryDescriptorMatcher()
	query := [][]byte{{0b0000_0000}}
	train := [][]byte{
		{0b0000_0001}, // dist 1
		{0b0000_1111}, // dist 4
		{0b0000_0000}, // dist 0
	}
	for _, dm := range m.RadiusMatch(query, train, 2)[0] {
		fmt.Printf("train %d dist %d\n", dm.TrainIdx, dm.Distance)
	}
	// Output:
	// train 2 dist 0
	// train 0 dist 1
}

// ExampleMatchLineSegments verifies matches with geometry, rejecting an
// appearance-only match whose orientation is inconsistent.
func ExampleMatchLineSegments() {
	// A horizontal query.
	q := []linedescriptor.KeyLine{{
		StartPoint: cv.Point{X: 0, Y: 50}, EndPoint: cv.Point{X: 60, Y: 50}, Length: 60,
	}}
	qc := [][]byte{{0b0000_0000}}
	// Train: a vertical segment (identical code) and a horizontal one (near code).
	tr := []linedescriptor.KeyLine{
		{StartPoint: cv.Point{X: 30, Y: 0}, EndPoint: cv.Point{X: 30, Y: 60}, Angle: 1.5708, Length: 60},
		{StartPoint: cv.Point{X: 0, Y: 52}, EndPoint: cv.Point{X: 60, Y: 52}, Length: 60},
	}
	tc := [][]byte{{0b0000_0000}, {0b0000_0001}}

	matches := linedescriptor.MatchLineSegments(q, qc, tr, tc, linedescriptor.DefaultGeometricMatchParams())
	fmt.Println("matched train:", matches[0].TrainIdx)
	// Output:
	// matched train: 1
}

// ExampleDrawLineMatches renders two images side by side and links matched
// segments.
func ExampleDrawLineMatches() {
	img1 := cv.NewMat(50, 60, 1)
	img2 := cv.NewMat(50, 60, 1)
	lines1 := []linedescriptor.KeyLine{{StartPoint: cv.Point{X: 5, Y: 25}, EndPoint: cv.Point{X: 55, Y: 25}}}
	lines2 := []linedescriptor.KeyLine{{StartPoint: cv.Point{X: 5, Y: 25}, EndPoint: cv.Point{X: 55, Y: 25}}}
	matches := []linedescriptor.DMatch{{QueryIdx: 0, TrainIdx: 0}}

	out := linedescriptor.DrawLineMatches(img1, lines1, img2, lines2, matches,
		cv.NewScalar(0, 255, 0), cv.NewScalar(255, 0, 0), 1)
	fmt.Println(out.Rows, out.Cols, out.Channels)
	// Output:
	// 50 120 3
}
