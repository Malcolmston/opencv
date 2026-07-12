package linedescriptor_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/linedescriptor"
)

// ExampleLSDDetector_Detect detects the segments of a single drawn line. A
// bright bar has two gradient edges, so the detector recovers two collinear
// segments of the same orientation.
func ExampleLSDDetector_Detect() {
	img := cv.NewMat(90, 90, 1)
	cv.Line(img, cv.Point{X: 15, Y: 45}, cv.Point{X: 75, Y: 45}, cv.NewScalar(255), 3)

	lines := linedescriptor.NewLSDDetector().Detect(img)
	fmt.Println("segments:", len(lines))
	fmt.Printf("orientation: %.2f rad\n", lines[0].Angle)
	// Output:
	// segments: 2
	// orientation: 0.00 rad
}

// ExampleBinaryDescriptor_Compute describes a detected segment with a
// fixed-length binary code.
func ExampleBinaryDescriptor_Compute() {
	img := cv.NewMat(120, 120, 1)
	cv.Line(img, cv.Point{X: 30, Y: 60}, cv.Point{X: 90, Y: 60}, cv.NewScalar(255), 3)

	bd := linedescriptor.NewBinaryDescriptor()
	lines := linedescriptor.NewLSDDetector().Detect(img)
	kept, codes := bd.Compute(img, lines)
	fmt.Println("lines:", len(kept))
	fmt.Println("code bytes:", len(codes[0]))
	// Output:
	// lines: 2
	// code bytes: 4
}

// ExampleBinaryDescriptorMatcher_Match pairs each query code with its closest
// train code by Hamming distance.
func ExampleBinaryDescriptorMatcher_Match() {
	query := [][]byte{
		{0b0000_0000},
		{0b1111_1111},
	}
	train := [][]byte{
		{0b1111_1110}, // closest to the all-ones query
		{0b0000_0001}, // closest to the all-zeros query
	}
	m := linedescriptor.NewBinaryDescriptorMatcher()
	for _, mt := range m.Match(query, train) {
		fmt.Printf("query %d -> train %d (dist %d)\n", mt.QueryIdx, mt.TrainIdx, mt.Distance)
	}
	// Output:
	// query 0 -> train 1 (dist 1)
	// query 1 -> train 0 (dist 1)
}

// ExampleDrawKeylines renders detected segments onto a colour canvas.
func ExampleDrawKeylines() {
	img := cv.NewMat(60, 60, 1)
	det := linedescriptor.NewLSDDetector()
	cv.Line(img, cv.Point{X: 10, Y: 30}, cv.Point{X: 50, Y: 30}, cv.NewScalar(255), 3)
	lines := det.Detect(img)

	out := linedescriptor.DrawKeylines(img, lines, cv.NewScalar(255, 0, 0), 1)
	fmt.Println(out.Rows, out.Cols, out.Channels)
	// Output:
	// 60 60 3
}
