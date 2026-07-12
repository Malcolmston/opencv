package features2d

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

func ExampleKeyPointsFilter_retainBest() {
	var f KeyPointsFilter
	kps := []KeyPoint{
		{Pt: cv.Point{X: 1, Y: 1}, Response: 0.2},
		{Pt: cv.Point{X: 2, Y: 2}, Response: 0.9},
		{Pt: cv.Point{X: 3, Y: 3}, Response: 0.5},
	}
	for _, kp := range f.RetainBest(kps, 2) {
		fmt.Printf("(%d,%d) resp=%.1f\n", kp.Pt.X, kp.Pt.Y, kp.Response)
	}
	// Output:
	// (2,2) resp=0.9
	// (3,3) resp=0.5
}

func ExampleKeyPointsFilter_runByImageBorder() {
	var f KeyPointsFilter
	kps := []KeyPoint{
		{Pt: cv.Point{X: 2, Y: 40}}, // within 5px of the left edge
		{Pt: cv.Point{X: 35, Y: 35}},
	}
	out := f.RunByImageBorder(kps, 70, 70, 5)
	fmt.Println(len(out), out[0].Pt)
	// Output: 1 {35 35}
}

func ExampleComputeRecallPrecisionCurve() {
	matches := [][]DMatch{
		{{Distance: 1}, {Distance: 5}},
		{{Distance: 2}, {Distance: 6}},
	}
	correct := [][]bool{
		{true, false},
		{true, false},
	}
	curve := ComputeRecallPrecisionCurve(matches, correct)
	p := curve[1] // after the two correct (closest) detections
	fmt.Printf("recall=%.2f precision=%.2f\n", p.Recall, p.Precision)
	// Output: recall=1.00 precision=1.00
}

func ExampleBOWKMeansTrainer() {
	// Two obvious clusters of 1-D descriptors.
	trainer := NewBOWKMeansTrainer(2, 0, 0)
	trainer.Add(NewFloatDescriptors([][]float64{{0}, {0.1}, {10}, {10.1}}))
	vocab := trainer.Cluster()
	fmt.Println(vocab.Len())
	// Output: 2
}

func ExampleFlannBasedMatcher() {
	train := NewFloatDescriptors([][]float64{{0, 0}, {9, 9}})
	query := NewFloatDescriptors([][]float64{{9.1, 9.1}, {0.1, 0.1}})
	// Exhaustive search (Checks < 0) is exact.
	for _, m := range NewFlannBasedMatcher(1, -1).Match(query, train) {
		fmt.Printf("query %d -> train %d\n", m.QueryIdx, m.TrainIdx)
	}
	// Output:
	// query 0 -> train 1
	// query 1 -> train 0
}
