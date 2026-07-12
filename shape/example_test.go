package shape_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/shape"
)

func ExampleMinEnclosingCircle() {
	// The four corners of a 2×2 square.
	pts := []cv.Point{{X: 0, Y: 0}, {X: 2, Y: 0}, {X: 2, Y: 2}, {X: 0, Y: 2}}
	cx, cy, r := shape.MinEnclosingCircle(pts)
	fmt.Printf("centre=(%.1f, %.1f) radius=%.3f\n", cx, cy, r)
	// Output: centre=(1.0, 1.0) radius=1.414
}

func ExampleFitLine() {
	// Points lying on the line y = 2x.
	pts := []cv.Point{{X: -2, Y: -4}, {X: -1, Y: -2}, {X: 0, Y: 0}, {X: 1, Y: 2}, {X: 2, Y: 4}}
	vx, vy, x0, y0 := shape.FitLine(pts)
	fmt.Printf("dir=(%.3f, %.3f) through=(%.1f, %.1f)\n", vx, vy, x0, y0)
	// Output: dir=(0.447, 0.894) through=(0.0, 0.0)
}

func ExampleMatchShapes() {
	tri := []cv.Point{{X: 0, Y: 0}, {X: 40, Y: 0}, {X: 20, Y: 30}}
	// The same triangle scaled 2× and shifted: a congruent shape.
	big := []cv.Point{{X: 100, Y: 100}, {X: 180, Y: 100}, {X: 140, Y: 160}}
	score := shape.MatchShapes(tri, big, shape.ContoursMatchI1)
	fmt.Printf("congruent match score < 0.001: %v\n", score < 0.001)
	// Output: congruent match score < 0.001: true
}

func ExampleConvexityDefects() {
	// A rectangle with a V-notch cut into its top edge.
	contour := []cv.Point{
		{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 5, Y: 3}, {X: 6, Y: 0},
		{X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10},
	}
	hull := shape.ConvexHullIndices(contour)
	defects := shape.ConvexityDefects(contour, hull)
	for _, d := range defects {
		fmt.Printf("notch at %v depth %.0f\n", d.Far, d.Depth)
	}
	// Output: notch at {5 3} depth 3
}
