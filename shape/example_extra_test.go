package shape_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/shape"
)

func ExampleIsContourConvex() {
	square := []cv.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 4}, {X: 0, Y: 4}}
	notched := []cv.Point{
		{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 2, Y: 1}, {X: 4, Y: 4}, {X: 0, Y: 4},
	}
	fmt.Println(shape.IsContourConvex(square), shape.IsContourConvex(notched))
	// Output: true false
}

func ExamplePointPolygonTest() {
	square := []cv.Point{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10}}
	fmt.Printf("inside=%.0f edge=%.0f outside=%.0f\n",
		shape.PointPolygonTest(square, shape.Point2D{X: 5, Y: 5}, false),
		shape.PointPolygonTest(square, shape.Point2D{X: 0, Y: 5}, false),
		shape.PointPolygonTest(square, shape.Point2D{X: 20, Y: 5}, false),
	)
	// Output: inside=1 edge=0 outside=-1
}

func ExampleEMDL1() {
	// Moving one unit of mass across two bins costs two bin-widths of work.
	fmt.Printf("%.1f\n", shape.EMDL1([]float64{1, 0, 0}, []float64{0, 0, 1}))
	// Output: 2.0
}

func ExampleSolveAssignment() {
	cost := [][]float64{
		{4, 1, 3},
		{2, 0, 5},
		{3, 2, 2},
	}
	assign, total := shape.SolveAssignment(cost)
	fmt.Printf("assignment=%v total=%.0f\n", assign, total)
	// Output: assignment=[1 0 2] total=5
}

func ExampleThinPlateSplineShapeTransformer() {
	src := []shape.Point2D{{X: 0, Y: 0}, {X: 2, Y: 0}, {X: 0, Y: 2}, {X: 2, Y: 2}}
	dst := []shape.Point2D{{X: 0, Y: 0}, {X: 2, Y: 0}, {X: 0, Y: 2}, {X: 3, Y: 3}}
	tps := shape.NewThinPlateSplineShapeTransformer(0)
	tps.EstimateTransformation(src, dst)
	// The spline interpolates every control point exactly.
	got := tps.ApplyTransformation([]shape.Point2D{{X: 2, Y: 2}})
	fmt.Printf("corner -> (%.1f, %.1f)\n", got[0].X, got[0].Y)
	// Output: corner -> (3.0, 3.0)
}

func ExampleHausdorffDistanceExtractor() {
	a := []cv.Point{{X: 0, Y: 0}, {X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10}}
	b := []cv.Point{{X: 0, Y: 0}, {X: 12, Y: 0}, {X: 12, Y: 10}, {X: 0, Y: 10}}
	ext := shape.NewHausdorffDistanceExtractor(shape.HausdorffL2, 1.0)
	fmt.Printf("%.0f\n", ext.ComputeDistance(a, b))
	// Output: 2
}
