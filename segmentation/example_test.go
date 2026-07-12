package segmentation_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/segmentation"
)

func ExampleFloodFill() {
	// A 5x5 gray canvas with a 3x3 block of value 100 in the middle.
	m := cv.NewMat(5, 5, 1)
	m.SetTo(0)
	for y := 1; y < 4; y++ {
		for x := 1; x < 4; x++ {
			m.Set(y, x, 0, 100)
		}
	}
	count, rect := segmentation.FloodFill(m, cv.Point{X: 2, Y: 2}, cv.NewScalar(255),
		cv.NewScalar(0), cv.NewScalar(0), 4)
	fmt.Printf("filled %d pixels, rect %+v\n", count, rect)
	// Output: filled 9 pixels, rect {X:1 Y:1 Width:3 Height:3}
}

func ExampleWatershed() {
	// Two basins separated by a bright ridge at column 4.
	m := cv.NewMat(8, 8, 1)
	m.SetTo(30)
	for y := 0; y < 8; y++ {
		m.Set(y, 4, 0, 220)
	}
	markers := cv.NewMat(8, 8, 1)
	markers.Set(4, 1, 0, 1)
	markers.Set(4, 6, 0, 2)
	labels := segmentation.Watershed(m, markers)
	fmt.Println(labels.At(4, 1, 0), labels.At(4, 6, 0))
	// Output: 1 2
}

func ExampleMeanShiftFiltering() {
	// A single-pixel spike inside a flat field is smoothed toward the field.
	m := cv.NewMat(5, 5, 3)
	for i := range m.Data {
		m.Data[i] = 100
	}
	m.SetPixel(2, 2, []uint8{160, 160, 160})
	out := segmentation.MeanShiftFiltering(m, 2, 120)
	fmt.Println(out.At(2, 2, 0) < 160)
	// Output: true
}
