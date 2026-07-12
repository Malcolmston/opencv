package text_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/text"
)

// ExampleDetectRegionsMSER finds two dark square "characters" on a light
// background and reports how many stable regions were detected.
func ExampleDetectRegionsMSER() {
	img := cv.NewMat(20, 30, 1)
	img.SetTo(220) // light background
	// Two dark 5x5 blobs.
	for _, bx := range []int{4, 18} {
		for y := 6; y < 11; y++ {
			for x := bx; x < bx+5; x++ {
				img.Set(y, x, 0, 40)
			}
		}
	}

	boxes := text.DetectRegionsMSER(img, 5, 8, 100, 0.5)
	fmt.Printf("%d regions\n", len(boxes))
	// Output: 2 regions
}

// ExampleGroupTextRegions groups a row of character boxes into a single text
// line.
func ExampleGroupTextRegions() {
	regions := []cv.Rect{
		{X: 0, Y: 0, Width: 8, Height: 10},
		{X: 10, Y: 0, Width: 8, Height: 10},
		{X: 20, Y: 0, Width: 8, Height: 10},
	}
	lines := text.GroupTextRegions(regions)
	fmt.Printf("%d line, %d chars\n", len(lines), len(lines[0]))
	// Output: 1 line, 3 chars
}
