package tracking_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/tracking"
)

// brightPatch renders a small grayscale frame with a textured bright square
// centred at (cx, cy) (a dark mark in one corner breaks the symmetry so the
// patch has the intensity variation the correlation trackers need), used to
// illustrate the trackers.
func brightPatch(cx, cy int) *cv.Mat {
	m := cv.NewMat(40, 40, 1)
	set := func(x, y int, v uint8) {
		if x >= 0 && x < 40 && y >= 0 && y < 40 {
			m.Data[y*40+x] = v
		}
	}
	for y := cy - 5; y <= cy+5; y++ {
		for x := cx - 5; x <= cx+5; x++ {
			set(x, y, 220)
		}
	}
	for y := cy - 4; y <= cy-2; y++ {
		for x := cx - 4; x <= cx-2; x++ {
			set(x, y, 60)
		}
	}
	return m
}

// ExampleTrackerTemplate initialises the template tracker on one frame and
// follows the patch into the next.
func ExampleTrackerTemplate() {
	tr := tracking.NewTrackerTemplate()
	tr.Init(brightPatch(15, 15), cv.Rect{X: 10, Y: 10, Width: 11, Height: 11})
	box, ok := tr.Update(brightPatch(18, 16)) // patch moved by (3, 1)
	fmt.Printf("box=%+v ok=%v\n", box, ok)
	// Output: box={X:13 Y:11 Width:11 Height:11} ok=true
}

// ExampleMeanShift walks a window to the mode of a probability image.
func ExampleMeanShift() {
	prob := cv.NewMat(40, 40, 1)
	// A bright 6×6 blob centred at (25, 20).
	for y := 17; y <= 23; y++ {
		for x := 22; x <= 28; x++ {
			prob.Data[y*40+x] = 255
		}
	}
	win := tracking.MeanShift(prob, cv.Rect{X: 18, Y: 13, Width: 9, Height: 9}, 20)
	fmt.Printf("center=(%d,%d)\n", win.X+win.Width/2, win.Y+win.Height/2)
	// Output: center=(25,20)
}
