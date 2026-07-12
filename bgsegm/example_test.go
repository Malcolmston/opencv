package bgsegm_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/bgsegm"
)

// solid returns a rows×cols single-channel Mat filled with val.
func solid(rows, cols int, val uint8) *cv.Mat {
	m := cv.NewMat(rows, cols, 1)
	m.SetTo(val)
	return m
}

// countFG counts foreground pixels in a mask.
func countFG(mask *cv.Mat) int {
	n := 0
	for _, v := range mask.Data {
		if v == bgsegm.ForegroundValue {
			n++
		}
	}
	return n
}

// ExampleBackgroundSubtractorMOG2 warms the model on a static background, then
// applies a frame with a bright moving blob and reports the foreground area.
func ExampleBackgroundSubtractorMOG2() {
	sub := bgsegm.NewBackgroundSubtractorMOG2(10, 16, false)
	for i := 0; i < 30; i++ {
		sub.Apply(solid(8, 8, 40))
	}
	frame := solid(8, 8, 40)
	for y := 2; y < 5; y++ {
		for x := 3; x < 6; x++ {
			frame.Set(y, x, 0, 220) // a 3×3 bright blob
		}
	}
	mask := sub.Apply(frame)
	fmt.Println("foreground pixels:", countFG(mask))
	// Output: foreground pixels: 9
}

// ExampleBackgroundSubtractorKNN detects the same blob with the KNN model.
func ExampleBackgroundSubtractorKNN() {
	sub := bgsegm.NewBackgroundSubtractorKNN(20, 400, false)
	for i := 0; i < 15; i++ {
		sub.Apply(solid(8, 8, 40))
	}
	frame := solid(8, 8, 40)
	for y := 2; y < 5; y++ {
		for x := 3; x < 6; x++ {
			frame.Set(y, x, 0, 220)
		}
	}
	mask := sub.Apply(frame)
	fmt.Println("foreground pixels:", countFG(mask))
	// Output: foreground pixels: 9
}

// ExampleRunningAverage shows the simplest model and its converged background.
func ExampleRunningAverage() {
	sub := bgsegm.NewRunningAverage(10, 40)
	for i := 0; i < 10; i++ {
		sub.Apply(solid(4, 4, 50))
	}
	bg := sub.GetBackgroundImage()
	fmt.Println("background value:", bg.At(0, 0, 0))
	// Output: background value: 50
}

// ExampleCleanupMask removes an isolated speck while keeping a solid blob.
func ExampleCleanupMask() {
	mask := cv.NewMat(6, 6, 1)
	for y := 1; y < 5; y++ {
		for x := 1; x < 5; x++ {
			mask.Set(y, x, 0, bgsegm.ForegroundValue) // solid blob
		}
	}
	mask.Set(0, 5, 0, bgsegm.ForegroundValue) // isolated speck
	cleaned := bgsegm.CleanupMask(mask, 3)
	fmt.Println("speck after opening:", cleaned.At(0, 5, 0))
	fmt.Println("blob after opening:", cleaned.At(2, 2, 0))
	// Output:
	// speck after opening: 0
	// blob after opening: 255
}
