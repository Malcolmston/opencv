package bgsegm_test

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/bgsegm"
)

// blob8 returns an 8×8 background frame with a 3×3 bright blob composited on it.
func blob8() *cv.Mat {
	frame := solid(8, 8, 40)
	for y := 2; y < 5; y++ {
		for x := 3; x < 6; x++ {
			frame.Set(y, x, 0, 220)
		}
	}
	return frame
}

// ExampleBackgroundSubtractorMOG detects a bright blob with the original MOG
// mixture model.
func ExampleBackgroundSubtractorMOG() {
	sub := bgsegm.NewBackgroundSubtractorMOG(10, 5, false)
	for i := 0; i < 30; i++ {
		sub.Apply(solid(8, 8, 40))
	}
	mask := sub.Apply(blob8())
	fmt.Println("foreground pixels:", countFG(mask))
	// Output: foreground pixels: 9
}

// ExampleBackgroundSubtractorCNT detects the blob with the fast counter model.
func ExampleBackgroundSubtractorCNT() {
	sub := bgsegm.NewBackgroundSubtractorCNT(5, true, 900)
	for i := 0; i < 10; i++ {
		sub.Apply(solid(8, 8, 40))
	}
	mask := sub.Apply(blob8())
	fmt.Println("foreground pixels:", countFG(mask))
	// Output: foreground pixels: 9
}

// ExampleBackgroundSubtractorLSBP detects the blob with the local-SVD binary
// pattern model.
func ExampleBackgroundSubtractorLSBP() {
	sub := bgsegm.NewBackgroundSubtractorLSBP(8, 30, false)
	for i := 0; i < 10; i++ {
		sub.Apply(solid(8, 8, 40))
	}
	mask := sub.Apply(blob8())
	fmt.Println("foreground pixels:", countFG(mask))
	// Output: foreground pixels: 9
}

// ExampleBackgroundSubtractorGSOC detects the blob with the sample-consensus
// GSOC model.
func ExampleBackgroundSubtractorGSOC() {
	sub := bgsegm.NewBackgroundSubtractorGSOC(20, 30, false)
	for i := 0; i < 10; i++ {
		sub.Apply(solid(8, 8, 40))
	}
	mask := sub.Apply(blob8())
	fmt.Println("foreground pixels:", countFG(mask))
	// Output: foreground pixels: 9
}

// ExampleCloseMask fills a single-pixel hole inside a solid blob.
func ExampleCloseMask() {
	mask := cv.NewMat(6, 6, 1)
	for y := 1; y < 5; y++ {
		for x := 1; x < 5; x++ {
			mask.Set(y, x, 0, bgsegm.ForegroundValue)
		}
	}
	mask.Set(3, 3, 0, 0) // punch a hole
	closed := bgsegm.CloseMask(mask, 3)
	fmt.Println("hole after closing:", closed.At(3, 3, 0))
	// Output: hole after closing: 255
}

// ExampleSyntheticSequenceGenerator manufactures a frame with a known moving
// object and its ground-truth mask.
func ExampleSyntheticSequenceGenerator() {
	bg := solid(16, 16, 40)
	obj := solid(2, 2, 200)
	gen := bgsegm.NewSyntheticSequenceGenerator(bg, obj, 3, 20, 1, 1, 0, 1)
	_, gt := gen.Next()
	n := 0
	for _, v := range gt.Data {
		if v == bgsegm.ForegroundValue {
			n++
		}
	}
	fmt.Println("object pixels:", n)
	// Output: object pixels: 4
}
