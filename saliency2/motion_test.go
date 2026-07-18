package saliency2_test

import (
	"testing"

	cv "github.com/malcolmston/opencv"
	"github.com/malcolmston/opencv/saliency2"
)

// TestMotionByDifferenceFirstFrameZero checks the first frame yields a zero map.
func TestMotionByDifferenceFirstFrameZero(t *testing.T) {
	det := saliency2.NewMotionSaliencyByDifference()
	det.Sigma = 0
	frame := squareImage(32, 4, 4, 8, 20, 200)
	sal := det.ComputeSaliencyMap(frame)
	for _, v := range sal.Data {
		if v != 0 {
			t.Fatal("first-frame motion map should be all zeros")
		}
	}
}

// TestMotionByDifferenceDetectsMovement checks a moved block produces high
// saliency where it appeared and vanished, and none in the static background.
func TestMotionByDifferenceDetectsMovement(t *testing.T) {
	det := saliency2.NewMotionSaliencyByDifference()
	det.Sigma = 0
	f1 := squareImage(32, 4, 12, 8, 20, 220)  // block on the left
	f2 := squareImage(32, 20, 12, 8, 20, 220) // block moved right
	det.ComputeSaliencyMap(f1)
	sal := det.ComputeSaliencyMap(f2)

	moved := sal.At(16, 22)  // inside the new block position
	vacated := sal.At(16, 6) // where the block used to be
	still := sal.At(2, 2)    // untouched background corner
	if !(moved > still) || !(vacated > still) {
		t.Fatalf("motion at moved %.1f / vacated %.1f should exceed still %.1f", moved, vacated, still)
	}
}

// TestMotionByDifferenceReset checks Reset clears temporal state.
func TestMotionByDifferenceReset(t *testing.T) {
	det := saliency2.NewMotionSaliencyByDifference()
	det.Sigma = 0
	det.ComputeSaliencyMap(squareImage(16, 2, 2, 4, 0, 200))
	det.Reset()
	sal := det.ComputeSaliencyMap(squareImage(16, 8, 8, 4, 0, 200))
	for _, v := range sal.Data {
		if v != 0 {
			t.Fatal("after Reset the next frame should be treated as the first (zero map)")
		}
	}
}

// TestMotionRunningAverage checks the running-average detector flags a newly
// appeared object against a learned background and exposes that background.
func TestMotionRunningAverage(t *testing.T) {
	det := saliency2.NewMotionSaliencyRunningAverage()
	det.Sigma = 0
	det.Alpha = 0.1

	bg := cv.NewMat(32, 32, 1)
	bg.SetTo(50)
	// Prime the background model with several identical empty frames.
	for i := 0; i < 5; i++ {
		det.ComputeSaliencyMap(bg)
	}
	if det.Background() == nil {
		t.Fatal("Background should be available after priming")
	}

	withObj := bg.Clone()
	for y := 10; y < 18; y++ {
		for x := 10; x < 18; x++ {
			withObj.Set(y, x, 0, 220)
		}
	}
	sal := det.ComputeSaliencyMap(withObj)
	object := sal.At(14, 14)
	corner := sal.At(2, 2)
	if !(object > corner) {
		t.Fatalf("object %.1f should exceed background corner %.1f", object, corner)
	}

	det.Reset()
	if det.Background() != nil {
		t.Fatal("Background should be nil after Reset")
	}
}
