package objdetect

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

// topBottomSoftCascade builds an 8x8 soft cascade with a single top-minus-bottom
// Haar feature and stump, matching the hand-crafted Haar test cascade.
func topBottomSoftCascade() *SoftCascade {
	return &SoftCascade{
		WinW: 8, WinH: 8,
		Features: []SoftFeature{
			{Rects: []WeightedRect{
				{X: 0, Y: 0, W: 8, H: 4, Weight: 1},
				{X: 0, Y: 4, W: 8, H: 4, Weight: -1},
			}},
		},
		Stumps: []SoftStump{
			{Feature: 0, Threshold: 0.5, Left: -1, Right: 1},
		},
		Reject: nil, // final score compared to 0
	}
}

func TestSoftCascadeDetect(t *testing.T) {
	sc := topBottomSoftCascade()

	pos := cv.NewMat(8, 8, 1)
	for y := 0; y < 4; y++ {
		for x := 0; x < 8; x++ {
			pos.Set(y, x, 0, 255)
		}
	}
	hits := sc.DetectMultiScale(pos)
	if len(hits) != 1 || hits[0] != (cv.Rect{X: 0, Y: 0, Width: 8, Height: 8}) {
		t.Fatalf("expected one detection {0 0 8 8}, got %v", hits)
	}

	neg := cv.NewMat(8, 8, 1)
	for y := 4; y < 8; y++ {
		for x := 0; x < 8; x++ {
			neg.Set(y, x, 0, 255)
		}
	}
	if hits := sc.DetectMultiScale(neg); len(hits) != 0 {
		t.Fatalf("expected no detection on inverted window, got %v", hits)
	}
}

func TestSoftCascadeEarlyExit(t *testing.T) {
	// Two stumps; feature 0 is the whole-window sum. Stump 0 always emits -10.
	sc := &SoftCascade{
		WinW: 8, WinH: 8,
		Features: []SoftFeature{
			{Rects: []WeightedRect{{X: 0, Y: 0, W: 8, H: 8, Weight: 1}}},
		},
		Stumps: []SoftStump{
			{Feature: 0, Threshold: math.MaxFloat64, Left: -10, Right: 10}, // always Left
			{Feature: 0, Threshold: math.MaxFloat64, Left: 1, Right: 1},
		},
	}
	img := cv.NewMat(8, 8, 1)
	img.SetTo(100)
	ii := NewIntegralImage(img)

	// High rejection threshold after stump 0 -> reject after 1 stump.
	sc.Reject = []float64{-1, math.Inf(-1)}
	ok, evaluated := sc.evalWindow(ii, 0, 0, 1.0)
	if ok || evaluated != 1 {
		t.Fatalf("early exit: ok=%v evaluated=%d, want false,1", ok, evaluated)
	}

	// Permissive threshold after stump 0 -> both stumps evaluated, accepted.
	sc.Reject = []float64{-100, -100}
	ok, evaluated = sc.evalWindow(ii, 0, 0, 1.0)
	if !ok || evaluated != 2 {
		t.Fatalf("no early exit: ok=%v evaluated=%d, want true,2", ok, evaluated)
	}
}

func TestSoftCascadeFromClassifier(t *testing.T) {
	var clf CascadeClassifier
	if err := clf.LoadFromString(oneStageCascadeXML); err != nil {
		t.Fatalf("LoadFromString: %v", err)
	}
	sc := clf.ToSoftCascade()
	if len(sc.Stumps) != 1 || len(sc.Reject) != 1 {
		t.Fatalf("converted cascade has %d stumps / %d rejects, want 1/1", len(sc.Stumps), len(sc.Reject))
	}
	pos := cv.NewMat(8, 8, 1)
	for y := 0; y < 4; y++ {
		for x := 0; x < 8; x++ {
			pos.Set(y, x, 0, 255)
		}
	}
	hits := sc.DetectMultiScale(pos)
	if len(hits) != 1 || hits[0] != (cv.Rect{X: 0, Y: 0, Width: 8, Height: 8}) {
		t.Fatalf("converted soft cascade detection = %v, want [{0 0 8 8}]", hits)
	}
}

func TestSoftCascadeNoStumpsPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on empty soft cascade")
		}
	}()
	sc := &SoftCascade{WinW: 8, WinH: 8}
	sc.DetectMultiScale(cv.NewMat(8, 8, 1))
}
