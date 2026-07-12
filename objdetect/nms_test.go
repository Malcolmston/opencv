package objdetect

import (
	"math"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestRectIoU(t *testing.T) {
	a := cv.Rect{X: 0, Y: 0, Width: 10, Height: 10}
	if got := RectIoU(a, a); got != 1 {
		t.Fatalf("IoU of identical rects = %v, want 1", got)
	}
	disjoint := cv.Rect{X: 100, Y: 100, Width: 10, Height: 10}
	if got := RectIoU(a, disjoint); got != 0 {
		t.Fatalf("IoU of disjoint rects = %v, want 0", got)
	}
	// A and B overlap in a 5x5 corner. inter=25, union=100+100-25=175.
	b := cv.Rect{X: 5, Y: 5, Width: 10, Height: 10}
	want := 25.0 / 175.0
	if got := RectIoU(a, b); math.Abs(got-want) > 1e-9 {
		t.Fatalf("IoU = %v, want %v", got, want)
	}
}

func TestNMSBoxes(t *testing.T) {
	boxes := []cv.Rect{
		{X: 0, Y: 0, Width: 10, Height: 10},     // 0: high score
		{X: 1, Y: 1, Width: 10, Height: 10},     // 1: overlaps 0 heavily, lower
		{X: 100, Y: 100, Width: 10, Height: 10}, // 2: far away
		{X: 200, Y: 200, Width: 10, Height: 10}, // 3: low score, filtered out
	}
	scores := []float64{0.9, 0.8, 0.7, 0.1}
	kept := NMSBoxes(boxes, scores, 0.5, 0.5)
	// Expect boxes 0 (suppresses 1) and 2; box 3 filtered by score threshold.
	if len(kept) != 2 {
		t.Fatalf("kept %d boxes, want 2 (%v)", len(kept), kept)
	}
	if kept[0] != 0 {
		t.Fatalf("first kept = %d, want 0 (highest score)", kept[0])
	}
	found2 := false
	for _, k := range kept {
		if k == 1 {
			t.Fatalf("box 1 should have been suppressed by box 0")
		}
		if k == 2 {
			found2 = true
		}
	}
	if !found2 {
		t.Fatal("box 2 (far away) should survive")
	}
}

func TestNMSBoxesLengthMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on length mismatch")
		}
	}()
	NMSBoxes([]cv.Rect{{}}, []float64{}, 0, 0.5)
}

func TestSoftNMSBoxes(t *testing.T) {
	boxes := []cv.Rect{
		{X: 0, Y: 0, Width: 10, Height: 10},
		{X: 1, Y: 1, Width: 10, Height: 10}, // heavy overlap with box 0
		{X: 100, Y: 100, Width: 10, Height: 10},
	}
	scores := []float64{0.9, 0.85, 0.8}
	idx, kept := SoftNMSBoxes(boxes, scores, 0.05, 0.5)
	if len(idx) != len(kept) {
		t.Fatalf("indices/scores length mismatch: %d vs %d", len(idx), len(kept))
	}
	// The first selected must be the global max, box 0, at its original score.
	if idx[0] != 0 || math.Abs(kept[0]-0.9) > 1e-12 {
		t.Fatalf("first selection = idx %d score %v, want idx 0 score 0.9", idx[0], kept[0])
	}
	// Box 1 overlaps box 0 heavily, so its decayed score must drop below its
	// original 0.85.
	for i, id := range idx {
		if id == 1 && kept[i] >= 0.85 {
			t.Fatalf("box 1 score not decayed: %v", kept[i])
		}
	}
	// Scores must be in non-increasing order (selection order).
	for i := 1; i < len(kept); i++ {
		if kept[i] > kept[i-1]+1e-12 {
			t.Fatalf("scores not descending at %d: %v > %v", i, kept[i], kept[i-1])
		}
	}
}
