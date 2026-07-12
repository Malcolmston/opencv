package cudaobjdetect

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// TestGroupRectangles checks the grouping pass-through clusters near-duplicates.
func TestGroupRectangles(t *testing.T) {
	rects := []cv.Rect{
		{X: 10, Y: 10, Width: 20, Height: 20},
		{X: 12, Y: 11, Width: 20, Height: 20},
		{X: 11, Y: 12, Width: 21, Height: 20},
		{X: 100, Y: 100, Width: 20, Height: 20}, // singleton outlier
	}
	got := GroupRectangles(rects, 2, 0)
	if len(got) != 1 {
		t.Fatalf("GroupRectangles = %d groups, want 1 (%v)", len(got), got)
	}
}

// TestGroupRectanglesWeights checks the weighted grouping pass-through keeps the
// cluster-maximum score.
func TestGroupRectanglesWeights(t *testing.T) {
	rects := []cv.Rect{
		{X: 10, Y: 10, Width: 20, Height: 20},
		{X: 12, Y: 11, Width: 20, Height: 20},
		{X: 11, Y: 12, Width: 20, Height: 20},
	}
	weights := []float64{0.4, 0.9, 0.5}
	gr, gw := GroupRectanglesWeights(rects, weights, 2, 0.2)
	if len(gr) != 1 || len(gw) != 1 {
		t.Fatalf("expected 1 grouped rect/weight, got %d/%d", len(gr), len(gw))
	}
	if gw[0] != 0.9 {
		t.Fatalf("grouped weight = %v, want 0.9", gw[0])
	}
}

// TestNMSBoxes checks greedy non-maximum suppression on overlapping boxes.
func TestNMSBoxes(t *testing.T) {
	boxes := []cv.Rect{
		{X: 0, Y: 0, Width: 10, Height: 10},   // score 0.9
		{X: 1, Y: 1, Width: 10, Height: 10},   // heavy overlap, score 0.8 -> suppressed
		{X: 50, Y: 50, Width: 10, Height: 10}, // distinct, score 0.7
	}
	scores := []float64{0.9, 0.8, 0.7}
	kept := NMSBoxes(boxes, scores, 0.5, 0.5)
	if len(kept) != 2 {
		t.Fatalf("NMSBoxes kept %d, want 2 (%v)", len(kept), kept)
	}
	if kept[0] != 0 {
		t.Fatalf("highest-scoring box should be kept first, got index %d", kept[0])
	}
	for _, idx := range kept {
		if idx == 1 {
			t.Fatal("overlapping lower-score box should have been suppressed")
		}
	}
}

// TestSoftNMSBoxes checks the Gaussian soft-NMS pass-through decays overlaps.
func TestSoftNMSBoxes(t *testing.T) {
	boxes := []cv.Rect{
		{X: 0, Y: 0, Width: 10, Height: 10},
		{X: 1, Y: 1, Width: 10, Height: 10},
	}
	scores := []float64{0.9, 0.8}
	idx, kept := SoftNMSBoxes(boxes, scores, 0.0, 0.5)
	if len(idx) != len(kept) {
		t.Fatalf("indices/scores mismatch %d vs %d", len(idx), len(kept))
	}
	if idx[0] != 0 || kept[0] != 0.9 {
		t.Fatalf("first kept should be box 0 with score 0.9, got idx=%d score=%v", idx[0], kept[0])
	}
	// The overlapping box's score must be decayed below its original 0.8.
	if len(kept) > 1 && kept[1] >= 0.8 {
		t.Fatalf("overlapping box score = %v, expected decay below 0.8", kept[1])
	}
}

// TestRectIoU checks the overlap metric pass-through.
func TestRectIoU(t *testing.T) {
	a := cv.Rect{X: 0, Y: 0, Width: 10, Height: 10}
	if iou := RectIoU(a, a); iou != 1 {
		t.Fatalf("RectIoU(a,a) = %v, want 1", iou)
	}
	b := cv.Rect{X: 100, Y: 100, Width: 10, Height: 10}
	if iou := RectIoU(a, b); iou != 0 {
		t.Fatalf("RectIoU(disjoint) = %v, want 0", iou)
	}
}
