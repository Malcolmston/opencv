package objdetect

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestGroupRectangles(t *testing.T) {
	rects := []cv.Rect{
		{X: 10, Y: 10, Width: 20, Height: 20},
		{X: 12, Y: 11, Width: 20, Height: 20},
		{X: 11, Y: 12, Width: 21, Height: 20},
		{X: 100, Y: 100, Width: 20, Height: 20}, // outlier singleton
	}
	got := GroupRectangles(rects, 2, 0)
	if len(got) != 1 {
		t.Fatalf("GroupRectangles returned %d groups, want 1 (%v)", len(got), got)
	}
	// Averaged centre of the first three near (11,11,20,20).
	r := got[0]
	if r.X < 9 || r.X > 13 || r.Y < 9 || r.Y > 13 {
		t.Fatalf("averaged rect %+v not near (11,11)", r)
	}
}

func TestGroupRectanglesMinNeighborsZero(t *testing.T) {
	rects := []cv.Rect{{X: 0, Y: 0, Width: 5, Height: 5}}
	got := GroupRectangles(rects, 0, 0.2)
	if len(got) != 1 || got[0] != rects[0] {
		t.Fatalf("minNeighbors=0 should pass rects through unchanged, got %v", got)
	}
}

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
	// Representative weight is the cluster maximum.
	if gw[0] != 0.9 {
		t.Fatalf("grouped weight = %v, want 0.9 (cluster max)", gw[0])
	}
}

func TestGroupRectanglesWeightsMismatchPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on length mismatch")
		}
	}()
	GroupRectanglesWeights([]cv.Rect{{}}, []float64{}, 1, 0.2)
}
