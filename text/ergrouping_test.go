package text

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestERGroupingHorizontalTwoRows(t *testing.T) {
	var boxes []cv.Rect
	for _, x := range []int{0, 12, 24, 36} {
		boxes = append(boxes, cv.Rect{X: x, Y: 0, Width: 8, Height: 10})
		boxes = append(boxes, cv.Rect{X: x, Y: 40, Width: 8, Height: 10})
	}
	lines := ERGrouping(boxes, OrientationHoriz)
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2: %+v", len(lines), lines)
	}
	for i, l := range lines {
		if len(l) != 4 {
			t.Errorf("line %d has %d boxes, want 4", i, len(l))
		}
	}
	if lines[0][0].Y >= lines[1][0].Y {
		t.Errorf("lines not ordered top-to-bottom")
	}
}

func TestERGroupingAnyLinksDiagonal(t *testing.T) {
	// A gently descending run of boxes: OrientationHoriz splits it (each step
	// exceeds the vertical tolerance relative to the small gap), OrientationAny
	// links it into a single line because the slope stays bounded.
	boxes := []cv.Rect{
		{X: 0, Y: 0, Width: 8, Height: 10},
		{X: 12, Y: 6, Width: 8, Height: 10},
		{X: 24, Y: 12, Width: 8, Height: 10},
		{X: 36, Y: 18, Width: 8, Height: 10},
	}
	horiz := ERGrouping(boxes, OrientationHoriz)
	any := ERGrouping(boxes, OrientationAny)
	if len(any) != 1 {
		t.Errorf("OrientationAny got %d lines, want 1: %+v", len(any), any)
	}
	if len(horiz) <= len(any) {
		t.Errorf("expected OrientationHoriz to split more than OrientationAny, got %d vs %d", len(horiz), len(any))
	}
}

func TestERGroupingBBoxMerges(t *testing.T) {
	// Two nearby boxes of different heights merge under the bbox grouper even
	// though the height-aware ERGrouping might keep them apart.
	boxes := []cv.Rect{
		{X: 0, Y: 0, Width: 8, Height: 10},
		{X: 10, Y: 2, Width: 8, Height: 8},
		{X: 200, Y: 0, Width: 8, Height: 10}, // far away: its own group
	}
	groups := ERGroupingBBox(boxes)
	if len(groups) != 2 {
		t.Fatalf("got %d groups, want 2: %+v", len(groups), groups)
	}
	largest := 0
	for _, g := range groups {
		if len(g) > largest {
			largest = len(g)
		}
	}
	if largest != 2 {
		t.Errorf("largest group has %d boxes, want 2: %+v", largest, groups)
	}
}

func TestERGroupingEmpty(t *testing.T) {
	if got := ERGrouping(nil, OrientationHoriz); got != nil {
		t.Errorf("empty ERGrouping = %+v, want nil", got)
	}
	if got := ERGroupingBBox(nil); got != nil {
		t.Errorf("empty ERGroupingBBox = %+v, want nil", got)
	}
}

func TestLineBoundingBox(t *testing.T) {
	group := []cv.Rect{
		{X: 5, Y: 4, Width: 6, Height: 10},
		{X: 20, Y: 2, Width: 6, Height: 12},
	}
	got := LineBoundingBox(group)
	want := cv.Rect{X: 5, Y: 2, Width: 21, Height: 12}
	if got != want {
		t.Errorf("LineBoundingBox = %+v, want %+v", got, want)
	}
}

func TestLineBoundingBoxPanicsEmpty(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Errorf("expected panic on empty group")
		}
	}()
	LineBoundingBox(nil)
}
