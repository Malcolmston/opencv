package text

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestGroupTextRegionsTwoRows(t *testing.T) {
	// Two horizontally-aligned rows of four same-height boxes each. Grouping
	// must return exactly two lines, each with its four boxes left-to-right.
	var regions []cv.Rect
	xs := []int{0, 12, 24, 36}
	for _, x := range xs {
		regions = append(regions, cv.Rect{X: x, Y: 0, Width: 8, Height: 10})  // top row
		regions = append(regions, cv.Rect{X: x, Y: 40, Width: 8, Height: 10}) // bottom row
	}

	lines := GroupTextRegions(regions)
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2: %+v", len(lines), lines)
	}
	for i, line := range lines {
		if len(line) != 4 {
			t.Fatalf("line %d has %d boxes, want 4", i, len(line))
		}
		for k := 1; k < len(line); k++ {
			if line[k].X < line[k-1].X {
				t.Errorf("line %d not sorted left-to-right: %+v", i, line)
			}
		}
	}
	// The top line must come first.
	if lines[0][0].Y >= lines[1][0].Y {
		t.Errorf("lines not ordered top-to-bottom: %+v", lines)
	}
}

func TestGroupTextRegionsSpacingSplit(t *testing.T) {
	// Same row, but a large horizontal gap separates two words. With the default
	// gap factor the two clusters become separate lines.
	regions := []cv.Rect{
		{X: 0, Y: 0, Width: 8, Height: 10},
		{X: 10, Y: 0, Width: 8, Height: 10},
		{X: 200, Y: 0, Width: 8, Height: 10},
		{X: 210, Y: 0, Width: 8, Height: 10},
	}
	lines := GroupTextRegions(regions)
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2 (spacing split): %+v", len(lines), lines)
	}
}

func TestGroupTextRegionsHeightMismatch(t *testing.T) {
	// A tall box next to short ones should not join their line.
	regions := []cv.Rect{
		{X: 0, Y: 0, Width: 8, Height: 10},
		{X: 12, Y: 0, Width: 8, Height: 10},
		{X: 24, Y: -20, Width: 8, Height: 50}, // very tall, mismatched
	}
	lines := GroupTextRegions(regions)
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2 (height split): %+v", len(lines), lines)
	}
}

func TestGroupTextRegionsEmpty(t *testing.T) {
	if got := GroupTextRegions(nil); got != nil {
		t.Errorf("empty input returned %+v, want nil", got)
	}
}

func TestERFilterRejectsNonCharacters(t *testing.T) {
	f := DefaultERFilter()
	char := Region{Rect: cv.Rect{X: 0, Y: 0, Width: 6, Height: 10}, Area: 48}
	if !f.Keep(char) {
		t.Errorf("expected character region to be kept: %+v", char)
	}
	// Extremely wide, sparse bar: bad aspect and fill.
	bar := Region{Rect: cv.Rect{X: 0, Y: 0, Width: 100, Height: 3}, Area: 60}
	if f.Keep(bar) {
		t.Errorf("expected wide bar to be rejected: %+v", bar)
	}
	// Sparse blob: low fill ratio.
	sparse := Region{Rect: cv.Rect{X: 0, Y: 0, Width: 10, Height: 10}, Area: 10}
	if f.Keep(sparse) {
		t.Errorf("expected sparse blob to be rejected: %+v", sparse)
	}

	kept := f.Filter([]Region{char, bar, sparse})
	if len(kept) != 1 || kept[0].Rect != char.Rect {
		t.Errorf("Filter kept %+v, want only the character", kept)
	}
}
