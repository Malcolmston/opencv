package text

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestProjectionProfileVertical(t *testing.T) {
	img := RenderText("11", 2, 1)
	prof := ProjectionProfile(img, 127, true)
	if len(prof) != img.Cols {
		t.Fatalf("vertical profile length %d, want %d", len(prof), img.Cols)
	}
	// There must be at least one empty column (the gap between the two glyphs).
	empty := 0
	for _, v := range prof {
		if v == 0 {
			empty++
		}
	}
	if empty == 0 {
		t.Errorf("expected empty columns between glyphs, got none: %v", prof)
	}
}

func TestSegmentCharsCounts(t *testing.T) {
	img := RenderText("ABCDE", 3, 1)
	boxes := SegmentChars(img, 127)
	if len(boxes) != 5 {
		t.Fatalf("got %d char boxes, want 5: %+v", len(boxes), boxes)
	}
	// Boxes are ordered left-to-right and do not overlap horizontally.
	for i := 1; i < len(boxes); i++ {
		if boxes[i].X < boxes[i-1].X+boxes[i-1].Width {
			t.Errorf("char boxes overlap or are unsorted: %+v", boxes)
		}
	}
}

func TestSegmentLinesCounts(t *testing.T) {
	top := RenderText("AB", 2, 1)
	bottom := RenderText("CD", 2, 1)
	cols := top.Cols
	if bottom.Cols > cols {
		cols = bottom.Cols
	}
	img := cv.NewMat(top.Rows+8+bottom.Rows, cols, 1)
	top.CopyTo(img, 0, 0)
	bottom.CopyTo(img, top.Rows+8, 0)

	lines := SegmentLines(img, 127)
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2: %+v", len(lines), lines)
	}
	if lines[0].Y >= lines[1].Y {
		t.Errorf("lines not ordered top-to-bottom: %+v", lines)
	}
}
