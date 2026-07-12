package text

import (
	"math/rand"
	"testing"

	cv "github.com/malcolmston/opencv"
)

func TestDetectTextLinesTwoRows(t *testing.T) {
	blobs := []blobSpec{
		{x: 4, y: 4, w: 5, h: 7}, {x: 14, y: 4, w: 5, h: 7},
		{x: 24, y: 4, w: 5, h: 7}, {x: 34, y: 4, w: 5, h: 7},
		{x: 4, y: 24, w: 5, h: 7}, {x: 14, y: 24, w: 5, h: 7},
		{x: 24, y: 24, w: 5, h: 7}, {x: 34, y: 24, w: 5, h: 7},
	}
	img := newBlobImage(36, 46, 220, 40, blobs)

	lines := DetectTextLines(img, DefaultTextDetectorParams())
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2: %+v", len(lines), lines)
	}
	for i, l := range lines {
		if len(l) != 4 {
			t.Errorf("line %d has %d chars, want 4", i, len(l))
		}
	}

	boxes := DetectRegions(img, DefaultTextDetectorParams())
	if len(boxes) != 2 {
		t.Fatalf("got %d line boxes, want 2: %+v", len(boxes), boxes)
	}
	// The first line box must cover the whole top row of blobs.
	top := boxes[0]
	if top.Y > 4 || top.X > 4 || top.X+top.Width < 39 {
		t.Errorf("top line box %+v does not cover the top row", top)
	}
}

// TestDetectRegionsSeededRows plants several random rows of evenly-spaced blobs
// with a fixed RNG seed and checks the pipeline recovers exactly that many lines,
// deterministically.
func TestDetectRegionsSeededRows(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	rows := 3
	perRow := 5
	rowGap := 24
	img := cv.NewMat(rows*rowGap+16, perRow*14+16, 1)
	img.SetTo(230)
	for r := 0; r < rows; r++ {
		y := 8 + r*rowGap
		// A small per-row jitter in height, still same-line consistent.
		h := 8 + rng.Intn(3)
		for c := 0; c < perRow; c++ {
			x := 8 + c*14
			for yy := y; yy < y+h; yy++ {
				for xx := x; xx < x+6; xx++ {
					img.Set(yy, xx, 0, 30)
				}
			}
		}
	}

	p := DefaultTextDetectorParams()
	lines := DetectTextLines(img, p)
	if len(lines) != rows {
		t.Fatalf("got %d lines, want %d: %+v", len(lines), rows, lines)
	}
	for i, l := range lines {
		if len(l) != perRow {
			t.Errorf("line %d has %d chars, want %d", i, len(l), perRow)
		}
	}

	// Determinism: a second run is byte-identical.
	again := DetectRegions(img, p)
	first := DetectRegions(img, p)
	if len(again) != len(first) {
		t.Fatalf("nondeterministic count %d vs %d", len(again), len(first))
	}
	for i := range again {
		if again[i] != first[i] {
			t.Errorf("line box %d differs: %+v vs %+v", i, again[i], first[i])
		}
	}
}
