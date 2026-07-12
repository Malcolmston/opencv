package text

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// regionFromRows builds a Region from ASCII art, one rune per pixel, '#' meaning
// an inked (region) pixel. Coordinates are taken directly from the grid.
func regionFromRows(rows []string) Region {
	var pts []cv.Point
	for y, row := range rows {
		for x, ch := range row {
			if ch == '#' {
				pts = append(pts, cv.Point{X: x, Y: y})
			}
		}
	}
	rect := cv.BoundingRect(pts)
	return Region{Rect: rect, Points: pts, Area: len(pts)}
}

// regionFromGlyph builds a Region from a built-in font glyph rendered at scale.
func regionFromGlyph(ch rune, scale int) Region {
	m, ok := FontGlyph(ch, scale)
	if !ok {
		panic("unknown glyph")
	}
	var pts []cv.Point
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			if m.At(y, x, 0) > 127 {
				pts = append(pts, cv.Point{X: x, Y: y})
			}
		}
	}
	rect := cv.BoundingRect(pts)
	return Region{Rect: rect, Points: pts, Area: len(pts)}
}

func TestComputeERFeaturesTopology(t *testing.T) {
	// A hollow ring has exactly one enclosed hole; a solid block has none.
	ring := regionFromRows([]string{
		".#####.",
		"#.....#",
		"#.....#",
		"#.....#",
		".#####.",
	})
	f := ComputeERFeatures(ring)
	if f.Holes != 1 {
		t.Errorf("ring Holes = %d, want 1", f.Holes)
	}
	if f.Area != len(ring.Points) {
		t.Errorf("ring Area = %d, want %d", f.Area, len(ring.Points))
	}
	if f.Compactness <= 0 || f.Perimeter <= 0 {
		t.Errorf("ring compactness/perimeter degenerate: %+v", f)
	}

	block := regionFromRows([]string{
		"#####",
		"#####",
		"#####",
	})
	fb := ComputeERFeatures(block)
	if fb.Holes != 0 {
		t.Errorf("block Holes = %d, want 0", fb.Holes)
	}
	if fb.FillRatio != 1 {
		t.Errorf("block FillRatio = %v, want 1", fb.FillRatio)
	}
	// A solid block has near-constant stroke thickness away from corners.
	if fb.StrokeWidthMean <= 0 {
		t.Errorf("block StrokeWidthMean = %v, want > 0", fb.StrokeWidthMean)
	}
}

func TestERFilterNMKeepsAllFontLetters(t *testing.T) {
	nm1 := DefaultERFilterNM1()
	nm2 := DefaultERFilterNM2()
	for _, ch := range SupportedChars() {
		r := regionFromGlyph(ch, 3)
		if !nm1.Keep(r) {
			t.Errorf("NM1 rejected font glyph %q: %+v", string(ch), ComputeERFeatures(r))
		}
		if !nm2.Keep(r) {
			t.Errorf("NM2 rejected font glyph %q: %+v", string(ch), ComputeERFeatures(r))
		}
	}
}

func TestERFilterNM1RejectsThinBar(t *testing.T) {
	// A long, one-pixel-tall bar has an extreme aspect ratio: stage one drops it.
	var rows []string
	bar := ""
	for i := 0; i < 40; i++ {
		bar += "#"
	}
	rows = append(rows, bar)
	r := regionFromRows(rows)
	if DefaultERFilterNM1().Keep(r) {
		t.Errorf("NM1 kept a 40x1 bar: %+v", ComputeERFeatures(r))
	}
}

// thinX is a 1-pixel-wide diagonal cross: it fills the corners of its bounding
// box (so its convex hull is the full box) yet inks only a small fraction of it,
// giving a low convexity that stage two rejects while stage one accepts.
func thinX() Region {
	return regionFromRows([]string{
		"#.....#",
		".#...#.",
		"..#.#..",
		"...#...",
		"..#.#..",
		".#...#.",
		"#.....#",
	})
}

func TestERFilterNM2RejectsThinCross(t *testing.T) {
	x := thinX()
	if !DefaultERFilterNM1().Keep(x) {
		t.Fatalf("expected NM1 to keep the thin cross: %+v", ComputeERFeatures(x))
	}
	if DefaultERFilterNM2().Keep(x) {
		t.Errorf("expected NM2 to reject the thin cross: %+v", ComputeERFeatures(x))
	}
}

func TestERFilterNM2FilterComposition(t *testing.T) {
	letter := regionFromGlyph('A', 3)
	x := thinX()
	kept := DefaultERFilterNM2().Filter([]Region{letter, x})
	if len(kept) != 1 || kept[0].Rect != letter.Rect {
		t.Errorf("NM2 Filter kept %d regions, want only the letter", len(kept))
	}
}
