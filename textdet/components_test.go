package textdet

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// twoSquares builds a 10x12 mask with two separated 3x3 foreground squares:
// square A at cols 1..3, square B at cols 7..9, both on rows 2..4.
func twoSquares() *cv.Mat {
	m := cv.NewMat(10, 12, 1)
	paintRect(m, 1, 2, 3, 3, 255)
	paintRect(m, 7, 2, 3, 3, 255)
	return m
}

func TestLabelComponentsStats(t *testing.T) {
	set, err := LabelComponents(twoSquares(), Conn8)
	if err != nil {
		t.Fatal(err)
	}
	if set.Count() != 2 {
		t.Fatalf("Count = %d, want 2", set.Count())
	}
	for _, c := range set.Components {
		if c.Area != 9 {
			t.Fatalf("component area = %d, want 9", c.Area)
		}
		if c.Bounds.Width != 3 || c.Bounds.Height != 3 {
			t.Fatalf("component bounds = %+v, want 3x3", c.Bounds)
		}
	}
	// First component (lowest label) is square A, centroid at (2,3).
	a := set.Components[0]
	if a.CentroidX != 2 || a.CentroidY != 3 {
		t.Fatalf("A centroid = (%v,%v), want (2,3)", a.CentroidX, a.CentroidY)
	}
	if a.AspectRatio() != 1 {
		t.Fatalf("A aspect = %v, want 1", a.AspectRatio())
	}
	if a.FillRatio() != 1 {
		t.Fatalf("A fill = %v, want 1", a.FillRatio())
	}
}

func TestLabelComponentsConnectivity(t *testing.T) {
	// Two pixels touching only at a diagonal: one component under Conn8, two
	// under Conn4.
	m := cv.NewMat(3, 3, 1)
	m.Data[0*3+0] = 255
	m.Data[1*3+1] = 255
	if s, _ := LabelComponents(m, Conn8); s.Count() != 1 {
		t.Fatalf("Conn8 count = %d, want 1", s.Count())
	}
	if s, _ := LabelComponents(m, Conn4); s.Count() != 2 {
		t.Fatalf("Conn4 count = %d, want 2", s.Count())
	}
}

func TestFilters(t *testing.T) {
	comps := []Component{
		{Label: 1, Area: 5, Bounds: cv.Rect{Width: 2, Height: 10}},   // tall, small
		{Label: 2, Area: 50, Bounds: cv.Rect{Width: 10, Height: 10}}, // square, big
		{Label: 3, Area: 100, Bounds: cv.Rect{Width: 20, Height: 4}}, // wide, big
	}
	if got := FilterBySize(comps, 10, 0); len(got) != 2 {
		t.Fatalf("FilterBySize kept %d, want 2", len(got))
	}
	if got := FilterByAspectRatio(comps, 0.8, 1.2); len(got) != 1 || got[0].Label != 2 {
		t.Fatalf("FilterByAspectRatio = %v, want [label 2]", got)
	}
	// Fill ratios: label1 = 5/20 = 0.25, label2 = 50/100 = 0.50,
	// label3 = 100/80 = 1.25; the [0.4,0.6] band keeps only label 2.
	if got := FilterByFillRatio(comps, 0.4, 0.6); len(got) != 1 || got[0].Label != 2 {
		t.Fatalf("FilterByFillRatio = %v, want [label 2]", got)
	}
}

func TestGroupTextLines(t *testing.T) {
	// Two squares on the same rows, horizontal gap of 3 pixels
	// (square A cols 1..3, square B cols 7..9 => gap = 7-(1+3) = 3).
	set, _ := LabelComponents(twoSquares(), Conn8)
	// maxGap 3 => single line containing both.
	lines, err := GroupTextLines(set.Components, 0.5, 3)
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) != 1 {
		t.Fatalf("maxGap 3 lines = %d, want 1", len(lines))
	}
	if len(lines[0].Components) != 2 {
		t.Fatalf("line has %d comps, want 2", len(lines[0].Components))
	}
	if lines[0].Bounds.X != 1 || lines[0].Bounds.Width != 9 {
		t.Fatalf("line bounds = %+v, want X=1 W=9", lines[0].Bounds)
	}
	// maxGap 2 => the 3-pixel gap splits them into two lines.
	lines2, _ := GroupTextLines(set.Components, 0.5, 2)
	if len(lines2) != 2 {
		t.Fatalf("maxGap 2 lines = %d, want 2", len(lines2))
	}
}

func TestEdgeDensityLocalization(t *testing.T) {
	// A striped (period-4 vertical stripes) block on a flat background yields
	// high edge density only over the block. A single-pixel checkerboard would
	// be invisible to the Sobel operator, so stripes of width 2 are used.
	m := newGray(30, 30, 128)
	for y := 5; y < 20; y++ {
		for x := 5; x < 20; x++ {
			if (x/2)%2 == 0 {
				m.Data[y*30+x] = 0
			} else {
				m.Data[y*30+x] = 255
			}
		}
	}
	dm, err := EdgeDensityMap(m, 2, 100)
	if err != nil {
		t.Fatal(err)
	}
	// Density inside the textured block exceeds density in the flat corner.
	inside := dm.At(12, 12)
	corner := dm.At(0, 0)
	if !(inside > corner) {
		t.Fatalf("edge density inside=%v not greater than corner=%v", inside, corner)
	}
	rects, err := LocalizeByEdgeDensity(m, 2, 100, 0.5, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(rects) == 0 {
		t.Fatalf("LocalizeByEdgeDensity found no regions")
	}
}

func TestComponentsEmpty(t *testing.T) {
	var empty cv.Mat
	if _, err := LabelComponents(&empty, Conn8); err != ErrEmpty {
		t.Fatalf("empty err = %v, want ErrEmpty", err)
	}
	if _, err := LabelComponents(newGray(3, 3, 0), 5); err != ErrInvalidArgument {
		t.Fatalf("bad conn err = %v, want ErrInvalidArgument", err)
	}
}
