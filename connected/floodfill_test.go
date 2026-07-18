package connected

import "testing"

func TestFloodFillReplacesRegion(t *testing.T) {
	img := matFromRows([]string{
		"###..",
		"###..",
		"...##",
		"...##",
	})
	// Seed inside the top-left 3x2 block; fill it to background.
	n := FloodFill(img, 0, 0, 0, Conn4)
	if n != 6 {
		t.Fatalf("filled = %d, want 6", n)
	}
	// The other block (4 pixels) is untouched.
	if countValue(img, 255) != 4 {
		t.Errorf("remaining fg = %d, want 4", countValue(img, 255))
	}
}

func TestFloodFillNoOpWhenSameValue(t *testing.T) {
	img := matFromRows([]string{
		"##",
		"##",
	})
	if n := FloodFill(img, 0, 0, 255, Conn4); n != 0 {
		t.Errorf("no-op fill = %d, want 0", n)
	}
}

func TestFloodFillMask(t *testing.T) {
	img := matFromRows([]string{
		"#.#",
		"#.#",
		"###",
	})
	// This is a single 8- and 4-connected U shape (7 pixels).
	mask, n := FloodFillMask(img, 0, 0, Conn4)
	if n != 7 {
		t.Fatalf("region size = %d, want 7", n)
	}
	if countValue(mask, 255) != 7 {
		t.Errorf("mask fg = %d, want 7", countValue(mask, 255))
	}
	if mask.Data[1] != 0 { // (1,0) is background inside the U
		t.Errorf("background pixel marked in mask")
	}
}

func TestFloodFillDiagonalConnectivity(t *testing.T) {
	img := matFromRows([]string{
		"#.",
		".#",
	})
	// 4-connected: seed reaches only its own pixel.
	_, n4 := FloodFillMask(img, 0, 0, Conn4)
	if n4 != 1 {
		t.Errorf("4-conn region = %d, want 1", n4)
	}
	// 8-connected: the diagonal pixel joins.
	_, n8 := FloodFillMask(img, 0, 0, Conn8)
	if n8 != 2 {
		t.Errorf("8-conn region = %d, want 2", n8)
	}
}

func TestFloodFillTolerance(t *testing.T) {
	img := matFromRows([]string{
		"...",
		"...",
		"...",
	})
	// Build a graded region: seed value 10, neighbours 12 and 20.
	img.Data[0] = 10
	img.Data[1] = 12
	img.Data[2] = 20
	// tolerance 3 admits 10 and 12 (diff<=3 from seed 10) but not 20.
	n := FloodFillTolerance(img, 0, 0, 200, 3, Conn4)
	if n != 2 {
		t.Errorf("tolerance fill = %d, want 2", n)
	}
	if img.Data[0] != 200 || img.Data[1] != 200 {
		t.Errorf("expected pixels not filled: %v", img.Data[:3])
	}
	if img.Data[2] != 20 {
		t.Errorf("out-of-tolerance pixel was filled")
	}
}

func TestRegionPointsAndConnectedAt(t *testing.T) {
	img := matFromRows([]string{
		"##.",
		"##.",
		"..#",
	})
	pts := RegionPoints(img, 0, 0, Conn4)
	if len(pts) != 4 {
		t.Fatalf("region points = %d, want 4", len(pts))
	}
	set := pointSet(pts)
	for _, want := range [][2]int{{0, 0}, {1, 0}, {0, 1}, {1, 1}} {
		if !set[want] {
			t.Errorf("missing region point %v", want)
		}
	}
	// ConnectedAt on background returns empty mask.
	empty := ConnectedAt(img, 2, 0, Conn4)
	if countValue(empty, 255) != 0 {
		t.Errorf("ConnectedAt on background not empty")
	}
	// Under 4-connectivity the bottom-right pixel is isolated.
	comp := ConnectedAt(img, 2, 2, Conn4)
	if countValue(comp, 255) != 1 {
		t.Errorf("ConnectedAt single pixel = %d, want 1", countValue(comp, 255))
	}
	// Under 8-connectivity it joins the block diagonally (5 pixels total).
	comp8 := ConnectedAt(img, 2, 2, Conn8)
	if countValue(comp8, 255) != 5 {
		t.Errorf("ConnectedAt 8-conn = %d, want 5", countValue(comp8, 255))
	}
}
