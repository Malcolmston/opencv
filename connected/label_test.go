package connected

import "testing"

func TestLabelTwoComponents(t *testing.T) {
	img := matFromRows([]string{
		"#....",
		"#....",
		"...##",
		"...##",
		".....",
	})
	lbl := Label(img, Conn4)
	if lbl.Count != 2 {
		t.Fatalf("Count = %d, want 2", lbl.Count)
	}
	areas := lbl.Areas()
	// areas[0] is background: 25 total - 6 foreground.
	if areas[0] != 19 {
		t.Errorf("background area = %d, want 19", areas[0])
	}
	// Left component (label 1, first in raster order) has area 2; the 2x2 block
	// (label 2) has area 4.
	if areas[1] != 2 || areas[2] != 4 {
		t.Errorf("component areas = %v, want [_,2,4]", areas)
	}
	if got := lbl.At(0, 0); got != 1 {
		t.Errorf("At(0,0) = %d, want 1", got)
	}
	if got := lbl.At(4, 3); got != 2 {
		t.Errorf("At(4,3) = %d, want 2", got)
	}
	if got := lbl.At(2, 0); got != 0 {
		t.Errorf("At(2,0) = %d, want 0 (background)", got)
	}
}

func TestLabelConnectivityDifference(t *testing.T) {
	img := matFromRows([]string{
		"#.",
		".#",
	})
	if c := CountComponents(img, Conn4); c != 2 {
		t.Errorf("4-connected count = %d, want 2", c)
	}
	if c := CountComponents(img, Conn8); c != 1 {
		t.Errorf("8-connected count = %d, want 1", c)
	}
}

func TestLabelMaskAndCentroid(t *testing.T) {
	img := matFromRows([]string{
		".....",
		".###.",
		".###.",
		".###.",
		".....",
	})
	lbl := Label(img, Conn8)
	if lbl.Count != 1 {
		t.Fatalf("Count = %d, want 1", lbl.Count)
	}
	cx, cy := lbl.Centroid(1)
	if cx != 2 || cy != 2 {
		t.Errorf("centroid = (%v,%v), want (2,2)", cx, cy)
	}
	mask := lbl.Mask(1)
	if countValue(mask, 255) != 9 {
		t.Errorf("mask foreground = %d, want 9", countValue(mask, 255))
	}
	bb := lbl.BoundingBox(1)
	if bb.X != 1 || bb.Y != 1 || bb.Width != 3 || bb.Height != 3 {
		t.Errorf("bbox = %+v, want {1 1 3 3}", bb)
	}
}

func TestLabelEmptyImage(t *testing.T) {
	img := matFromRows([]string{
		"...",
		"...",
	})
	lbl := Label(img, Conn4)
	if lbl.Count != 0 {
		t.Errorf("Count = %d, want 0", lbl.Count)
	}
	if lbl.Components() != nil {
		t.Errorf("Components() = %v, want nil", lbl.Components())
	}
}

func TestLabelColorImageDeterministic(t *testing.T) {
	img := matFromRows([]string{
		"#.#",
		"...",
		"#.#",
	})
	lbl := Label(img, Conn4)
	a := lbl.ColorImage()
	b := lbl.ColorImage()
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("ColorImage not deterministic at %d", i)
		}
	}
	if a.Channels != 3 {
		t.Errorf("ColorImage channels = %d, want 3", a.Channels)
	}
	// Background stays black.
	if a.Data[3*1] != 0 || a.Data[3*1+1] != 0 || a.Data[3*1+2] != 0 {
		t.Errorf("background pixel not black")
	}
}
