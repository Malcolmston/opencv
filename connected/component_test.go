package connected

import "testing"

func TestComponentStats(t *testing.T) {
	img := matFromRows([]string{
		"##...",
		"##...",
		".....",
		"....#",
		"...##",
	})
	comps := ComponentStats(img, Conn8)
	if len(comps) != 2 {
		t.Fatalf("got %d components, want 2", len(comps))
	}
	c0 := comps[0]
	if c0.Area != 4 {
		t.Errorf("comp0 area = %d, want 4", c0.Area)
	}
	if c0.CentroidX != 0.5 || c0.CentroidY != 0.5 {
		t.Errorf("comp0 centroid = (%v,%v), want (0.5,0.5)", c0.CentroidX, c0.CentroidY)
	}
	if c0.Width() != 2 || c0.Height() != 2 {
		t.Errorf("comp0 bbox size = %dx%d, want 2x2", c0.Width(), c0.Height())
	}
	if c0.Extent() != 1.0 {
		t.Errorf("comp0 extent = %v, want 1.0", c0.Extent())
	}
	if !c0.Contains(1, 1) || c0.Contains(3, 3) {
		t.Errorf("Contains behaves unexpectedly")
	}
}

func TestLargestAndSmallest(t *testing.T) {
	img := matFromRows([]string{
		"#....",
		".....",
		".###.",
		".###.",
		".....",
	})
	big, ok := LargestComponent(img, Conn4)
	if !ok || big.Area != 6 {
		t.Errorf("largest area = %d ok=%v, want 6 true", big.Area, ok)
	}
	small, ok := SmallestComponent(img, Conn4)
	if !ok || small.Area != 1 {
		t.Errorf("smallest area = %d ok=%v, want 1 true", small.Area, ok)
	}
	mask := LargestComponentMask(img, Conn4)
	if countValue(mask, 255) != 6 {
		t.Errorf("largest mask fg = %d, want 6", countValue(mask, 255))
	}
	// The single stray pixel must be gone.
	if mask.Data[0] != 0 {
		t.Errorf("stray pixel survived in largest mask")
	}
}

func TestFilterByAreaAndRemoveSmall(t *testing.T) {
	img := matFromRows([]string{
		"#..##",
		"...##",
		".....",
		"###..",
		"###..",
	})
	// Components: single pixel (1), 2x2 block (4), 3x2 block (6).
	kept := RemoveSmallComponents(img, Conn4, 4)
	if countValue(kept, 255) != 10 {
		t.Errorf("despeckle fg = %d, want 10", countValue(kept, 255))
	}
	// Keep only components with area exactly 4.
	only4 := FilterByArea(img, Conn4, 4, 4)
	if countValue(only4, 255) != 4 {
		t.Errorf("area==4 fg = %d, want 4", countValue(only4, 255))
	}
}

func TestKeepLargestN(t *testing.T) {
	img := matFromRows([]string{
		"#..##",
		"...##",
		".....",
		"###..",
		"###..",
	})
	// Areas are 1, 4, 6. Keep the two largest -> 10 pixels.
	two := KeepLargestN(img, Conn4, 2)
	if countValue(two, 255) != 10 {
		t.Errorf("keep 2 largest fg = %d, want 10", countValue(two, 255))
	}
	// n larger than component count keeps everything.
	all := KeepLargestN(img, Conn4, 99)
	if countValue(all, 255) != 11 {
		t.Errorf("keep all fg = %d, want 11", countValue(all, 255))
	}
	none := KeepLargestN(img, Conn4, 0)
	if countValue(none, 255) != 0 {
		t.Errorf("keep 0 fg = %d, want 0", countValue(none, 255))
	}
}

func TestAreasAndBoundingBoxes(t *testing.T) {
	img := matFromRows([]string{
		"##...",
		"##...",
		"....#",
	})
	areas := Areas(img, Conn4)
	if len(areas) != 2 || areas[0] != 4 || areas[1] != 1 {
		t.Errorf("areas = %v, want [4 1]", areas)
	}
	boxes := BoundingBoxes(img, Conn4)
	if len(boxes) != 2 {
		t.Fatalf("got %d boxes, want 2", len(boxes))
	}
	if boxes[0].Width != 2 || boxes[0].Height != 2 {
		t.Errorf("box0 = %+v", boxes[0])
	}
	if boxes[1].X != 4 || boxes[1].Y != 2 {
		t.Errorf("box1 = %+v, want origin (4,2)", boxes[1])
	}
}
