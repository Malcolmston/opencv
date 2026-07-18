package connected

import "testing"

func TestBoundaryOfSquare(t *testing.T) {
	img := matFromRows([]string{
		".....",
		".###.",
		".###.",
		".###.",
		".....",
	})
	b := Boundary(img, Conn4)
	// The centre pixel (2,2) is fully surrounded, so 8 boundary pixels.
	if countValue(b, 255) != 8 {
		t.Errorf("boundary pixels = %d, want 8", countValue(b, 255))
	}
	if b.Data[2*5+2] != 0 {
		t.Errorf("interior pixel marked as boundary")
	}
	if p := Perimeter(img, Conn4); p != 8 {
		t.Errorf("perimeter = %d, want 8", p)
	}
}

func TestOuterBoundary(t *testing.T) {
	img := matFromRows([]string{
		".....",
		".###.",
		".###.",
		".###.",
		".....",
	})
	ob := OuterBoundary(img, Conn4)
	// The 4-connected outer ring around a 3x3 block has 12 pixels.
	if countValue(ob, 255) != 12 {
		t.Errorf("outer boundary = %d, want 12", countValue(ob, 255))
	}
	// No foreground pixel is marked.
	for y := 1; y <= 3; y++ {
		for x := 1; x <= 3; x++ {
			if ob.Data[y*5+x] != 0 {
				t.Errorf("foreground pixel (%d,%d) marked in outer boundary", x, y)
			}
		}
	}
}

func TestBoundaryPoints(t *testing.T) {
	img := matFromRows([]string{
		"##",
		"##",
	})
	pts := BoundaryPoints(img, Conn4)
	// Every pixel of a 2x2 block touches the image edge, so all are boundary.
	if len(pts) != 4 {
		t.Errorf("boundary points = %d, want 4", len(pts))
	}
}

func TestTraceBoundarySquare(t *testing.T) {
	img := matFromRows([]string{
		".....",
		".###.",
		".###.",
		".###.",
		".....",
	})
	trace := TraceBoundary(img, 1, 1)
	// The 3x3 block's outer contour is its 8 ring pixels (centre excluded).
	if len(trace) != 8 {
		t.Fatalf("trace length = %d, want 8", len(trace))
	}
	if trace[0].X != 1 || trace[0].Y != 1 {
		t.Errorf("trace start = %+v, want (1,1)", trace[0])
	}
	set := pointSet(trace)
	// Centre must not appear; all ring pixels must.
	if set[[2]int{2, 2}] {
		t.Errorf("centre pixel appears in trace")
	}
	ring := [][2]int{{1, 1}, {2, 1}, {3, 1}, {3, 2}, {3, 3}, {2, 3}, {1, 3}, {1, 2}}
	for _, p := range ring {
		if !set[p] {
			t.Errorf("ring pixel %v missing from trace", p)
		}
	}
}

func TestTraceBoundaryIsolatedPixel(t *testing.T) {
	img := matFromRows([]string{
		"...",
		".#.",
		"...",
	})
	trace := TraceBoundary(img, 1, 1)
	if len(trace) != 1 || trace[0].X != 1 || trace[0].Y != 1 {
		t.Errorf("isolated trace = %+v, want single (1,1)", trace)
	}
}

func TestComponentBoundaries(t *testing.T) {
	img := matFromRows([]string{
		"##...",
		"##...",
		"....#",
	})
	bs := ComponentBoundaries(img, Conn4)
	if len(bs) != 2 {
		t.Fatalf("got %d boundaries, want 2", len(bs))
	}
	// The 2x2 block's contour is its 4 pixels; the single pixel is length 1.
	if len(bs[0]) != 4 {
		t.Errorf("block boundary length = %d, want 4", len(bs[0]))
	}
	if len(bs[1]) != 1 {
		t.Errorf("single-pixel boundary length = %d, want 1", len(bs[1]))
	}
}
