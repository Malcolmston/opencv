package segmentation

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// fillRect paints a solid-colour axis-aligned rectangle [x0,x1)x[y0,y1) into m.
func fillRect(m *cv.Mat, x0, y0, x1, y1 int, vals ...uint8) {
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			m.SetPixel(y, x, vals)
		}
	}
}

func TestFloodFillSolidRegion(t *testing.T) {
	// A 40x30 gray canvas at value 50 with a solid rectangle of value 200
	// spanning columns [10,25) and rows [5,20).
	m := cv.NewMat(30, 40, 1)
	m.SetTo(50)
	const (
		rx0, ry0 = 10, 5
		rx1, ry1 = 25, 20
	)
	fillRect(m, rx0, ry0, rx1, ry1, 200)

	// Seed inside the rectangle, zero tolerance: the fill must exactly cover the
	// rectangle and stop at the surrounding colour edge.
	count, rect := FloodFill(m, cv.Point{X: 15, Y: 10}, cv.NewScalar(120),
		cv.NewScalar(0), cv.NewScalar(0), 4)

	wantArea := (rx1 - rx0) * (ry1 - ry0)
	if count != wantArea {
		t.Fatalf("count = %d, want %d", count, wantArea)
	}
	want := cv.Rect{X: rx0, Y: ry0, Width: rx1 - rx0, Height: ry1 - ry0}
	if rect != want {
		t.Fatalf("rect = %+v, want %+v", rect, want)
	}
	// Every rectangle pixel is now the new value and the fill did not leak.
	if got := m.At(10, 15, 0); got != 120 {
		t.Errorf("interior pixel = %d, want 120", got)
	}
	if got := m.At(0, 0, 0); got != 50 {
		t.Errorf("background pixel changed to %d, want 50 (fill leaked past edge)", got)
	}
	if got := m.At(ry1, rx0, 0); got != 50 {
		t.Errorf("pixel just below rect = %d, want 50", got)
	}
}

func TestFloodFillConnectivity(t *testing.T) {
	// Two same-coloured blobs touching only at a corner. 4-connectivity must not
	// cross the diagonal; 8-connectivity must.
	m := cv.NewMat(6, 6, 1)
	m.SetTo(0)
	m.Set(1, 1, 0, 100)
	m.Set(2, 2, 0, 100)

	m4 := m.Clone()
	c4, _ := FloodFill(m4, cv.Point{X: 1, Y: 1}, cv.NewScalar(200),
		cv.NewScalar(0), cv.NewScalar(0), 4)
	if c4 != 1 {
		t.Errorf("4-connected count = %d, want 1", c4)
	}

	m8 := m.Clone()
	c8, _ := FloodFill(m8, cv.Point{X: 1, Y: 1}, cv.NewScalar(200),
		cv.NewScalar(0), cv.NewScalar(0), 8)
	if c8 != 2 {
		t.Errorf("8-connected count = %d, want 2", c8)
	}
}

func TestWatershedTwoBasins(t *testing.T) {
	// A 20x20 image split by a bright vertical ridge at column 10. Two markers,
	// one per side, must produce two basins with a watershed line along the ridge.
	rows, cols := 20, 20
	img := cv.NewMat(rows, cols, 1)
	img.SetTo(40)
	for y := 0; y < rows; y++ {
		img.Set(y, 10, 0, 240) // the ridge
	}

	markers := cv.NewMat(rows, cols, 1)
	markers.Set(10, 3, 0, 1)  // left seed
	markers.Set(10, 16, 0, 2) // right seed

	labels := Watershed(img, markers)

	// A representative pixel on each side gets the corresponding basin label.
	if got := labels.At(5, 3, 0); got != 1 {
		t.Errorf("left pixel label = %d, want 1", got)
	}
	if got := labels.At(15, 16, 0); got != 2 {
		t.Errorf("right pixel label = %d, want 2", got)
	}

	// Some watershed-line pixels exist and they sit on the ridge column.
	boundary := 0
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if labels.At(y, x, 0) == WatershedMarker {
				boundary++
				if x < 8 || x > 12 {
					t.Errorf("watershed pixel at (%d,%d) is far from the ridge", x, y)
				}
			}
		}
	}
	if boundary == 0 {
		t.Error("no watershed-line pixels found between the two basins")
	}

	// Left and right halves are cleanly separated: no label-2 pixel on the far
	// left, no label-1 pixel on the far right.
	for y := 0; y < rows; y++ {
		if labels.At(y, 1, 0) == 2 {
			t.Errorf("label 2 leaked to the left edge at row %d", y)
		}
		if labels.At(y, 18, 0) == 1 {
			t.Errorf("label 1 leaked to the right edge at row %d", y)
		}
	}
}

func TestGrabCutForegroundRect(t *testing.T) {
	// Blue background with a solid red foreground rectangle. GrabCut initialised
	// with a rectangle around the red block should label most of it foreground
	// and most of the background background.
	rows, cols := 40, 40
	img := cv.NewMat(rows, cols, 3)
	fillRect(img, 0, 0, cols, rows, 30, 30, 200) // blue-ish background
	fillRect(img, 12, 12, 28, 28, 220, 40, 40)   // red foreground block

	rect := cv.Rect{X: 10, Y: 10, Width: 20, Height: 20}
	mask := GrabCut(img, nil, rect, 3)

	fgInside, fgTotal := 0, 0
	for y := 12; y < 28; y++ {
		for x := 12; x < 28; x++ {
			fgTotal++
			if mask.At(y, x, 0)&1 == 1 {
				fgInside++
			}
		}
	}
	if fgInside*100 < fgTotal*80 {
		t.Errorf("only %d/%d foreground pixels labelled fg (<80%%)", fgInside, fgTotal)
	}

	// The far corners (definite background outside rect) stay background.
	for _, p := range []cv.Point{{X: 0, Y: 0}, {X: 39, Y: 0}, {X: 0, Y: 39}, {X: 39, Y: 39}} {
		if mask.At(p.Y, p.X, 0)&1 == 1 {
			t.Errorf("background corner (%d,%d) labelled foreground", p.X, p.Y)
		}
	}
}

func TestMeanShiftSmoothsButKeepsEdges(t *testing.T) {
	// Two flat regions (left dark, right bright) with a deterministic checker of
	// noise added on top. Mean shift should shrink the within-region variation
	// while preserving the step between the two region means.
	rows, cols := 24, 24
	img := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			base := 60
			if x >= cols/2 {
				base = 190
			}
			// Deterministic +/-8 checkerboard noise.
			if (x+y)%2 == 0 {
				base += 8
			} else {
				base -= 8
			}
			img.SetPixel(y, x, []uint8{uint8(base), uint8(base), uint8(base)})
		}
	}

	// sr must exceed the noise gap (per-channel +/-8 => Euclidean ~27.7) to merge
	// within-region samples, while staying far below the 130-per-channel edge.
	out := MeanShiftFiltering(img, 4, 40)

	// Within-region spread (max-min on channel 0) must shrink on the left region.
	spread := func(m *cv.Mat) int {
		lo, hi := 255, 0
		for y := 4; y < rows-4; y++ {
			for x := 2; x < cols/2-2; x++ {
				v := int(m.At(y, x, 0))
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
		}
		return hi - lo
	}
	inSpread := spread(img)
	outSpread := spread(out)
	if outSpread >= inSpread {
		t.Errorf("mean shift did not reduce noise: spread %d -> %d", inSpread, outSpread)
	}

	// The edge between region means is preserved: left interior stays near 60,
	// right interior near 190, still clearly separated.
	left := int(out.At(12, 5, 0))
	right := int(out.At(12, 18, 0))
	if right-left < 100 {
		t.Errorf("edge collapsed: left=%d right=%d (gap %d, want >=100)", left, right, right-left)
	}
}

func TestPyrMeanShiftDeterministicAndSmooths(t *testing.T) {
	rows, cols := 32, 32
	img := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			base := 70
			if x >= cols/2 {
				base = 180
			}
			if (x*3+y)%2 == 0 {
				base += 6
			}
			img.SetPixel(y, x, []uint8{uint8(base), uint8(base), uint8(base)})
		}
	}
	a := PyrMeanShiftFiltering(img, 4, 20, 1)
	b := PyrMeanShiftFiltering(img, 4, 20, 1)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("PyrMeanShiftFiltering not deterministic at index %d: %d vs %d", i, a.Data[i], b.Data[i])
		}
	}
	// Edge still present across the pyramid variant.
	if int(a.At(16, 24, 0))-int(a.At(16, 6, 0)) < 80 {
		t.Errorf("pyramid mean shift collapsed the region edge")
	}
}

func TestFloodFillPanicsOnBadConnectivity(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("expected panic for connectivity 6")
		}
	}()
	m := cv.NewMat(4, 4, 1)
	FloodFill(m, cv.Point{X: 1, Y: 1}, cv.NewScalar(1), cv.NewScalar(0), cv.NewScalar(0), 6)
}

func TestWatershedDeterministic(t *testing.T) {
	rows, cols := 16, 16
	img := cv.NewMat(rows, cols, 1)
	img.SetTo(40)
	for y := 0; y < rows; y++ {
		img.Set(y, 8, 0, 220)
	}
	markers := cv.NewMat(rows, cols, 1)
	markers.Set(8, 2, 0, 1)
	markers.Set(8, 13, 0, 2)
	a := Watershed(img, markers)
	b := Watershed(img, markers)
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("Watershed not deterministic at index %d", i)
		}
	}
}
