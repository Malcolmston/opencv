package cv

import "testing"

func TestLineHorizontal(t *testing.T) {
	m := NewMat(5, 5, 1)
	Line(m, Point{1, 2}, Point{3, 2}, NewScalar(255), 1)
	for x := 1; x <= 3; x++ {
		if m.At(2, x, 0) != 255 {
			t.Errorf("line pixel (%d,2) not set", x)
		}
	}
	if m.At(0, 0, 0) != 0 {
		t.Error("line touched unrelated pixel")
	}
}

func TestRectangleOutlineAndFill(t *testing.T) {
	m := NewMat(6, 6, 1)
	Rectangle(m, Point{1, 1}, Point{4, 4}, NewScalar(255), 1)
	// Corner is on the outline.
	if m.At(1, 1, 0) != 255 {
		t.Error("rectangle corner not drawn")
	}
	// Interior stays empty for an outline.
	if m.At(2, 2, 0) != 0 {
		t.Error("rectangle outline filled interior")
	}

	f := NewMat(6, 6, 1)
	Rectangle(f, Point{1, 1}, Point{4, 4}, NewScalar(255), Filled)
	if f.At(2, 2, 0) != 255 {
		t.Error("filled rectangle interior not set")
	}
}

func TestCircleFilledCentre(t *testing.T) {
	m := NewMat(11, 11, 1)
	Circle(m, Point{5, 5}, 3, NewScalar(255), Filled)
	if m.At(5, 5, 0) != 255 {
		t.Error("filled circle centre not set")
	}
	// A point well outside the radius stays clear.
	if m.At(0, 0, 0) != 0 {
		t.Error("filled circle overreached")
	}
}

func TestCircleOutline(t *testing.T) {
	m := NewMat(11, 11, 1)
	Circle(m, Point{5, 5}, 3, NewScalar(255), 1)
	// A point on the horizontal extreme of the circle is set.
	if m.At(5, 8, 0) != 255 && m.At(5, 2, 0) != 255 {
		t.Error("circle outline extreme not drawn")
	}
	// The centre is empty for an outline.
	if m.At(5, 5, 0) != 0 {
		t.Error("circle outline filled the centre")
	}
}

func TestFillPolyTriangle(t *testing.T) {
	m := NewMat(10, 10, 1)
	tri := []Point{{2, 1}, {8, 1}, {5, 8}}
	FillPoly(m, [][]Point{tri}, NewScalar(255))
	// A point clearly inside the triangle.
	if m.At(3, 5, 0) != 255 {
		t.Error("point inside triangle not filled")
	}
	// A point clearly outside.
	if m.At(8, 0, 0) != 0 {
		t.Error("point outside triangle was filled")
	}
}

func TestPolylinesClosed(t *testing.T) {
	m := NewMat(10, 10, 1)
	sq := []Point{{2, 2}, {7, 2}, {7, 7}, {2, 7}}
	Polylines(m, [][]Point{sq}, true, NewScalar(255), 1)
	// Closing edge from last back to first vertical line at x=2.
	if m.At(4, 2, 0) != 255 {
		t.Error("closed polyline missing closing edge")
	}
}

func TestPutTextDrawsPixels(t *testing.T) {
	m := NewMat(20, 60, 1)
	before := 0
	for _, v := range m.Data {
		if v != 0 {
			before++
		}
	}
	PutText(m, "AB", Point{2, 15}, 1, NewScalar(255))
	set := 0
	for _, v := range m.Data {
		if v == 255 {
			set++
		}
	}
	if set == 0 {
		t.Error("PutText drew nothing")
	}
	if before != 0 {
		t.Fatal("precondition failed")
	}
}

func TestRGBDrawingColor(t *testing.T) {
	m := NewMat(5, 5, 3)
	Rectangle(m, Point{0, 0}, Point{4, 4}, NewScalar(10, 20, 30), Filled)
	if m.At(2, 2, 0) != 10 || m.At(2, 2, 1) != 20 || m.At(2, 2, 2) != 30 {
		t.Errorf("rgb fill = %v", m.AtPixel(2, 2))
	}
}
