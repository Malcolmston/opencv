package draw2

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// px returns the c-th channel sample of pixel (x, y).
func px(m *cv.Mat, x, y, c int) uint8 {
	return m.Data[(y*m.Cols+x)*m.Channels+c]
}

// countNonZero counts pixels whose first channel is non-zero.
func countNonZero(m *cv.Mat) int {
	n := 0
	for i := 0; i < m.Rows*m.Cols; i++ {
		if m.Data[i*m.Channels] != 0 {
			n++
		}
	}
	return n
}

func white() cv.Scalar { return cv.Scalar{255, 255, 255, 255} }

func TestLineHorizontal(t *testing.T) {
	m := cv.NewMat(10, 10, 1)
	Line(m, cv.Point{X: 2, Y: 5}, cv.Point{X: 7, Y: 5}, white(), 1)
	for x := 2; x <= 7; x++ {
		if got := px(m, x, 5, 0); got != 255 {
			t.Fatalf("line pixel (%d,5)=%d, want 255", x, got)
		}
	}
	if countNonZero(m) != 6 {
		t.Fatalf("horizontal line lit %d pixels, want 6", countNonZero(m))
	}
}

func TestFilledRectangleCount(t *testing.T) {
	m := cv.NewMat(20, 20, 1)
	FilledRectangle(m, cv.Point{X: 3, Y: 4}, cv.Point{X: 8, Y: 9}, white())
	// inclusive span: (8-3+1)*(9-4+1) = 6*6 = 36
	if got := countNonZero(m); got != 36 {
		t.Fatalf("filled rect count=%d, want 36", got)
	}
}

func TestFilledRectangleClips(t *testing.T) {
	m := cv.NewMat(10, 10, 1)
	FilledRectangle(m, cv.Point{X: -5, Y: -5}, cv.Point{X: 4, Y: 4}, white())
	// clipped to (0,0)-(4,4) = 25 pixels
	if got := countNonZero(m); got != 25 {
		t.Fatalf("clipped rect count=%d, want 25", got)
	}
}

func TestFilledCircleKnownCount(t *testing.T) {
	m := cv.NewMat(11, 11, 1)
	FilledCircle(m, cv.Point{X: 5, Y: 5}, 2, white())
	// pixels with dx^2+dy^2 <= 4: 1+3+5+3+1 = 13
	if got := countNonZero(m); got != 13 {
		t.Fatalf("filled circle r=2 count=%d, want 13", got)
	}
	if px(m, 5, 5, 0) != 255 {
		t.Fatal("circle centre not set")
	}
	if px(m, 5, 8, 0) != 0 {
		t.Fatal("pixel outside radius unexpectedly set")
	}
}

func TestCircleSymmetry(t *testing.T) {
	m := cv.NewMat(21, 21, 1)
	Circle(m, cv.Point{X: 10, Y: 10}, 5, white(), 1)
	// midpoint circle must touch the four axis extremes
	for _, p := range []cv.Point{{X: 15, Y: 10}, {X: 5, Y: 10}, {X: 10, Y: 15}, {X: 10, Y: 5}} {
		if px(m, p.X, p.Y, 0) != 255 {
			t.Fatalf("circle extreme (%d,%d) not set", p.X, p.Y)
		}
	}
	if px(m, 10, 10, 0) != 0 {
		t.Fatal("circle outline should not fill the centre")
	}
}

func TestWuLineOpaqueInterior(t *testing.T) {
	m := cv.NewMat(10, 10, 1)
	WuLine(m, 1, 5, 7, 5, white())
	// horizontal integer line: interior pixels get full coverage
	for x := 2; x <= 6; x++ {
		if got := px(m, x, 5, 0); got != 255 {
			t.Fatalf("Wu interior pixel (%d,5)=%d, want 255", x, got)
		}
	}
}

func TestWuLineBlends(t *testing.T) {
	m := cv.NewMat(10, 10, 1)
	WuLine(m, 0, 0, 6, 3, white())
	// a diagonal must produce at least one partially-covered (anti-aliased) pixel
	partial := false
	for _, v := range m.Data {
		if v > 0 && v < 255 {
			partial = true
			break
		}
	}
	if !partial {
		t.Fatal("Wu line produced no anti-aliased pixels")
	}
}

func TestFillPolygonSquare(t *testing.T) {
	m := cv.NewMat(10, 10, 1)
	poly := []cv.Point{{X: 2, Y: 2}, {X: 6, Y: 2}, {X: 6, Y: 6}, {X: 2, Y: 6}}
	FillPolygon(m, [][]cv.Point{poly}, white())
	if got := countNonZero(m); got != 20 {
		t.Fatalf("filled polygon square count=%d, want 20", got)
	}
	if px(m, 4, 4, 0) != 255 {
		t.Fatal("polygon interior not filled")
	}
	if px(m, 8, 8, 0) != 0 {
		t.Fatal("polygon exterior unexpectedly filled")
	}
}

func TestFillConvexPolygonTriangle(t *testing.T) {
	m := cv.NewMat(12, 12, 1)
	tri := []cv.Point{{X: 1, Y: 1}, {X: 9, Y: 1}, {X: 5, Y: 9}}
	FillConvexPolygon(m, tri, white())
	// apex and interior filled, a point clearly outside is not
	if px(m, 5, 5, 0) != 255 {
		t.Fatal("triangle interior not filled")
	}
	if px(m, 1, 9, 0) != 0 {
		t.Fatal("triangle exterior corner unexpectedly filled")
	}
}

func TestPaletteColorKnown(t *testing.T) {
	cases := []struct {
		p       Palette
		t       float64
		r, g, b uint8
		name    string
	}{
		{PaletteGray, 0, 0, 0, 0, "gray-min"},
		{PaletteGray, 1, 255, 255, 255, "gray-max"},
		{PaletteHot, 0, 0, 0, 0, "hot-min"},
		{PaletteHot, 1, 255, 255, 255, "hot-max"},
		{PaletteCool, 0, 0, 255, 255, "cool-min"},
		{PaletteCool, 1, 255, 0, 255, "cool-max"},
	}
	for _, c := range cases {
		col := PaletteColor(c.p, c.t)
		if draw2clamp8(col[0]) != c.r || draw2clamp8(col[1]) != c.g || draw2clamp8(col[2]) != c.b {
			t.Fatalf("%s: got (%d,%d,%d) want (%d,%d,%d)", c.name,
				draw2clamp8(col[0]), draw2clamp8(col[1]), draw2clamp8(col[2]), c.r, c.g, c.b)
		}
	}
}

func TestColorizeRamp(t *testing.T) {
	src := cv.NewMat(1, 3, 1)
	src.Data[0] = 0
	src.Data[1] = 128
	src.Data[2] = 255
	out := Colorize(src, PaletteGray)
	if out.Channels != 3 {
		t.Fatalf("Colorize channels=%d, want 3", out.Channels)
	}
	want := []uint8{0, 128, 255}
	for i, w := range want {
		if got := out.Data[i*3]; got != w {
			t.Fatalf("colorize pixel %d = %d, want %d", i, got, w)
		}
	}
}

func TestHeatmapNormalization(t *testing.T) {
	values := [][]float64{{0, 5, 10}}
	m := Heatmap(values, PaletteGray)
	if m.Rows != 1 || m.Cols != 3 {
		t.Fatalf("heatmap size %dx%d, want 1x3", m.Rows, m.Cols)
	}
	// min -> black, max -> white, midpoint -> mid-grey
	if m.Data[0] != 0 {
		t.Fatalf("heatmap min = %d, want 0", m.Data[0])
	}
	if m.Data[6] != 255 {
		t.Fatalf("heatmap max = %d, want 255", m.Data[6])
	}
	if mid := m.Data[3]; mid < 126 || mid > 129 {
		t.Fatalf("heatmap midpoint = %d, want ~128", mid)
	}
}

func TestAlphaBlendMidpoint(t *testing.T) {
	dst := cv.NewMat(4, 4, 3)
	src := cv.NewMat(4, 4, 3)
	src.SetTo(200)
	AlphaBlend(dst, src, 0.5)
	for _, v := range dst.Data {
		if v != 100 {
			t.Fatalf("alpha blend sample=%d, want 100", v)
		}
	}
}

func TestBlendAddWeighted(t *testing.T) {
	a := cv.NewMat(2, 2, 1)
	b := cv.NewMat(2, 2, 1)
	a.SetTo(100)
	b.SetTo(200)
	out := Blend(a, b, 0.5, 0.5, 0)
	for _, v := range out.Data {
		if v != 150 {
			t.Fatalf("blend sample=%d, want 150", v)
		}
	}
}

func TestAlphaCompositeMask(t *testing.T) {
	dst := cv.NewMat(1, 2, 1)
	src := cv.NewMat(1, 2, 1)
	src.SetTo(255)
	mask := cv.NewMat(1, 2, 1)
	mask.Data[0] = 0
	mask.Data[1] = 255
	AlphaCompositeMask(dst, src, mask)
	if dst.Data[0] != 0 {
		t.Fatalf("masked-off pixel changed to %d", dst.Data[0])
	}
	if dst.Data[1] != 255 {
		t.Fatalf("masked-on pixel = %d, want 255", dst.Data[1])
	}
}

func TestFillRectangleAlpha(t *testing.T) {
	m := cv.NewMat(4, 4, 1)
	m.SetTo(100)
	FillRectangleAlpha(m, cv.Point{X: 0, Y: 0}, cv.Point{X: 3, Y: 3}, cv.Scalar{200, 200, 200, 255}, 0.5)
	// 100*0.5 + 200*0.5 = 150
	for _, v := range m.Data {
		if v != 150 {
			t.Fatalf("alpha fill sample=%d, want 150", v)
		}
	}
}

func TestTextSize(t *testing.T) {
	w, h := TextSize("AB", 1)
	if w != 11 || h != 7 {
		t.Fatalf("TextSize(AB,1)=(%d,%d), want (11,7)", w, h)
	}
	w2, h2 := TextSize("AB", 2)
	if w2 != 22 || h2 != 14 {
		t.Fatalf("TextSize(AB,2)=(%d,%d), want (22,14)", w2, h2)
	}
	if w0, h0 := TextSize("", 3); w0 != 0 || h0 != 0 {
		t.Fatalf("TextSize empty=(%d,%d), want (0,0)", w0, h0)
	}
}

func TestPutTextRenders(t *testing.T) {
	m := cv.NewMat(12, 40, 1)
	PutText(m, "HI", cv.Point{X: 1, Y: 9}, 1, white())
	if countNonZero(m) == 0 {
		t.Fatal("PutText rendered nothing")
	}
}

func TestBezierEndpoints(t *testing.T) {
	p0 := PointF{1, 2}
	p1 := PointF{3, 8}
	p2 := PointF{5, 1}
	p3 := PointF{9, 4}
	if got := CubicBezierPoint(p0, p1, p2, p3, 0); got != p0 {
		t.Fatalf("cubic t=0 = %+v, want %+v", got, p0)
	}
	if got := CubicBezierPoint(p0, p1, p2, p3, 1); got != p3 {
		t.Fatalf("cubic t=1 = %+v, want %+v", got, p3)
	}
	if got := QuadBezierPoint(p0, p1, p2, 0); got != p0 {
		t.Fatalf("quad t=0 = %+v, want %+v", got, p0)
	}
	if got := QuadBezierPoint(p0, p1, p2, 1); got != p2 {
		t.Fatalf("quad t=1 = %+v, want %+v", got, p2)
	}
	mid := QuadBezierPoint(PointF{0, 0}, PointF{2, 4}, PointF{4, 0}, 0.5)
	if mid.X != 2 || mid.Y != 2 {
		t.Fatalf("quad midpoint = %+v, want {2,2}", mid)
	}
}

func TestCatmullRomThroughPoints(t *testing.T) {
	// A Catmull-Rom spline interpolates its control points, so the exact
	// segment endpoint p1..p2 evaluation must return those points.
	if got := draw2catmullRom(PointF{0, 0}, PointF{1, 1}, PointF{2, 4}, PointF{3, 9}, 0); got != (PointF{1, 1}) {
		t.Fatalf("catmull t=0 = %+v, want {1,1}", got)
	}
	if got := draw2catmullRom(PointF{0, 0}, PointF{1, 1}, PointF{2, 4}, PointF{3, 9}, 1); got != (PointF{2, 4}) {
		t.Fatalf("catmull t=1 = %+v, want {2,4}", got)
	}
}

func TestPolygonArea(t *testing.T) {
	sq := []cv.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 4}, {X: 0, Y: 4}}
	if a := PolygonArea(sq); a != 16 {
		t.Fatalf("square area = %v, want 16", a)
	}
}

func TestPolygonPerimeter(t *testing.T) {
	sq := []cv.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 4}, {X: 0, Y: 4}}
	if p := PolygonPerimeter(sq, true); p != 16 {
		t.Fatalf("square perimeter (closed) = %v, want 16", p)
	}
	if p := PolygonPerimeter(sq, false); p != 12 {
		t.Fatalf("square perimeter (open) = %v, want 12", p)
	}
}

func TestBoundingRectAndCentroid(t *testing.T) {
	pts := []cv.Point{{X: 2, Y: 3}, {X: 8, Y: 1}, {X: 5, Y: 9}}
	tl, br := BoundingRect(pts)
	if tl != (cv.Point{X: 2, Y: 1}) || br != (cv.Point{X: 8, Y: 9}) {
		t.Fatalf("bounding rect = %+v..%+v, want {2,1}..{8,9}", tl, br)
	}
	c := Centroid([]cv.Point{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 0, Y: 6}, {X: 4, Y: 6}})
	if c != (cv.Point{X: 2, Y: 3}) {
		t.Fatalf("centroid = %+v, want {2,3}", c)
	}
}

func TestDrawContoursFilled(t *testing.T) {
	m := cv.NewMat(10, 10, 1)
	c := []cv.Point{{X: 2, Y: 2}, {X: 6, Y: 2}, {X: 6, Y: 6}, {X: 2, Y: 6}}
	DrawContours(m, [][]cv.Point{c}, -1, white(), Filled)
	if countNonZero(m) != 20 {
		t.Fatalf("filled contour count=%d, want 20", countNonZero(m))
	}
}

func TestOverlayContoursBlends(t *testing.T) {
	m := cv.NewMat(10, 10, 1)
	m.SetTo(100)
	c := []cv.Point{{X: 2, Y: 2}, {X: 6, Y: 2}, {X: 6, Y: 6}, {X: 2, Y: 6}}
	OverlayContours(m, [][]cv.Point{c}, cv.Scalar{200, 200, 200, 255}, 0.5)
	// inside -> 150, outside -> unchanged 100
	if px(m, 4, 4, 0) != 150 {
		t.Fatalf("overlay inside = %d, want 150", px(m, 4, 4, 0))
	}
	if px(m, 8, 8, 0) != 100 {
		t.Fatalf("overlay outside = %d, want 100", px(m, 8, 8, 0))
	}
}

func TestColorBar(t *testing.T) {
	bar := ColorBar(4, 8, PaletteGray)
	if bar.Rows != 8 || bar.Cols != 4 {
		t.Fatalf("colorbar size %dx%d, want 8x4", bar.Rows, bar.Cols)
	}
	// top row is the maximum (white), bottom row the minimum (black)
	if bar.Data[0] != 255 {
		t.Fatalf("colorbar top = %d, want 255", bar.Data[0])
	}
	if bar.Data[(7*4)*3] != 0 {
		t.Fatalf("colorbar bottom = %d, want 0", bar.Data[(7*4)*3])
	}
}

func TestLerpColor(t *testing.T) {
	a := cv.Scalar{0, 0, 0, 0}
	b := cv.Scalar{100, 200, 40, 255}
	got := LerpColor(a, b, 0.5)
	want := cv.Scalar{50, 100, 20, 127.5}
	if got != want {
		t.Fatalf("lerp = %+v, want %+v", got, want)
	}
}

func BenchmarkColorizeHeatmap(b *testing.B) {
	// Heaviest routine: per-pixel palette evaluation over a large image.
	src := cv.NewMat(512, 512, 1)
	for i := range src.Data {
		src.Data[i] = uint8(i)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Colorize(src, PaletteJet)
	}
}
