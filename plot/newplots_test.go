package plot

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// nonWhiteCount counts pixels differing from the white background.
func nonWhiteCount(m *cv.Mat) int {
	n := 0
	for y := 0; y < m.Rows; y++ {
		for x := 0; x < m.Cols; x++ {
			if !isWhite(m, y, x) {
				n++
			}
		}
	}
	return n
}

// hasColor reports whether any pixel exactly equals the given RGB triple.
func hasColor(m *cv.Mat, r, g, b uint8) bool {
	for i := 0; i+2 < len(m.Data); i += 3 {
		if m.Data[i] == r && m.Data[i+1] == g && m.Data[i+2] == b {
			return true
		}
	}
	return false
}

func TestStemPlotMarkersAndStems(t *testing.T) {
	p := NewStemPlot([]float64{0, 1, 2}, []float64{1, 2, 3}).
		SetSize(200, 200).SetShowGrid(false).SetRangeY(0, 4)
	img := p.Render()
	if img.Rows != 200 || img.Cols != 200 || img.Channels != 3 {
		t.Fatalf("size = %dx%dx%d", img.Rows, img.Cols, img.Channels)
	}
	a := p.resolveAxes(true)
	base := p.baselineRow(a)
	for i, x := range []float64{0, 1, 2} {
		cx := a.x(x)
		top := a.y([]float64{1, 2, 3}[i])
		if isWhite(img, top, cx) {
			t.Errorf("marker %d at (%d,%d) not painted", i, cx, top)
		}
		mid := (top + base) / 2
		if isWhite(img, mid, cx) {
			t.Errorf("stem %d not painted at row %d col %d", i, mid, cx)
		}
	}
}

func TestStepPlotHorizontalTread(t *testing.T) {
	xs := []float64{0, 1, 2}
	ys := []float64{1, 3, 2}
	p := NewStepPlot(xs, ys).SetSize(300, 200).SetShowGrid(false)
	img := p.Render()
	a := p.resolveAxes(false)
	// The tread from x0 to x1 is held at y[0]; sample a column strictly between.
	c0, c1 := a.x(0), a.x(1)
	row := a.y(1)
	col := (c0 + c1) / 2
	if isWhite(img, row, col) {
		t.Errorf("step tread not painted at (%d,%d)", row, col)
	}
}

func TestAreaPlotFillsBelowCurve(t *testing.T) {
	xs := []float64{0, 1, 2, 3}
	ys := []float64{1, 2, 1, 2}
	p := NewAreaPlot(xs, ys).SetSize(300, 300).SetShowGrid(false)
	img := p.Render()
	if nonWhiteCount(img) == 0 {
		t.Fatal("area plot painted nothing")
	}
	a := p.resolveAxes(true)
	base := p.baselineRow(a)
	// Just above the baseline under the first sample must be filled.
	col := a.x(0)
	if isWhite(img, base-3, col) {
		t.Errorf("area not filled just above baseline at (%d,%d)", base-3, col)
	}
	// Top-left canvas corner (in the margin) stays white.
	if !isWhite(img, 0, 0) {
		t.Errorf("canvas corner unexpectedly painted")
	}
}

func TestBoxPlotMedianLine(t *testing.T) {
	groups := [][]float64{{1, 2, 3, 4, 5}, {2, 4, 6, 8, 10}}
	p := NewBoxPlot(groups).SetSize(400, 400).SetShowGrid(false)
	img := p.Render()
	if img.Rows != 400 || img.Cols != 400 {
		t.Fatalf("size = %dx%d", img.Rows, img.Cols)
	}
	lo, hi := p.valueRange()
	a := newAxes(p.Width, p.Height, p.MarginLeft, p.MarginRight, p.MarginTop, p.MarginBottom, 0, float64(len(groups)), lo, hi)
	for i, g := range groups {
		st := summarize(g)
		cx := a.x(float64(i) + 0.5)
		yMed := a.y(st.median)
		if isWhite(img, yMed, cx) {
			t.Errorf("group %d median line not painted at (%d,%d)", i, yMed, cx)
		}
	}
}

func TestViolinPlotSymmetricAndMedian(t *testing.T) {
	groups := [][]float64{{1, 2, 2, 3, 3, 3, 4, 4, 5}}
	p := NewViolinPlot(groups).SetSize(400, 400).SetShowGrid(false)
	img := p.Render()
	if nonWhiteCount(img) == 0 {
		t.Fatal("violin painted nothing")
	}
	lo, hi := p.valueRange()
	a := newAxes(p.Width, p.Height, p.MarginLeft, p.MarginRight, p.MarginTop, p.MarginBottom, 0, 1, lo, hi)
	cx := a.x(0.5)
	med := quantile(sortedCopy(groups[0]), 0.5)
	if isWhite(img, a.y(med), cx) {
		t.Errorf("violin median not painted at center")
	}
}

func TestErrorBarPlotCapsAndPoints(t *testing.T) {
	xs := []float64{0, 1, 2}
	ys := []float64{2, 3, 2}
	errs := []float64{0.5, 1, 0.5}
	p := NewErrorBarPlot(xs, ys, errs).SetSize(300, 300).SetShowGrid(false).SetRangeY(0, 5)
	img := p.Render()
	minX, maxX := minMax(xs)
	a := newAxes(p.Width, p.Height, p.MarginLeft, p.MarginRight, p.MarginTop, p.MarginBottom, minX, maxX, 0, 5)
	for i := range xs {
		cx := a.x(xs[i])
		if isWhite(img, a.y(ys[i]), cx) {
			t.Errorf("point %d not painted", i)
		}
		yTop := a.y(ys[i] + errs[i])
		if isWhite(img, yTop, cx) {
			t.Errorf("upper cap %d not painted at (%d,%d)", i, yTop, cx)
		}
	}
}

func TestHeatmapCellColorsAndSize(t *testing.T) {
	data := [][]float64{{0, 1}, {2, 3}}
	p := NewHeatmapPlot(data).SetCellSize(10).SetColormap(ColormapViridis)
	img := p.Render()
	// 2 rows * 10 = 20 tall; 2 cols*10 + gap(10) + bar(24) = 54 wide.
	if img.Rows != 20 {
		t.Fatalf("rows = %d, want 20", img.Rows)
	}
	if img.Cols != 2*10+10+24 {
		t.Fatalf("cols = %d, want %d", img.Cols, 2*10+10+24)
	}
	table := Table(ColormapViridis)
	// Cell (0,0) has value 0 -> table[0]; cell (1,1) value 3 -> table[255].
	c00 := px3(img, 5, 5)
	if c00 != table[0] {
		t.Errorf("cell(0,0) = %v, want %v", c00, table[0])
	}
	c11 := px3(img, 15, 15)
	if c11 != table[255] {
		t.Errorf("cell(1,1) = %v, want %v", c11, table[255])
	}
}

func TestHeatmapNoColorbar(t *testing.T) {
	data := [][]float64{{0, 1, 2}}
	img := NewHeatmapPlot(data).SetCellSize(5).SetShowColorbar(false).Render()
	if img.Cols != 15 {
		t.Fatalf("cols = %d, want 15", img.Cols)
	}
}

func TestColorbarEndpoints(t *testing.T) {
	cb := NewColorbar(ColormapJet, 20, 100)
	cb.Border = false // border would overwrite the endpoint rows
	img := cb.Render()
	if img.Rows != 100 || img.Cols != 20 {
		t.Fatalf("size = %dx%d", img.Rows, img.Cols)
	}
	table := Table(ColormapJet)
	// Vertical bar: bottom row is intensity 0, top row is intensity 255.
	bottom := px3(img, 99, 10)
	top := px3(img, 0, 10)
	if bottom != table[0] {
		t.Errorf("colorbar bottom = %v, want %v", bottom, table[0])
	}
	if top != table[255] {
		t.Errorf("colorbar top = %v, want %v", top, table[255])
	}
}

func TestContourPlotDrawsLines(t *testing.T) {
	// Radial field: a central bump so mid levels form closed loops.
	const n = 9
	data := make([][]float64, n)
	for r := 0; r < n; r++ {
		data[r] = make([]float64, n)
		for c := 0; c < n; c++ {
			dr := float64(r - n/2)
			dc := float64(c - n/2)
			data[r][c] = -(dr*dr + dc*dc)
		}
	}
	img := NewContourPlot(data).SetSize(200, 200).Render()
	if img.Rows != 200 || img.Cols != 200 {
		t.Fatalf("size = %dx%d", img.Rows, img.Cols)
	}
	if nonWhiteCount(img) == 0 {
		t.Fatal("contour drew no lines")
	}
}

func TestContourDeterministic(t *testing.T) {
	data := [][]float64{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}}
	build := func() *cv.Mat { return NewContourPlot(data).SetSize(120, 120).Render() }
	a, b := build(), build()
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("non-deterministic contour at byte %d", i)
		}
	}
}

func TestPiePlotCenterPainted(t *testing.T) {
	p := NewPiePlot([]float64{1, 2, 3}).SetSize(200, 200).SetLabels([]string{"a", "b", "c"})
	img := p.Render()
	if img.Rows != 200 || img.Cols != 200 {
		t.Fatalf("size = %dx%d", img.Rows, img.Cols)
	}
	// Pie occupies the left region. Measure the fill fraction of a disc around
	// the centre; it must be mostly painted (thin white wedge separators aside).
	legendW := 200 / 4
	pieW := 200 - legendW
	cx, cy := pieW/2, 100
	const rr = 20
	total, painted := 0, 0
	for dy := -rr; dy <= rr; dy++ {
		for dx := -rr; dx <= rr; dx++ {
			if dx*dx+dy*dy > rr*rr {
				continue
			}
			total++
			if !isWhite(img, cy+dy, cx+dx) {
				painted++
			}
		}
	}
	if float64(painted)/float64(total) < 0.6 {
		t.Errorf("pie interior fill fraction = %.2f, want > 0.6", float64(painted)/float64(total))
	}
}

func TestMultiSeriesOverlayColors(t *testing.T) {
	p := NewMultiSeriesPlot().
		Add("linear", []float64{0, 1, 2, 3}, []float64{0, 1, 2, 3}, KindLine).
		Add("square", []float64{0, 1, 2, 3}, []float64{0, 1, 4, 9}, KindScatter).
		SetSize(320, 240).SetShowGrid(false)
	img := p.Render()
	if img.Rows != 240 || img.Cols != 320 {
		t.Fatalf("size = %dx%d", img.Rows, img.Cols)
	}
	c0 := defaultWedgeColors[0]
	c1 := defaultWedgeColors[1]
	if !hasColor(img, uint8(c0[0]), uint8(c0[1]), uint8(c0[2])) {
		t.Error("series 0 colour missing")
	}
	if !hasColor(img, uint8(c1[0]), uint8(c1[1]), uint8(c1[2])) {
		t.Error("series 1 colour missing")
	}
}

func TestMultiSeriesEmptyRenderPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on empty series")
		}
	}()
	NewMultiSeriesPlot().Render()
}

func TestRenderAnnotatedAddsTitleInk(t *testing.T) {
	base := CreatePlotY([]float64{1, 2, 3, 4}).SetSize(320, 240).SetShowGrid(false)
	plain := base.Render()
	x0, y0, _, _ := base.plotArea()

	annotated := CreatePlotY([]float64{1, 2, 3, 4}).SetSize(320, 240).SetShowGrid(false).
		RenderAnnotated(Annotation{Title: "DEMO", XLabel: "X", ShowTicks: true})
	if annotated.Rows != plain.Rows || annotated.Cols != plain.Cols {
		t.Fatalf("annotated size changed: %dx%d", annotated.Rows, annotated.Cols)
	}
	// The title sits in the top margin (rows above y0); count ink there.
	countTop := func(m *cv.Mat) int {
		n := 0
		for y := 0; y < y0; y++ {
			for x := x0; x < m.Cols; x++ {
				if !isWhite(m, y, x) {
					n++
				}
			}
		}
		return n
	}
	if countTop(annotated) <= countTop(plain) {
		t.Errorf("title added no ink in top margin")
	}
}

func TestTextSize(t *testing.T) {
	if w, h := TextSize("Hi", 1); w != 11 || h != 7 {
		t.Errorf("TextSize(Hi,1) = %d,%d, want 11,7", w, h)
	}
	if w, h := TextSize("", 2); w != 0 || h != 14 {
		t.Errorf("TextSize(empty,2) = %d,%d, want 0,14", w, h)
	}
	if w, _ := TextSize("ABC", 2); w != (6*3-1)*2 {
		t.Errorf("TextSize(ABC,2) width = %d, want %d", w, (6*3-1)*2)
	}
}

func TestDrawLegendPaints(t *testing.T) {
	m := cv.NewMat(80, 120, 3)
	fillBackground(m, cv.NewScalar(255, 255, 255))
	DrawLegend(m, []LegendEntry{
		{Label: "red", Color: cv.NewScalar(255, 0, 0)},
		{Label: "green", Color: cv.NewScalar(0, 255, 0)},
	}, 5, 5, 1, cv.NewScalar(0, 0, 0), cv.NewScalar(255, 255, 255))
	if !hasColor(m, 255, 0, 0) {
		t.Error("legend swatch red missing")
	}
	if !hasColor(m, 0, 255, 0) {
		t.Error("legend swatch green missing")
	}
}

func TestContourFillBelowColoursBackground(t *testing.T) {
	data := [][]float64{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}}
	cp := NewContourPlot(data).SetSize(60, 60).SetFillBelow(true)
	cp.Colormap = ColormapViridis
	img := cp.Render()
	// With a colormap tint every pixel is coloured, so none stay white.
	if nonWhiteCount(img) != img.Rows*img.Cols {
		t.Errorf("fill-below left %d white pixels", img.Rows*img.Cols-nonWhiteCount(img))
	}
}

func TestHorizontalColorbarEndpoints(t *testing.T) {
	cb := NewColorbar(ColormapJet, 100, 20).SetHorizontal(true)
	cb.Border = false
	img := cb.Render()
	table := Table(ColormapJet)
	if got := px3(img, 10, 0); got != table[0] {
		t.Errorf("left = %v, want %v", got, table[0])
	}
	if got := px3(img, 10, 99); got != table[255] {
		t.Errorf("right = %v, want %v", got, table[255])
	}
}

func TestMultiSeriesAddSeriesExplicit(t *testing.T) {
	img := NewMultiSeriesPlot().
		AddSeries(Series{Name: "s", X: []float64{0, 1}, Y: []float64{0, 1}, Color: cv.NewScalar(10, 20, 30)}).
		SetSize(120, 120).SetShowLegend(false).Render()
	if !hasColor(img, 10, 20, 30) {
		t.Error("explicit series colour missing")
	}
}

func TestNewPiePlotPanicsOnZeroSum(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic on zero-sum values")
		}
	}()
	NewPiePlot([]float64{0, 0})
}
