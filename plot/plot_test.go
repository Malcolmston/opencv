package plot

import (
	"testing"

	cv "github.com/malcolmston/opencv"
)

// isWhite reports whether pixel (y,x) is the default white background.
func isWhite(m *cv.Mat, y, x int) bool {
	i := (y*m.Cols + x) * 3
	return m.Data[i] == 255 && m.Data[i+1] == 255 && m.Data[i+2] == 255
}

// countNonWhite counts non-background pixels in row y over columns [x0,x1].
func countNonWhiteRow(m *cv.Mat, y, x0, x1 int) int {
	n := 0
	for x := x0; x <= x1; x++ {
		if !isWhite(m, y, x) {
			n++
		}
	}
	return n
}

// countNonWhiteCol counts non-background pixels in column x over rows [y0,y1].
func countNonWhiteCol(m *cv.Mat, x, y0, y1 int) int {
	n := 0
	for y := y0; y <= y1; y++ {
		if !isWhite(m, y, x) {
			n++
		}
	}
	return n
}

func TestRenderSizeAndChannels(t *testing.T) {
	img := CreatePlotY([]float64{1, 2, 3}).SetSize(320, 240).Render()
	if img.Rows != 240 || img.Cols != 320 {
		t.Fatalf("size = %dx%d, want 240x320", img.Rows, img.Cols)
	}
	if img.Channels != 3 {
		t.Fatalf("channels = %d, want 3", img.Channels)
	}
}

func TestHorizontalLinePaintsExpectedRow(t *testing.T) {
	// A constant series with a pinned y range must paint one predictable row.
	p := CreatePlotY([]float64{5, 5, 5, 5, 5}).
		SetSize(200, 100).
		SetShowGrid(false).
		SetRangeY(0, 10)
	img := p.Render()

	x0, y0, x1, y1 := p.plotArea()
	if x0 != 50 || y0 != 20 || x1 != 179 || y1 != 59 {
		t.Fatalf("unexpected plot area %d,%d,%d,%d", x0, y0, x1, y1)
	}
	row := p.mapY(5) // expected row for y=5 within [0,10]
	if row != 39 {
		t.Fatalf("mapY(5) = %d, want 39", row)
	}

	// The plotted row is heavily painted across the plot width (excluding the
	// axis column at x0).
	if got := countNonWhiteRow(img, row, x0+1, x1); got < (x1-x0)-5 {
		t.Errorf("row %d has %d painted px, want ~%d", row, got, x1-x0)
	}
	// Rows just above/below carry no line ink (thickness 1).
	for _, r := range []int{row - 5, row + 5} {
		if got := countNonWhiteRow(img, r, x0+1, x1); got != 0 {
			t.Errorf("row %d has %d painted px, want 0", r, got)
		}
	}
}

func TestScatterPaintsMarkersAtPoints(t *testing.T) {
	xs := []float64{0, 1, 2}
	ys := []float64{0, 1, 2}
	p := ScatterPlot(xs, ys).
		SetSize(200, 200).
		SetShowGrid(false).
		SetPointRadius(4).
		SetRangeX(0, 2).
		SetRangeY(0, 2)
	img := p.Render()
	for i := range xs {
		cx, cy := p.mapX(xs[i]), p.mapY(ys[i])
		if isWhite(img, cy, cx) {
			t.Errorf("point %d at (%d,%d) not painted", i, cx, cy)
		}
	}
	// A location between two markers (well off any marker) stays background.
	if !isWhite(img, p.mapY(0.5)-0, p.mapX(0.5)) {
		// midpoint of the first gap should be empty in a scatter plot
		t.Errorf("unexpected ink at scatter gap")
	}
}

func TestBarHeightsProportionalToCounts(t *testing.T) {
	// Data engineered so bins [0,1),[1,2),[2,3),[3,4) hold 1,2,3,4 samples.
	data := []float64{
		0.0,
		1.1, 1.2,
		2.1, 2.2, 2.3,
		3.1, 3.2, 3.3, 4.0,
	}
	p := HistogramPlot(data, 4).
		SetSize(400, 400).
		SetShowGrid(false)
	img := p.Render()

	_, y0, _, y1 := p.plotArea()
	counts := []float64{1, 2, 3, 4}
	centres := []float64{0.5, 1.5, 2.5, 3.5}
	heights := make([]int, 4)
	for i := range centres {
		cx := p.mapX(centres[i])
		heights[i] = countNonWhiteCol(img, cx, y0, y1)
	}

	// Strictly increasing heights.
	for i := 1; i < 4; i++ {
		if heights[i] <= heights[i-1] {
			t.Fatalf("bar heights not increasing: %v", heights)
		}
	}
	// Proportional to counts within tolerance (relative to the tallest bar).
	for i := 0; i < 4; i++ {
		ratio := float64(heights[i]) / float64(heights[3])
		want := counts[i] / counts[3]
		if d := ratio - want; d > 0.06 || d < -0.06 {
			t.Errorf("bar %d ratio = %.3f, want ~%.3f (heights %v)", i, ratio, want, heights)
		}
	}
}

func TestRenderIsDeterministic(t *testing.T) {
	build := func() *cv.Mat {
		return LinePlot([]float64{0, 1, 4, 9, 16}, []float64{1, 3, 2, 5, 4}).
			SetSize(256, 256).Render()
	}
	a, b := build(), build()
	if len(a.Data) != len(b.Data) {
		t.Fatal("length mismatch")
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatalf("non-deterministic render at byte %d", i)
		}
	}
}

func TestCreatePlotPanicsOnLengthMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected panic")
		}
	}()
	CreatePlot([]float64{1, 2}, []float64{1})
}

func TestBackgroundIsConfigurable(t *testing.T) {
	img := CreatePlotY([]float64{1, 2, 3}).
		SetSize(40, 40).
		SetShowGrid(false).
		SetBackgroundColor(cv.NewScalar(10, 20, 30)).
		Render()
	// A corner pixel is background, untouched by axes or data.
	i := (0*img.Cols + img.Cols - 1) * 3
	if img.Data[i] != 10 || img.Data[i+1] != 20 || img.Data[i+2] != 30 {
		t.Errorf("background = %d,%d,%d, want 10,20,30", img.Data[i], img.Data[i+1], img.Data[i+2])
	}
}
