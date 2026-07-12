package plot

import (
	cv "github.com/malcolmston/opencv"
)

// seriesPlot is the shared configuration for the single-series line-family charts
// [StemPlot], [StepPlot] and [AreaPlot]. It carries the data, canvas geometry and
// styling; each concrete type embeds it and supplies its own Render.
type seriesPlot struct {
	X []float64
	Y []float64

	Width, Height                    int
	MarginLeft, MarginRight          int
	MarginTop, MarginBottom          int
	Background, AxisColor, GridColor cv.Scalar
	LineColor, PointColor            cv.Scalar
	LineThickness, PointRadius       int
	ShowGrid                         bool
	GridStepsX, GridStepsY           int
	autoX, autoY                     bool
	minX, maxX, minY, maxY           float64
}

func newSeriesPlot(x, y []float64) seriesPlot {
	return seriesPlot{
		X: append([]float64(nil), x...), Y: append([]float64(nil), y...),
		Width: 640, Height: 480,
		MarginLeft: 50, MarginRight: 20, MarginTop: 20, MarginBottom: 40,
		Background:    cv.NewScalar(255, 255, 255),
		AxisColor:     cv.NewScalar(0, 0, 0),
		GridColor:     cv.NewScalar(200, 200, 200),
		LineColor:     cv.NewScalar(0, 0, 255),
		PointColor:    cv.NewScalar(255, 0, 0),
		LineThickness: 1, PointRadius: 3,
		ShowGrid: true, GridStepsX: 10, GridStepsY: 10,
		autoX: true, autoY: true,
	}
}

// resolveAxes builds the coordinate mapper, computing data ranges when they have
// not been pinned. Bars/areas include the zero baseline in y.
func (s *seriesPlot) resolveAxes(includeZeroBaseline bool) *axes {
	minX, maxX := s.minX, s.maxX
	minY, maxY := s.minY, s.maxY
	if s.autoX {
		minX, maxX = minMax(s.X)
	}
	if s.autoY {
		minY, maxY = minMax(s.Y)
		if includeZeroBaseline {
			if minY > 0 {
				minY = 0
			}
			if maxY < 0 {
				maxY = 0
			}
		}
	}
	return newAxes(s.Width, s.Height, s.MarginLeft, s.MarginRight, s.MarginTop, s.MarginBottom, minX, maxX, minY, maxY)
}

// drawFrame paints the background, optional grid and the two axis lines and
// returns the fresh canvas together with its coordinate mapper.
func (s *seriesPlot) drawFrame(includeZeroBaseline bool) (*cv.Mat, *axes) {
	if s.Width <= 0 || s.Height <= 0 {
		panic("plot: Render requires positive canvas dimensions")
	}
	canvas := cv.NewMat(s.Height, s.Width, 3)
	fillBackground(canvas, s.Background)
	a := s.resolveAxes(includeZeroBaseline)
	x0, y0, x1, y1 := a.area()
	if s.ShowGrid {
		drawGridLines(canvas, x0, y0, x1, y1, s.GridStepsX, s.GridStepsY, s.GridColor)
	}
	drawFrameAxes(canvas, x0, y0, x1, y1, s.AxisColor)
	return canvas, a
}

// baselineRow returns the pixel row of the y baseline (zero when it lies inside
// the range, otherwise the nearer axis bound), clamped to the plot area.
func (s *seriesPlot) baselineRow(a *axes) int {
	base := clampBaseline(a.minY, a.maxY)
	_, _, _, y1 := a.area()
	row := a.y(base)
	if row > y1 {
		row = y1
	}
	return row
}

// StemPlot draws a stem (lollipop) chart: a vertical stem rises from the y
// baseline to each sample and a filled circular marker caps it. Construct one
// with [NewStemPlot].
type StemPlot struct{ seriesPlot }

// NewStemPlot builds a [StemPlot] from explicit x and y series with default
// styling. It panics if the slices differ in length or are empty.
func NewStemPlot(x, y []float64) *StemPlot {
	if len(x) != len(y) {
		panic("plot: NewStemPlot requires len(x) == len(y)")
	}
	if len(y) == 0 {
		panic("plot: NewStemPlot requires at least one point")
	}
	return &StemPlot{newSeriesPlot(x, y)}
}

// SetSize sets the canvas dimensions and returns the plot for chaining.
func (p *StemPlot) SetSize(w, h int) *StemPlot { p.Width, p.Height = w, h; return p }

// SetShowGrid toggles the grid and returns the plot for chaining.
func (p *StemPlot) SetShowGrid(on bool) *StemPlot { p.ShowGrid = on; return p }

// SetLineColor sets the stem colour and returns the plot for chaining.
func (p *StemPlot) SetLineColor(c cv.Scalar) *StemPlot { p.LineColor = c; return p }

// SetRangeY pins the y axis bounds and returns the plot for chaining. It panics
// if min >= max.
func (p *StemPlot) SetRangeY(min, max float64) *StemPlot {
	if min >= max {
		panic("plot: SetRangeY requires min < max")
	}
	p.minY, p.maxY, p.autoY = min, max, false
	return p
}

// Render draws the stem chart onto a fresh three-channel RGB [cv.Mat] of size
// Height×Width and returns it.
func (p *StemPlot) Render() *cv.Mat {
	canvas, a := p.drawFrame(true)
	base := p.baselineRow(a)
	for i := range p.X {
		cx := a.x(p.X[i])
		top := a.y(p.Y[i])
		cv.Line(canvas, cv.Point{X: cx, Y: base}, cv.Point{X: cx, Y: top}, p.LineColor, p.LineThickness)
		cv.Circle(canvas, cv.Point{X: cx, Y: top}, p.PointRadius, p.PointColor, cv.Filled)
	}
	return canvas
}

// StepPlot draws a step (staircase) chart: successive samples are connected by a
// horizontal segment held at the earlier value followed by a vertical segment to
// the next value. Construct one with [NewStepPlot].
type StepPlot struct{ seriesPlot }

// NewStepPlot builds a [StepPlot] from explicit x and y series with default
// styling. It panics if the slices differ in length or are empty.
func NewStepPlot(x, y []float64) *StepPlot {
	if len(x) != len(y) {
		panic("plot: NewStepPlot requires len(x) == len(y)")
	}
	if len(y) == 0 {
		panic("plot: NewStepPlot requires at least one point")
	}
	return &StepPlot{newSeriesPlot(x, y)}
}

// SetSize sets the canvas dimensions and returns the plot for chaining.
func (p *StepPlot) SetSize(w, h int) *StepPlot { p.Width, p.Height = w, h; return p }

// SetShowGrid toggles the grid and returns the plot for chaining.
func (p *StepPlot) SetShowGrid(on bool) *StepPlot { p.ShowGrid = on; return p }

// SetLineColor sets the step-line colour and returns the plot for chaining.
func (p *StepPlot) SetLineColor(c cv.Scalar) *StepPlot { p.LineColor = c; return p }

// SetLineThickness sets the step-line width in pixels (minimum 1) and returns
// the plot for chaining.
func (p *StepPlot) SetLineThickness(t int) *StepPlot {
	if t < 1 {
		t = 1
	}
	p.LineThickness = t
	return p
}

// Render draws the step chart onto a fresh three-channel RGB [cv.Mat] of size
// Height×Width and returns it.
func (p *StepPlot) Render() *cv.Mat {
	canvas, a := p.drawFrame(false)
	prev := cv.Point{X: a.x(p.X[0]), Y: a.y(p.Y[0])}
	for i := 1; i < len(p.X); i++ {
		cur := cv.Point{X: a.x(p.X[i]), Y: a.y(p.Y[i])}
		mid := cv.Point{X: cur.X, Y: prev.Y}
		cv.Line(canvas, prev, mid, p.LineColor, p.LineThickness)
		cv.Line(canvas, mid, cur, p.LineColor, p.LineThickness)
		prev = cur
	}
	if len(p.X) == 1 {
		cv.Circle(canvas, prev, p.LineThickness, p.LineColor, cv.Filled)
	}
	return canvas
}

// AreaPlot draws a filled area chart: the region between the data curve and the
// y baseline is flooded with FillColor and the curve itself is stroked in
// LineColor. Construct one with [NewAreaPlot].
type AreaPlot struct {
	seriesPlot
	// FillColor is the area fill, interpreted as RGB.
	FillColor cv.Scalar
}

// NewAreaPlot builds an [AreaPlot] from explicit x and y series with default
// styling. It panics if the slices differ in length or are empty.
func NewAreaPlot(x, y []float64) *AreaPlot {
	if len(x) != len(y) {
		panic("plot: NewAreaPlot requires len(x) == len(y)")
	}
	if len(y) == 0 {
		panic("plot: NewAreaPlot requires at least one point")
	}
	return &AreaPlot{seriesPlot: newSeriesPlot(x, y), FillColor: cv.NewScalar(150, 190, 255)}
}

// SetSize sets the canvas dimensions and returns the plot for chaining.
func (p *AreaPlot) SetSize(w, h int) *AreaPlot { p.Width, p.Height = w, h; return p }

// SetShowGrid toggles the grid and returns the plot for chaining.
func (p *AreaPlot) SetShowGrid(on bool) *AreaPlot { p.ShowGrid = on; return p }

// SetFillColor sets the area fill colour and returns the plot for chaining.
func (p *AreaPlot) SetFillColor(c cv.Scalar) *AreaPlot { p.FillColor = c; return p }

// SetLineColor sets the outline colour and returns the plot for chaining.
func (p *AreaPlot) SetLineColor(c cv.Scalar) *AreaPlot { p.LineColor = c; return p }

// Render draws the area chart onto a fresh three-channel RGB [cv.Mat] of size
// Height×Width and returns it.
func (p *AreaPlot) Render() *cv.Mat {
	canvas, a := p.drawFrame(true)
	base := p.baselineRow(a)
	// Build a closed polygon: the curve, then back along the baseline.
	poly := make([]cv.Point, 0, len(p.X)+2)
	for i := range p.X {
		poly = append(poly, cv.Point{X: a.x(p.X[i]), Y: a.y(p.Y[i])})
	}
	poly = append(poly,
		cv.Point{X: a.x(p.X[len(p.X)-1]), Y: base},
		cv.Point{X: a.x(p.X[0]), Y: base},
	)
	cv.FillPoly(canvas, [][]cv.Point{poly}, p.FillColor)
	// Stroke the curve on top.
	line := make([]cv.Point, len(p.X))
	for i := range p.X {
		line[i] = cv.Point{X: a.x(p.X[i]), Y: a.y(p.Y[i])}
	}
	if len(line) > 1 {
		cv.Polylines(canvas, [][]cv.Point{line}, false, p.LineColor, p.LineThickness)
	}
	return canvas
}
