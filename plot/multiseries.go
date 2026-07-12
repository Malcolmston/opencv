package plot

import (
	cv "github.com/malcolmston/opencv"
)

// Series is one named data series drawn by [MultiSeriesPlot]. It carries its own
// styling so several series can be overlaid on shared axes and distinguished in
// the legend.
type Series struct {
	// Name labels the series in the legend.
	Name string
	// X and Y are the series data; they must have equal length.
	X, Y []float64
	// Kind selects line or scatter rendering (bars are not supported in overlay).
	Kind PlotKind
	// Color is the series colour, interpreted as RGB.
	Color cv.Scalar
	// Thickness is the line width; Radius is the scatter marker radius.
	Thickness, Radius int
}

// MultiSeriesPlot overlays several [Series] on one set of axes with a shared,
// auto-fitted coordinate range and an optional legend. Construct one with
// [NewMultiSeriesPlot].
type MultiSeriesPlot struct {
	Series []Series

	Width, Height                    int
	MarginLeft, MarginRight          int
	MarginTop, MarginBottom          int
	Background, AxisColor, GridColor cv.Scalar
	TextColor                        cv.Scalar
	TextScale                        int
	ShowGrid                         bool
	GridStepsX, GridStepsY           int
	ShowLegend                       bool

	autoX, autoY           bool
	minX, maxX, minY, maxY float64
}

// NewMultiSeriesPlot builds an empty [MultiSeriesPlot] with default styling. Add
// series with [MultiSeriesPlot.Add].
func NewMultiSeriesPlot() *MultiSeriesPlot {
	return &MultiSeriesPlot{
		Width: 640, Height: 480,
		MarginLeft: 50, MarginRight: 20, MarginTop: 20, MarginBottom: 40,
		Background: cv.NewScalar(255, 255, 255),
		AxisColor:  cv.NewScalar(0, 0, 0),
		GridColor:  cv.NewScalar(200, 200, 200),
		TextColor:  cv.NewScalar(0, 0, 0),
		TextScale:  1,
		ShowGrid:   true, GridStepsX: 10, GridStepsY: 10,
		ShowLegend: true,
		autoX:      true, autoY: true,
	}
}

// Add appends a series with the given name, data and kind, assigning it the next
// colour from the default palette. It panics if x and y differ in length or are
// empty. Returns the plot for chaining.
func (p *MultiSeriesPlot) Add(name string, x, y []float64, kind PlotKind) *MultiSeriesPlot {
	if len(x) != len(y) {
		panic("plot: MultiSeriesPlot.Add requires len(x) == len(y)")
	}
	if len(x) == 0 {
		panic("plot: MultiSeriesPlot.Add requires at least one point")
	}
	p.Series = append(p.Series, Series{
		Name:      name,
		X:         append([]float64(nil), x...),
		Y:         append([]float64(nil), y...),
		Kind:      kind,
		Color:     defaultWedgeColors[len(p.Series)%len(defaultWedgeColors)],
		Thickness: 1, Radius: 3,
	})
	return p
}

// AddSeries appends a fully specified [Series] (its Color and sizes are used
// as-is) and returns the plot for chaining. It panics if the series data is
// empty or mismatched.
func (p *MultiSeriesPlot) AddSeries(s Series) *MultiSeriesPlot {
	if len(s.X) != len(s.Y) {
		panic("plot: AddSeries requires len(X) == len(Y)")
	}
	if len(s.X) == 0 {
		panic("plot: AddSeries requires at least one point")
	}
	cp := s
	cp.X = append([]float64(nil), s.X...)
	cp.Y = append([]float64(nil), s.Y...)
	if cp.Thickness < 1 {
		cp.Thickness = 1
	}
	if cp.Radius < 1 {
		cp.Radius = 3
	}
	p.Series = append(p.Series, cp)
	return p
}

// SetSize sets the canvas dimensions and returns the plot for chaining.
func (p *MultiSeriesPlot) SetSize(w, h int) *MultiSeriesPlot { p.Width, p.Height = w, h; return p }

// SetShowLegend toggles the legend and returns the plot for chaining.
func (p *MultiSeriesPlot) SetShowLegend(on bool) *MultiSeriesPlot { p.ShowLegend = on; return p }

// SetShowGrid toggles the grid and returns the plot for chaining.
func (p *MultiSeriesPlot) SetShowGrid(on bool) *MultiSeriesPlot { p.ShowGrid = on; return p }

// SetRangeX pins the x axis bounds and returns the plot for chaining. It panics
// if min >= max.
func (p *MultiSeriesPlot) SetRangeX(min, max float64) *MultiSeriesPlot {
	if min >= max {
		panic("plot: SetRangeX requires min < max")
	}
	p.minX, p.maxX, p.autoX = min, max, false
	return p
}

// SetRangeY pins the y axis bounds and returns the plot for chaining. It panics
// if min >= max.
func (p *MultiSeriesPlot) SetRangeY(min, max float64) *MultiSeriesPlot {
	if min >= max {
		panic("plot: SetRangeY requires min < max")
	}
	p.minY, p.maxY, p.autoY = min, max, false
	return p
}

// dataRange returns the axis bounds, fitting them to all series when auto.
func (p *MultiSeriesPlot) dataRange() (minX, maxX, minY, maxY float64) {
	minX, maxX, minY, maxY = p.minX, p.maxX, p.minY, p.maxY
	first := true
	for _, s := range p.Series {
		xl, xh := minMax(s.X)
		yl, yh := minMax(s.Y)
		if first {
			minX, maxX, minY, maxY = xl, xh, yl, yh
			first = false
			continue
		}
		if xl < minX {
			minX = xl
		}
		if xh > maxX {
			maxX = xh
		}
		if yl < minY {
			minY = yl
		}
		if yh > maxY {
			maxY = yh
		}
	}
	if !p.autoX {
		minX, maxX = p.minX, p.maxX
	}
	if !p.autoY {
		minY, maxY = p.minY, p.maxY
	}
	return minX, maxX, minY, maxY
}

// Render draws every series onto a fresh three-channel RGB [cv.Mat] of size
// Height×Width and returns it. It panics if no series have been added.
func (p *MultiSeriesPlot) Render() *cv.Mat {
	if p.Width <= 0 || p.Height <= 0 {
		panic("plot: Render requires positive canvas dimensions")
	}
	if len(p.Series) == 0 {
		panic("plot: MultiSeriesPlot.Render requires at least one series")
	}
	canvas := cv.NewMat(p.Height, p.Width, 3)
	fillBackground(canvas, p.Background)
	minX, maxX, minY, maxY := p.dataRange()
	a := newAxes(p.Width, p.Height, p.MarginLeft, p.MarginRight, p.MarginTop, p.MarginBottom, minX, maxX, minY, maxY)
	x0, y0, x1, y1 := a.area()
	if p.ShowGrid {
		drawGridLines(canvas, x0, y0, x1, y1, p.GridStepsX, p.GridStepsY, p.GridColor)
	}
	drawFrameAxes(canvas, x0, y0, x1, y1, p.AxisColor)

	for _, s := range p.Series {
		switch s.Kind {
		case KindScatter:
			for i := range s.X {
				cv.Circle(canvas, cv.Point{X: a.x(s.X[i]), Y: a.y(s.Y[i])}, s.Radius, s.Color, cv.Filled)
			}
		default:
			pts := make([]cv.Point, len(s.X))
			for i := range s.X {
				pts[i] = cv.Point{X: a.x(s.X[i]), Y: a.y(s.Y[i])}
			}
			if len(pts) == 1 {
				cv.Circle(canvas, pts[0], s.Thickness, s.Color, cv.Filled)
			} else {
				cv.Polylines(canvas, [][]cv.Point{pts}, false, s.Color, s.Thickness)
			}
		}
	}

	if p.ShowLegend {
		entries := make([]LegendEntry, len(p.Series))
		for i, s := range p.Series {
			entries[i] = LegendEntry{Label: s.Name, Color: s.Color}
		}
		DrawLegend(canvas, entries, x0+6, y0+6, p.TextScale, p.TextColor, p.Background)
	}
	return canvas
}
