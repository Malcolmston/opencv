package plot

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// PlotKind selects how a [Plot2D] renders its data series.
type PlotKind int

const (
	// KindLine connects successive (x,y) samples with straight line segments.
	KindLine PlotKind = iota
	// KindScatter draws an unconnected filled marker at each sample.
	KindScatter
	// KindBar draws a filled vertical bar from the y baseline up to each sample.
	KindBar
)

// Plot2D describes a 2-D chart of one data series and the styling used to draw
// it. Construct it with [CreatePlot], [CreatePlotY] or a kind-specific
// constructor; the exported fields may be set directly or through the chainable
// Set* methods. Rendering is performed by [Plot2D.Render].
type Plot2D struct {
	// X and Y hold the data series; they have equal length.
	X []float64
	Y []float64

	// Kind selects the rendering style.
	Kind PlotKind

	// Width and Height are the canvas size in pixels.
	Width  int
	Height int

	// Margin* inset the plot area from the canvas edges (in pixels), leaving
	// room for axes and labels.
	MarginLeft   int
	MarginRight  int
	MarginTop    int
	MarginBottom int

	// MinX, MaxX, MinY and MaxY are the axis bounds in data coordinates. Each
	// axis is recomputed from the data at render time unless pinned with
	// SetRangeX / SetRangeY.
	MinX, MaxX float64
	MinY, MaxY float64
	autoRangeX bool
	autoRangeY bool

	// ShowGrid toggles the background gridlines; GridStepsX and GridStepsY are
	// the number of grid cells along each axis.
	ShowGrid   bool
	GridStepsX int
	GridStepsY int

	// Styling colours, interpreted as RGB (see package docs).
	BackgroundColor cv.Scalar
	AxisColor       cv.Scalar
	GridColor       cv.Scalar
	LineColor       cv.Scalar
	PointColor      cv.Scalar

	// LineThickness is the line/bar-outline width in pixels; PointRadius is the
	// scatter marker radius in pixels.
	LineThickness int
	PointRadius   int
}

// defaultPlot returns a Plot2D populated with the shared default styling.
func defaultPlot(x, y []float64) *Plot2D {
	p := &Plot2D{
		X:               x,
		Y:               y,
		Kind:            KindLine,
		Width:           640,
		Height:          480,
		MarginLeft:      50,
		MarginRight:     20,
		MarginTop:       20,
		MarginBottom:    40,
		autoRangeX:      true,
		autoRangeY:      true,
		ShowGrid:        true,
		GridStepsX:      10,
		GridStepsY:      10,
		BackgroundColor: cv.NewScalar(255, 255, 255),
		AxisColor:       cv.NewScalar(0, 0, 0),
		GridColor:       cv.NewScalar(200, 200, 200),
		LineColor:       cv.NewScalar(0, 0, 255),
		PointColor:      cv.NewScalar(255, 0, 0),
		LineThickness:   1,
		PointRadius:     3,
	}
	return p
}

// indexX returns the integer indices 0..n-1 as float64, used as x values when
// only y data is supplied.
func indexX(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = float64(i)
	}
	return x
}

// CreatePlot builds a line [Plot2D] from explicit x and y series. It panics if
// the two slices differ in length or are empty.
func CreatePlot(x, y []float64) *Plot2D {
	if len(x) != len(y) {
		panic("plot: CreatePlot requires len(x) == len(y)")
	}
	if len(y) == 0 {
		panic("plot: CreatePlot requires at least one point")
	}
	return defaultPlot(append([]float64(nil), x...), append([]float64(nil), y...))
}

// CreatePlotY builds a line [Plot2D] from y values alone, using the sample
// index (0,1,2,...) as x. It panics if y is empty.
func CreatePlotY(y []float64) *Plot2D {
	if len(y) == 0 {
		panic("plot: CreatePlotY requires at least one point")
	}
	return defaultPlot(indexX(len(y)), append([]float64(nil), y...))
}

// LinePlot builds a [Plot2D] of kind [KindLine] from x and y series.
func LinePlot(x, y []float64) *Plot2D {
	return CreatePlot(x, y).SetKind(KindLine)
}

// ScatterPlot builds a [Plot2D] of kind [KindScatter] from x and y series.
func ScatterPlot(x, y []float64) *Plot2D {
	return CreatePlot(x, y).SetKind(KindScatter)
}

// BarPlot builds a bar [Plot2D] from a slice of values, one bar per value with
// x taken as the value index. It panics if values is empty.
func BarPlot(values []float64) *Plot2D {
	if len(values) == 0 {
		panic("plot: BarPlot requires at least one value")
	}
	return defaultPlot(indexX(len(values)), append([]float64(nil), values...)).SetKind(KindBar)
}

// HistogramPlot bins data into the given number of equal-width bins spanning the
// data range and returns a bar [Plot2D] whose bar heights are the bin counts.
// The returned plot's x axis spans the data range and its bars sit at the bin
// centres. It panics if data is empty or bins < 1.
//
// The i-th bin covers [min + i*w, min + (i+1)*w) for bin width w =
// (max-min)/bins; the maximum value falls in the last bin. When all data are
// equal every sample lands in bin 0.
func HistogramPlot(data []float64, bins int) *Plot2D {
	if len(data) == 0 {
		panic("plot: HistogramPlot requires at least one value")
	}
	if bins < 1 {
		panic("plot: HistogramPlot requires bins >= 1")
	}
	lo, hi := data[0], data[0]
	for _, v := range data {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	counts := make([]float64, bins)
	width := (hi - lo) / float64(bins)
	for _, v := range data {
		var b int
		if width > 0 {
			b = int((v - lo) / width)
			if b >= bins {
				b = bins - 1
			}
			if b < 0 {
				b = 0
			}
		}
		counts[b]++
	}
	centres := make([]float64, bins)
	for i := range centres {
		if width > 0 {
			centres[i] = lo + (float64(i)+0.5)*width
		} else {
			centres[i] = lo
		}
	}
	p := defaultPlot(centres, counts).SetKind(KindBar)
	// Pin the x range to the full data span so bars tile the axis.
	if width > 0 {
		p.SetRangeX(lo, hi)
	}
	p.SetRangeY(0, maxFloat(counts))
	return p
}

// --- builder methods -------------------------------------------------------

// SetSize sets the canvas dimensions in pixels and returns the plot for
// chaining. It panics if either dimension is not positive.
func (p *Plot2D) SetSize(width, height int) *Plot2D {
	if width <= 0 || height <= 0 {
		panic("plot: SetSize requires positive dimensions")
	}
	p.Width, p.Height = width, height
	return p
}

// SetKind sets the rendering style and returns the plot for chaining.
func (p *Plot2D) SetKind(k PlotKind) *Plot2D { p.Kind = k; return p }

// SetRangeX pins the x axis bounds (disabling auto-ranging on x). It panics if
// min >= max.
func (p *Plot2D) SetRangeX(min, max float64) *Plot2D {
	if min >= max {
		panic("plot: SetRangeX requires min < max")
	}
	p.MinX, p.MaxX = min, max
	p.autoRangeX = false
	return p
}

// SetRangeY pins the y axis bounds (disabling auto-ranging). It panics if
// min >= max.
func (p *Plot2D) SetRangeY(min, max float64) *Plot2D {
	if min >= max {
		panic("plot: SetRangeY requires min < max")
	}
	p.MinY, p.MaxY = min, max
	p.autoRangeY = false
	return p
}

// SetShowGrid toggles the gridlines and returns the plot for chaining.
func (p *Plot2D) SetShowGrid(on bool) *Plot2D { p.ShowGrid = on; return p }

// SetLineColor sets the colour of the plotted line (and bar fill) and returns
// the plot for chaining.
func (p *Plot2D) SetLineColor(c cv.Scalar) *Plot2D { p.LineColor = c; return p }

// SetPointColor sets the colour of scatter markers and returns the plot for
// chaining.
func (p *Plot2D) SetPointColor(c cv.Scalar) *Plot2D { p.PointColor = c; return p }

// SetBackgroundColor sets the canvas fill colour and returns the plot for
// chaining.
func (p *Plot2D) SetBackgroundColor(c cv.Scalar) *Plot2D { p.BackgroundColor = c; return p }

// SetGridColor sets the gridline colour and returns the plot for chaining.
func (p *Plot2D) SetGridColor(c cv.Scalar) *Plot2D { p.GridColor = c; return p }

// SetAxisColor sets the axis-line colour and returns the plot for chaining.
func (p *Plot2D) SetAxisColor(c cv.Scalar) *Plot2D { p.AxisColor = c; return p }

// SetLineThickness sets the line/bar-outline width in pixels (minimum 1) and
// returns the plot for chaining.
func (p *Plot2D) SetLineThickness(t int) *Plot2D {
	if t < 1 {
		t = 1
	}
	p.LineThickness = t
	return p
}

// SetPointRadius sets the scatter marker radius in pixels (minimum 1) and
// returns the plot for chaining.
func (p *Plot2D) SetPointRadius(r int) *Plot2D {
	if r < 1 {
		r = 1
	}
	p.PointRadius = r
	return p
}

// --- rendering -------------------------------------------------------------

// plotArea returns the pixel bounds of the drawable region: x in [x0,x1] and y
// in [y0,y1], inclusive.
func (p *Plot2D) plotArea() (x0, y0, x1, y1 int) {
	x0 = p.MarginLeft
	y0 = p.MarginTop
	x1 = p.Width - 1 - p.MarginRight
	y1 = p.Height - 1 - p.MarginBottom
	if x1 <= x0 {
		x1 = x0 + 1
	}
	if y1 <= y0 {
		y1 = y0 + 1
	}
	return x0, y0, x1, y1
}

// resolveRange fills MinX..MaxY from the data when auto-ranging, expanding any
// degenerate (zero-width) span so the mapping stays finite.
func (p *Plot2D) resolveRange() {
	if p.autoRangeX {
		p.MinX, p.MaxX = minFloat(p.X), maxFloat(p.X)
		if p.MaxX == p.MinX {
			p.MinX -= 1
			p.MaxX += 1
		}
	}
	if p.autoRangeY {
		p.MinY, p.MaxY = minFloat(p.Y), maxFloat(p.Y)
		// Bars are measured from a zero baseline, so include it in the y span.
		if p.Kind == KindBar {
			if p.MinY > 0 {
				p.MinY = 0
			}
			if p.MaxY < 0 {
				p.MaxY = 0
			}
		}
		if p.MaxY == p.MinY {
			p.MinY -= 1
			p.MaxY += 1
		}
	}
}

// mapX converts a data x coordinate to a canvas column.
func (p *Plot2D) mapX(dx float64) int {
	x0, _, x1, _ := p.plotArea()
	t := (dx - p.MinX) / (p.MaxX - p.MinX)
	return x0 + int(math.Round(t*float64(x1-x0)))
}

// mapY converts a data y coordinate to a canvas row (y increases upward).
func (p *Plot2D) mapY(dy float64) int {
	_, y0, _, y1 := p.plotArea()
	t := (dy - p.MinY) / (p.MaxY - p.MinY)
	return y1 - int(math.Round(t*float64(y1-y0)))
}

// Render draws the configured chart onto a freshly allocated three-channel RGB
// [cv.Mat] of size Height×Width and returns it. It paints, in order, the
// background, the gridlines (when enabled), the x and y axis lines, and the data
// series according to Kind.
//
// Data point (dx, dy) maps to the pixel
//
//	col = MarginLeft + round((dx-MinX)/(MaxX-MinX) * areaWidth)
//	row = (Height-1-MarginBottom) - round((dy-MinY)/(MaxY-MinY) * areaHeight)
//
// where areaWidth and areaHeight are the plot-area extents. It panics if the
// canvas dimensions are not positive.
func (p *Plot2D) Render() *cv.Mat {
	if p.Width <= 0 || p.Height <= 0 {
		panic("plot: Render requires positive canvas dimensions")
	}
	canvas := cv.NewMat(p.Height, p.Width, 3)
	fill(canvas, p.BackgroundColor)
	p.resolveRange()

	x0, y0, x1, y1 := p.plotArea()

	if p.ShowGrid {
		p.drawGrid(canvas, x0, y0, x1, y1)
	}
	// Axis lines along the left and bottom of the plot area.
	cv.Line(canvas, cv.Point{X: x0, Y: y0}, cv.Point{X: x0, Y: y1}, p.AxisColor, 1)
	cv.Line(canvas, cv.Point{X: x0, Y: y1}, cv.Point{X: x1, Y: y1}, p.AxisColor, 1)

	switch p.Kind {
	case KindScatter:
		p.drawScatter(canvas)
	case KindBar:
		p.drawBars(canvas, x0, x1, y1)
	default:
		p.drawLine(canvas)
	}
	return canvas
}

// drawGrid paints evenly spaced horizontal and vertical gridlines.
func (p *Plot2D) drawGrid(canvas *cv.Mat, x0, y0, x1, y1 int) {
	if p.GridStepsX > 0 {
		for i := 0; i <= p.GridStepsX; i++ {
			x := x0 + int(math.Round(float64(i)/float64(p.GridStepsX)*float64(x1-x0)))
			cv.Line(canvas, cv.Point{X: x, Y: y0}, cv.Point{X: x, Y: y1}, p.GridColor, 1)
		}
	}
	if p.GridStepsY > 0 {
		for i := 0; i <= p.GridStepsY; i++ {
			y := y0 + int(math.Round(float64(i)/float64(p.GridStepsY)*float64(y1-y0)))
			cv.Line(canvas, cv.Point{X: x0, Y: y}, cv.Point{X: x1, Y: y}, p.GridColor, 1)
		}
	}
}

// drawLine connects successive samples with line segments.
func (p *Plot2D) drawLine(canvas *cv.Mat) {
	pts := make([]cv.Point, len(p.X))
	for i := range p.X {
		pts[i] = cv.Point{X: p.mapX(p.X[i]), Y: p.mapY(p.Y[i])}
	}
	if len(pts) == 1 {
		cv.Circle(canvas, pts[0], p.LineThickness, p.LineColor, cv.Filled)
		return
	}
	cv.Polylines(canvas, [][]cv.Point{pts}, false, p.LineColor, p.LineThickness)
}

// drawScatter draws a filled marker at each sample.
func (p *Plot2D) drawScatter(canvas *cv.Mat) {
	for i := range p.X {
		c := cv.Point{X: p.mapX(p.X[i]), Y: p.mapY(p.Y[i])}
		cv.Circle(canvas, c, p.PointRadius, p.PointColor, cv.Filled)
	}
}

// drawBars draws one filled vertical bar per sample from the y baseline.
func (p *Plot2D) drawBars(canvas *cv.Mat, x0, x1, y1 int) {
	n := len(p.X)
	// Bar half-width: half of the average x spacing, leaving a small gap.
	area := float64(x1 - x0)
	half := area / float64(n) / 2
	if half < 1 {
		half = 1
	}
	baseline := p.mapY(clampBaseline(p.MinY, p.MaxY))
	if baseline > y1 {
		baseline = y1
	}
	for i := range p.X {
		cx := p.mapX(p.X[i])
		top := p.mapY(p.Y[i])
		left := cx - int(half) + 1
		right := cx + int(half) - 1
		if right < left {
			right = left
		}
		cv.Rectangle(canvas,
			cv.Point{X: left, Y: baseline},
			cv.Point{X: right, Y: top},
			p.LineColor, cv.Filled)
	}
}

// clampBaseline returns the y baseline for bars: zero when it lies inside the
// range, otherwise the nearer bound.
func clampBaseline(lo, hi float64) float64 {
	if lo <= 0 && hi >= 0 {
		return 0
	}
	if lo > 0 {
		return lo
	}
	return hi
}

// fill paints every pixel of m with the given colour.
func fill(m *cv.Mat, color cv.Scalar) {
	cv.Rectangle(m, cv.Point{X: 0, Y: 0}, cv.Point{X: m.Cols - 1, Y: m.Rows - 1}, color, cv.Filled)
}

func minFloat(s []float64) float64 {
	m := s[0]
	for _, v := range s {
		if v < m {
			m = v
		}
	}
	return m
}

func maxFloat(s []float64) float64 {
	m := s[0]
	for _, v := range s {
		if v > m {
			m = v
		}
	}
	return m
}
