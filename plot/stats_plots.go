package plot

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// boxStats holds the five-number summary and outliers of one data group.
type boxStats struct {
	q1, median, q3   float64
	loWhisk, hiWhisk float64
	outliers         []float64
}

// summarize computes the quartiles, Tukey whiskers (nearest datum within 1.5×IQR
// of the quartiles) and outliers of a non-empty group.
func summarize(group []float64) boxStats {
	s := sortedCopy(group)
	q1 := quantile(s, 0.25)
	med := quantile(s, 0.5)
	q3 := quantile(s, 0.75)
	iqr := q3 - q1
	loFence := q1 - 1.5*iqr
	hiFence := q3 + 1.5*iqr
	loW, hiW := q3, q1
	var outliers []float64
	for _, v := range s {
		if v < loFence || v > hiFence {
			outliers = append(outliers, v)
			continue
		}
		if v < loW {
			loW = v
		}
		if v > hiW {
			hiW = v
		}
	}
	return boxStats{q1: q1, median: med, q3: q3, loWhisk: loW, hiWhisk: hiW, outliers: outliers}
}

// BoxPlot draws a box-and-whisker chart with one box per data group laid out
// left to right along a shared value axis. Construct one with [NewBoxPlot].
type BoxPlot struct {
	// Groups holds one slice of samples per box.
	Groups [][]float64
	// Labels, when non-nil, names each group beneath its box.
	Labels []string

	Width, Height                    int
	MarginLeft, MarginRight          int
	MarginTop, MarginBottom          int
	Background, AxisColor, GridColor cv.Scalar
	BoxColor, MedianColor            cv.Scalar
	WhiskerColor, OutlierColor       cv.Scalar
	TextColor                        cv.Scalar
	TextScale                        int
	ShowGrid                         bool
	GridStepsY                       int
}

// NewBoxPlot builds a [BoxPlot] over the given groups with default styling. It
// panics if no groups are given or any group is empty.
func NewBoxPlot(groups [][]float64) *BoxPlot {
	if len(groups) == 0 {
		panic("plot: NewBoxPlot requires at least one group")
	}
	cp := make([][]float64, len(groups))
	for i, g := range groups {
		if len(g) == 0 {
			panic("plot: NewBoxPlot group is empty")
		}
		cp[i] = append([]float64(nil), g...)
	}
	return &BoxPlot{
		Groups: cp,
		Width:  640, Height: 480,
		MarginLeft: 50, MarginRight: 20, MarginTop: 20, MarginBottom: 40,
		Background:   cv.NewScalar(255, 255, 255),
		AxisColor:    cv.NewScalar(0, 0, 0),
		GridColor:    cv.NewScalar(200, 200, 200),
		BoxColor:     cv.NewScalar(120, 170, 220),
		MedianColor:  cv.NewScalar(200, 40, 40),
		WhiskerColor: cv.NewScalar(0, 0, 0),
		OutlierColor: cv.NewScalar(120, 120, 120),
		TextColor:    cv.NewScalar(0, 0, 0),
		TextScale:    1,
		ShowGrid:     true,
		GridStepsY:   10,
	}
}

// SetSize sets the canvas dimensions and returns the plot for chaining.
func (p *BoxPlot) SetSize(w, h int) *BoxPlot { p.Width, p.Height = w, h; return p }

// SetLabels sets the per-group labels and returns the plot for chaining.
func (p *BoxPlot) SetLabels(labels []string) *BoxPlot { p.Labels = labels; return p }

// SetShowGrid toggles the horizontal grid and returns the plot for chaining.
func (p *BoxPlot) SetShowGrid(on bool) *BoxPlot { p.ShowGrid = on; return p }

// valueRange returns the global min/max across all groups (including outliers),
// padded by 5% so boxes do not touch the frame.
func (p *BoxPlot) valueRange() (lo, hi float64) {
	lo, hi = p.Groups[0][0], p.Groups[0][0]
	for _, g := range p.Groups {
		gl, gh := minMax(g)
		if gl < lo {
			lo = gl
		}
		if gh > hi {
			hi = gh
		}
	}
	pad := (hi - lo) * 0.05
	if pad == 0 {
		pad = 1
	}
	return lo - pad, hi + pad
}

// Render draws the box plot onto a fresh three-channel RGB [cv.Mat] of size
// Height×Width and returns it.
func (p *BoxPlot) Render() *cv.Mat {
	if p.Width <= 0 || p.Height <= 0 {
		panic("plot: Render requires positive canvas dimensions")
	}
	canvas := cv.NewMat(p.Height, p.Width, 3)
	fillBackground(canvas, p.Background)
	lo, hi := p.valueRange()
	n := len(p.Groups)
	a := newAxes(p.Width, p.Height, p.MarginLeft, p.MarginRight, p.MarginTop, p.MarginBottom, 0, float64(n), lo, hi)
	x0, y0, x1, y1 := a.area()
	if p.ShowGrid {
		drawGridLines(canvas, x0, y0, x1, y1, 0, p.GridStepsY, p.GridColor)
	}
	drawFrameAxes(canvas, x0, y0, x1, y1, p.AxisColor)

	slot := float64(x1-x0) / float64(n)
	half := int(slot * 0.3)
	if half < 2 {
		half = 2
	}
	for i, g := range p.Groups {
		st := summarize(g)
		cx := a.x(float64(i) + 0.5)
		yQ1 := a.y(st.q1)
		yQ3 := a.y(st.q3)
		yMed := a.y(st.median)
		yLo := a.y(st.loWhisk)
		yHi := a.y(st.hiWhisk)
		// Box body (Q1..Q3).
		cv.Rectangle(canvas, cv.Point{X: cx - half, Y: yQ3}, cv.Point{X: cx + half, Y: yQ1}, p.BoxColor, cv.Filled)
		cv.Rectangle(canvas, cv.Point{X: cx - half, Y: yQ3}, cv.Point{X: cx + half, Y: yQ1}, p.AxisColor, 1)
		// Median line.
		cv.Line(canvas, cv.Point{X: cx - half, Y: yMed}, cv.Point{X: cx + half, Y: yMed}, p.MedianColor, 2)
		// Whiskers with caps.
		cv.Line(canvas, cv.Point{X: cx, Y: yQ3}, cv.Point{X: cx, Y: yHi}, p.WhiskerColor, 1)
		cv.Line(canvas, cv.Point{X: cx, Y: yQ1}, cv.Point{X: cx, Y: yLo}, p.WhiskerColor, 1)
		cv.Line(canvas, cv.Point{X: cx - half/2, Y: yHi}, cv.Point{X: cx + half/2, Y: yHi}, p.WhiskerColor, 1)
		cv.Line(canvas, cv.Point{X: cx - half/2, Y: yLo}, cv.Point{X: cx + half/2, Y: yLo}, p.WhiskerColor, 1)
		// Outliers.
		for _, o := range st.outliers {
			cv.Circle(canvas, cv.Point{X: cx, Y: a.y(o)}, 2, p.OutlierColor, cv.Filled)
		}
		// Label.
		if i < len(p.Labels) {
			putTextCentered(canvas, p.Labels[i], cx, y1+p.MarginBottom-2, p.TextScale, p.TextColor)
		}
	}
	return canvas
}

// kernelDensity evaluates a Gaussian kernel-density estimate of samples at value
// v, using bandwidth h (Silverman's rule when h<=0). It is used to shape violins.
func kernelDensity(sorted []float64, v, h float64) float64 {
	n := len(sorted)
	if h <= 0 {
		_, std := meanStd(sorted)
		h = 1.06 * std * math.Pow(float64(n), -0.2)
		if h <= 0 {
			h = 1
		}
	}
	var sum float64
	for _, s := range sorted {
		u := (v - s) / h
		sum += math.Exp(-0.5 * u * u)
	}
	return sum / (float64(n) * h * math.Sqrt(2*math.Pi))
}

// ViolinPlot draws a violin chart: each group becomes a vertical, symmetric
// kernel-density silhouette whose half-width is proportional to the estimated
// density at each value, with a marker at the median. Construct one with
// [NewViolinPlot].
type ViolinPlot struct {
	// Groups holds one slice of samples per violin.
	Groups [][]float64
	// Labels, when non-nil, names each group beneath its violin.
	Labels []string

	Width, Height                    int
	MarginLeft, MarginRight          int
	MarginTop, MarginBottom          int
	Background, AxisColor, GridColor cv.Scalar
	FillColor, MedianColor           cv.Scalar
	TextColor                        cv.Scalar
	TextScale                        int
	// Resolution is the number of vertical samples used to trace each silhouette.
	Resolution int
	ShowGrid   bool
	GridStepsY int
}

// NewViolinPlot builds a [ViolinPlot] over the given groups with default styling.
// It panics if no groups are given or any group is empty.
func NewViolinPlot(groups [][]float64) *ViolinPlot {
	if len(groups) == 0 {
		panic("plot: NewViolinPlot requires at least one group")
	}
	cp := make([][]float64, len(groups))
	for i, g := range groups {
		if len(g) == 0 {
			panic("plot: NewViolinPlot group is empty")
		}
		cp[i] = append([]float64(nil), g...)
	}
	return &ViolinPlot{
		Groups: cp,
		Width:  640, Height: 480,
		MarginLeft: 50, MarginRight: 20, MarginTop: 20, MarginBottom: 40,
		Background:  cv.NewScalar(255, 255, 255),
		AxisColor:   cv.NewScalar(0, 0, 0),
		GridColor:   cv.NewScalar(200, 200, 200),
		FillColor:   cv.NewScalar(170, 200, 240),
		MedianColor: cv.NewScalar(200, 40, 40),
		TextColor:   cv.NewScalar(0, 0, 0),
		TextScale:   1,
		Resolution:  48,
		ShowGrid:    true,
		GridStepsY:  10,
	}
}

// SetSize sets the canvas dimensions and returns the plot for chaining.
func (p *ViolinPlot) SetSize(w, h int) *ViolinPlot { p.Width, p.Height = w, h; return p }

// SetLabels sets the per-group labels and returns the plot for chaining.
func (p *ViolinPlot) SetLabels(labels []string) *ViolinPlot { p.Labels = labels; return p }

// SetShowGrid toggles the horizontal grid and returns the plot for chaining.
func (p *ViolinPlot) SetShowGrid(on bool) *ViolinPlot { p.ShowGrid = on; return p }

func (p *ViolinPlot) valueRange() (lo, hi float64) {
	lo, hi = p.Groups[0][0], p.Groups[0][0]
	for _, g := range p.Groups {
		gl, gh := minMax(g)
		if gl < lo {
			lo = gl
		}
		if gh > hi {
			hi = gh
		}
	}
	pad := (hi - lo) * 0.08
	if pad == 0 {
		pad = 1
	}
	return lo - pad, hi + pad
}

// Render draws the violin plot onto a fresh three-channel RGB [cv.Mat] of size
// Height×Width and returns it.
func (p *ViolinPlot) Render() *cv.Mat {
	if p.Width <= 0 || p.Height <= 0 {
		panic("plot: Render requires positive canvas dimensions")
	}
	res := p.Resolution
	if res < 2 {
		res = 2
	}
	canvas := cv.NewMat(p.Height, p.Width, 3)
	fillBackground(canvas, p.Background)
	lo, hi := p.valueRange()
	n := len(p.Groups)
	a := newAxes(p.Width, p.Height, p.MarginLeft, p.MarginRight, p.MarginTop, p.MarginBottom, 0, float64(n), lo, hi)
	x0, y0, x1, y1 := a.area()
	if p.ShowGrid {
		drawGridLines(canvas, x0, y0, x1, y1, 0, p.GridStepsY, p.GridColor)
	}
	drawFrameAxes(canvas, x0, y0, x1, y1, p.AxisColor)

	slot := float64(x1-x0) / float64(n)
	maxHalf := slot * 0.42
	for i, g := range p.Groups {
		sorted := sortedCopy(g)
		cx := a.x(float64(i) + 0.5)
		// Sample the density along the value axis and find its peak.
		dens := make([]float64, res)
		peak := 0.0
		for k := 0; k < res; k++ {
			v := lo + (hi-lo)*float64(k)/float64(res-1)
			dens[k] = kernelDensity(sorted, v, 0)
			if dens[k] > peak {
				peak = dens[k]
			}
		}
		if peak == 0 {
			peak = 1
		}
		// Trace the right edge downward then the left edge upward.
		poly := make([]cv.Point, 0, 2*res)
		for k := 0; k < res; k++ {
			v := lo + (hi-lo)*float64(k)/float64(res-1)
			w := maxHalf * dens[k] / peak
			poly = append(poly, cv.Point{X: cx + int(w), Y: a.y(v)})
		}
		for k := res - 1; k >= 0; k-- {
			v := lo + (hi-lo)*float64(k)/float64(res-1)
			w := maxHalf * dens[k] / peak
			poly = append(poly, cv.Point{X: cx - int(w), Y: a.y(v)})
		}
		cv.FillPoly(canvas, [][]cv.Point{poly}, p.FillColor)
		cv.Polylines(canvas, [][]cv.Point{poly}, true, p.AxisColor, 1)
		// Median marker line.
		med := quantile(sorted, 0.5)
		yMed := a.y(med)
		hw := int(maxHalf * kernelDensity(sorted, med, 0) / peak)
		cv.Line(canvas, cv.Point{X: cx - hw, Y: yMed}, cv.Point{X: cx + hw, Y: yMed}, p.MedianColor, 2)
		if i < len(p.Labels) {
			putTextCentered(canvas, p.Labels[i], cx, y1+p.MarginBottom-2, p.TextScale, p.TextColor)
		}
	}
	return canvas
}

// ErrorBarPlot draws a scatter of (x,y) points each capped with a symmetric
// vertical error bar of half-height YErr. Construct one with [NewErrorBarPlot].
type ErrorBarPlot struct {
	X, Y, YErr []float64

	Width, Height                    int
	MarginLeft, MarginRight          int
	MarginTop, MarginBottom          int
	Background, AxisColor, GridColor cv.Scalar
	BarColor, PointColor             cv.Scalar
	PointRadius, CapWidth            int
	ShowGrid                         bool
	GridStepsX, GridStepsY           int
	autoY                            bool
	minY, maxY                       float64
}

// NewErrorBarPlot builds an [ErrorBarPlot]. The three slices must share a length
// and be non-empty; it panics otherwise. YErr holds the half-height of each bar.
func NewErrorBarPlot(x, y, yErr []float64) *ErrorBarPlot {
	if len(x) != len(y) || len(x) != len(yErr) {
		panic("plot: NewErrorBarPlot requires equal-length slices")
	}
	if len(x) == 0 {
		panic("plot: NewErrorBarPlot requires at least one point")
	}
	return &ErrorBarPlot{
		X: append([]float64(nil), x...), Y: append([]float64(nil), y...), YErr: append([]float64(nil), yErr...),
		Width: 640, Height: 480,
		MarginLeft: 50, MarginRight: 20, MarginTop: 20, MarginBottom: 40,
		Background:  cv.NewScalar(255, 255, 255),
		AxisColor:   cv.NewScalar(0, 0, 0),
		GridColor:   cv.NewScalar(200, 200, 200),
		BarColor:    cv.NewScalar(0, 0, 0),
		PointColor:  cv.NewScalar(200, 40, 40),
		PointRadius: 3, CapWidth: 6,
		ShowGrid: true, GridStepsX: 10, GridStepsY: 10,
		autoY: true,
	}
}

// SetSize sets the canvas dimensions and returns the plot for chaining.
func (p *ErrorBarPlot) SetSize(w, h int) *ErrorBarPlot { p.Width, p.Height = w, h; return p }

// SetShowGrid toggles the grid and returns the plot for chaining.
func (p *ErrorBarPlot) SetShowGrid(on bool) *ErrorBarPlot { p.ShowGrid = on; return p }

// SetRangeY pins the y axis bounds and returns the plot for chaining. It panics
// if min >= max.
func (p *ErrorBarPlot) SetRangeY(min, max float64) *ErrorBarPlot {
	if min >= max {
		panic("plot: SetRangeY requires min < max")
	}
	p.minY, p.maxY, p.autoY = min, max, false
	return p
}

// Render draws the error-bar chart onto a fresh three-channel RGB [cv.Mat] of
// size Height×Width and returns it.
func (p *ErrorBarPlot) Render() *cv.Mat {
	if p.Width <= 0 || p.Height <= 0 {
		panic("plot: Render requires positive canvas dimensions")
	}
	canvas := cv.NewMat(p.Height, p.Width, 3)
	fillBackground(canvas, p.Background)
	minX, maxX := minMax(p.X)
	minY, maxY := p.minY, p.maxY
	if p.autoY {
		minY, maxY = p.Y[0], p.Y[0]
		for i := range p.Y {
			if lo := p.Y[i] - p.YErr[i]; lo < minY {
				minY = lo
			}
			if hi := p.Y[i] + p.YErr[i]; hi > maxY {
				maxY = hi
			}
		}
	}
	a := newAxes(p.Width, p.Height, p.MarginLeft, p.MarginRight, p.MarginTop, p.MarginBottom, minX, maxX, minY, maxY)
	x0, y0, x1, y1 := a.area()
	if p.ShowGrid {
		drawGridLines(canvas, x0, y0, x1, y1, p.GridStepsX, p.GridStepsY, p.GridColor)
	}
	drawFrameAxes(canvas, x0, y0, x1, y1, p.AxisColor)

	capHalf := p.CapWidth / 2
	for i := range p.X {
		cx := a.x(p.X[i])
		yTop := a.y(p.Y[i] + p.YErr[i])
		yBot := a.y(p.Y[i] - p.YErr[i])
		cv.Line(canvas, cv.Point{X: cx, Y: yTop}, cv.Point{X: cx, Y: yBot}, p.BarColor, 1)
		cv.Line(canvas, cv.Point{X: cx - capHalf, Y: yTop}, cv.Point{X: cx + capHalf, Y: yTop}, p.BarColor, 1)
		cv.Line(canvas, cv.Point{X: cx - capHalf, Y: yBot}, cv.Point{X: cx + capHalf, Y: yBot}, p.BarColor, 1)
		cv.Circle(canvas, cv.Point{X: cx, Y: a.y(p.Y[i])}, p.PointRadius, p.PointColor, cv.Filled)
	}
	return canvas
}
