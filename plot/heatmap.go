package plot

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// HeatmapPlot renders a dense 2-D scalar field as a false-colour image: each
// matrix cell is normalised to [0,1] over the data range, mapped through a
// [Colormap] and drawn as a solid rectangle of CellSize×CellSize pixels. An
// optional vertical colorbar keyed to the same range is appended on the right.
// Construct one with [NewHeatmapPlot].
type HeatmapPlot struct {
	// Data is the field in row-major order: Data[r][c].
	Data [][]float64
	// Colormap selects the colour lookup table.
	Colormap Colormap
	// CellSize is the pixel edge length of each cell (minimum 1).
	CellSize int
	// MinValue and MaxValue pin the colour-mapped range; when Auto is true they
	// are recomputed from the data.
	MinValue, MaxValue float64
	Auto               bool
	// ShowColorbar appends a colour scale on the right when true.
	ShowColorbar bool
	// ColorbarWidth is the colorbar's pixel width when shown.
	ColorbarWidth int
	// Background fills any padding around the cells and colorbar.
	Background cv.Scalar
}

// NewHeatmapPlot builds a [HeatmapPlot] over a rectangular field with default
// styling. It panics if data is empty or its rows are ragged.
func NewHeatmapPlot(data [][]float64) *HeatmapPlot {
	if len(data) == 0 || len(data[0]) == 0 {
		panic("plot: NewHeatmapPlot requires a non-empty field")
	}
	w := len(data[0])
	cp := make([][]float64, len(data))
	for r, row := range data {
		if len(row) != w {
			panic("plot: NewHeatmapPlot requires rectangular data")
		}
		cp[r] = append([]float64(nil), row...)
	}
	return &HeatmapPlot{
		Data:          cp,
		Colormap:      ColormapViridis,
		CellSize:      16,
		Auto:          true,
		ShowColorbar:  true,
		ColorbarWidth: 24,
		Background:    cv.NewScalar(255, 255, 255),
	}
}

// SetColormap sets the colour lookup table and returns the plot for chaining.
func (p *HeatmapPlot) SetColormap(cm Colormap) *HeatmapPlot { p.Colormap = cm; return p }

// SetCellSize sets the per-cell pixel size (minimum 1) and returns the plot for
// chaining.
func (p *HeatmapPlot) SetCellSize(s int) *HeatmapPlot {
	if s < 1 {
		s = 1
	}
	p.CellSize = s
	return p
}

// SetRange pins the colour-mapped value range (disabling auto-ranging) and
// returns the plot for chaining. It panics if min >= max.
func (p *HeatmapPlot) SetRange(min, max float64) *HeatmapPlot {
	if min >= max {
		panic("plot: SetRange requires min < max")
	}
	p.MinValue, p.MaxValue, p.Auto = min, max, false
	return p
}

// SetShowColorbar toggles the colorbar and returns the plot for chaining.
func (p *HeatmapPlot) SetShowColorbar(on bool) *HeatmapPlot { p.ShowColorbar = on; return p }

// dataRange returns the value range used for normalisation.
func (p *HeatmapPlot) dataRange() (lo, hi float64) {
	if !p.Auto {
		return p.MinValue, p.MaxValue
	}
	lo, hi = p.Data[0][0], p.Data[0][0]
	for _, row := range p.Data {
		for _, v := range row {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
	}
	if hi == lo {
		hi = lo + 1
	}
	return lo, hi
}

// Render draws the heatmap onto a fresh three-channel RGB [cv.Mat] and returns
// it. The image is (rows*CellSize) tall; it is (cols*CellSize) wide, plus the
// colorbar and a small gap when ShowColorbar is set.
func (p *HeatmapPlot) Render() *cv.Mat {
	cs := p.CellSize
	if cs < 1 {
		cs = 1
	}
	rows := len(p.Data)
	cols := len(p.Data[0])
	gridW := cols * cs
	gridH := rows * cs
	gap := 0
	barW := 0
	if p.ShowColorbar {
		gap = cs
		barW = p.ColorbarWidth
		if barW < 1 {
			barW = 1
		}
	}
	width := gridW + gap + barW
	canvas := cv.NewMat(gridH, width, 3)
	fillBackground(canvas, p.Background)

	lo, hi := p.dataRange()
	table := Table(p.Colormap)
	span := hi - lo
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			t := (p.Data[r][c] - lo) / span
			idx := int(math.Round(clamp01(t) * 255))
			col := table[idx]
			cv.Rectangle(canvas,
				cv.Point{X: c * cs, Y: r * cs},
				cv.Point{X: c*cs + cs - 1, Y: r*cs + cs - 1},
				cv.NewScalar(float64(col[0]), float64(col[1]), float64(col[2])), cv.Filled)
		}
	}
	if p.ShowColorbar {
		drawColorbarInto(canvas, gridW+gap, 0, barW, gridH, table, false)
	}
	return canvas
}

// Colorbar renders a standalone vertical (or horizontal) colour scale for a
// [Colormap] onto a fresh three-channel RGB [cv.Mat]. Construct one with
// [NewColorbar]. Low values are at the bottom of a vertical bar and at the left
// of a horizontal bar.
type Colorbar struct {
	Colormap      Colormap
	Width, Height int
	Horizontal    bool
	Border        bool
	BorderColor   cv.Scalar
	Background    cv.Scalar
}

// NewColorbar builds a vertical [Colorbar] of the given pixel size for cm with
// default styling. It panics if either dimension is not positive.
func NewColorbar(cm Colormap, width, height int) *Colorbar {
	if width <= 0 || height <= 0 {
		panic("plot: NewColorbar requires positive dimensions")
	}
	return &Colorbar{
		Colormap: cm, Width: width, Height: height,
		Border:      true,
		BorderColor: cv.NewScalar(0, 0, 0),
		Background:  cv.NewScalar(255, 255, 255),
	}
}

// SetHorizontal orients the colorbar horizontally (low at left) and returns it
// for chaining.
func (c *Colorbar) SetHorizontal(on bool) *Colorbar { c.Horizontal = on; return c }

// Render draws the colorbar onto a fresh three-channel RGB [cv.Mat] of size
// Height×Width and returns it.
func (c *Colorbar) Render() *cv.Mat {
	canvas := cv.NewMat(c.Height, c.Width, 3)
	fillBackground(canvas, c.Background)
	drawColorbarInto(canvas, 0, 0, c.Width, c.Height, Table(c.Colormap), c.Horizontal)
	if c.Border {
		cv.Rectangle(canvas, cv.Point{X: 0, Y: 0}, cv.Point{X: c.Width - 1, Y: c.Height - 1}, c.BorderColor, 1)
	}
	return canvas
}

// drawColorbarInto paints a colour scale into the rectangle [x,x+w)×[y,y+h) of
// dst. When horizontal is false low intensities are drawn at the bottom; when
// true they are drawn at the left.
func drawColorbarInto(dst *cv.Mat, x, y, w, h int, table [][3]uint8, horizontal bool) {
	if horizontal {
		for i := 0; i < w; i++ {
			idx := int(math.Round(float64(i) / float64(maxOne(w-1)) * 255))
			col := table[idx]
			cv.Line(dst, cv.Point{X: x + i, Y: y}, cv.Point{X: x + i, Y: y + h - 1},
				cv.NewScalar(float64(col[0]), float64(col[1]), float64(col[2])), 1)
		}
		return
	}
	for j := 0; j < h; j++ {
		// Row 0 is the top; low intensity belongs at the bottom.
		idx := int(math.Round(float64(h-1-j) / float64(maxOne(h-1)) * 255))
		col := table[idx]
		cv.Line(dst, cv.Point{X: x, Y: y + j}, cv.Point{X: x + w - 1, Y: y + j},
			cv.NewScalar(float64(col[0]), float64(col[1]), float64(col[2])), 1)
	}
}

// maxOne returns v when positive, otherwise 1, to guard divisions.
func maxOne(v int) int {
	if v < 1 {
		return 1
	}
	return v
}

// ContourPlot draws iso-value contour lines of a 2-D scalar field using the
// marching-squares algorithm. Each requested level becomes a set of line
// segments where the bilinearly interpolated field crosses that value.
// Construct one with [NewContourPlot].
type ContourPlot struct {
	// Data is the field in row-major order: Data[r][c].
	Data [][]float64
	// Levels are the iso-values to trace; when nil, Render picks evenly spaced
	// levels across the data range.
	Levels []float64
	// NumLevels is the count of auto levels used when Levels is nil.
	NumLevels int

	Width, Height int
	Background    cv.Scalar
	LineColor     cv.Scalar
	LineThickness int
	// FillBelow floods the background with the low-value colormap tint when true.
	FillBelow bool
	Colormap  Colormap
}

// NewContourPlot builds a [ContourPlot] over a rectangular field with default
// styling. It panics if data is empty, ragged, or smaller than 2×2.
func NewContourPlot(data [][]float64) *ContourPlot {
	if len(data) < 2 || len(data[0]) < 2 {
		panic("plot: NewContourPlot requires at least a 2x2 field")
	}
	w := len(data[0])
	cp := make([][]float64, len(data))
	for r, row := range data {
		if len(row) != w {
			panic("plot: NewContourPlot requires rectangular data")
		}
		cp[r] = append([]float64(nil), row...)
	}
	return &ContourPlot{
		Data:      cp,
		NumLevels: 6,
		Width:     480, Height: 480,
		Background:    cv.NewScalar(255, 255, 255),
		LineColor:     cv.NewScalar(0, 0, 0),
		LineThickness: 1,
		Colormap:      ColormapViridis,
	}
}

// SetSize sets the canvas dimensions and returns the plot for chaining.
func (p *ContourPlot) SetSize(w, h int) *ContourPlot { p.Width, p.Height = w, h; return p }

// SetLevels sets explicit iso-values and returns the plot for chaining.
func (p *ContourPlot) SetLevels(levels []float64) *ContourPlot {
	p.Levels = append([]float64(nil), levels...)
	return p
}

// SetFillBelow toggles the colormap background tint and returns the plot for
// chaining.
func (p *ContourPlot) SetFillBelow(on bool) *ContourPlot { p.FillBelow = on; return p }

// levels returns the iso-values to draw, computing evenly spaced interior levels
// when none were set explicitly.
func (p *ContourPlot) levelValues() []float64 {
	if len(p.Levels) > 0 {
		return p.Levels
	}
	lo, hi := p.Data[0][0], p.Data[0][0]
	for _, row := range p.Data {
		for _, v := range row {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
	}
	n := p.NumLevels
	if n < 1 {
		n = 1
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = lo + (hi-lo)*float64(i+1)/float64(n+1)
	}
	return out
}

// Render draws the contour lines onto a fresh three-channel RGB [cv.Mat] of size
// Height×Width and returns it. Field coordinates are scaled so the grid fills
// the canvas: cell column c maps to x = c/(cols-1)*(Width-1) and row r to
// y = r/(rows-1)*(Height-1).
func (p *ContourPlot) Render() *cv.Mat {
	if p.Width <= 0 || p.Height <= 0 {
		panic("plot: Render requires positive canvas dimensions")
	}
	rows := len(p.Data)
	cols := len(p.Data[0])
	canvas := cv.NewMat(p.Height, p.Width, 3)
	fillBackground(canvas, p.Background)

	if p.FillBelow {
		table := Table(p.Colormap)
		lo, hi := p.Data[0][0], p.Data[0][0]
		for _, row := range p.Data {
			for _, v := range row {
				if v < lo {
					lo = v
				}
				if v > hi {
					hi = v
				}
			}
		}
		span := hi - lo
		if span == 0 {
			span = 1
		}
		for py := 0; py < p.Height; py++ {
			fr := float64(py) / float64(maxOne(p.Height-1)) * float64(rows-1)
			for px := 0; px < p.Width; px++ {
				fc := float64(px) / float64(maxOne(p.Width-1)) * float64(cols-1)
				v := bilinear(p.Data, fr, fc)
				idx := int(math.Round(clamp01((v-lo)/span) * 255))
				col := table[idx]
				canvas.SetPixel(py, px, []uint8{col[0], col[1], col[2]})
			}
		}
	}

	sx := func(c float64) int { return int(math.Round(c / float64(maxOne(cols-1)) * float64(p.Width-1))) }
	sy := func(r float64) int { return int(math.Round(r / float64(maxOne(rows-1)) * float64(p.Height-1))) }

	for _, level := range p.levelValues() {
		for r := 0; r < rows-1; r++ {
			for c := 0; c < cols-1; c++ {
				segs := marchingSquaresCell(p.Data, r, c, level)
				for _, s := range segs {
					cv.Line(canvas,
						cv.Point{X: sx(s[1]), Y: sy(s[0])},
						cv.Point{X: sx(s[3]), Y: sy(s[2])},
						p.LineColor, p.LineThickness)
				}
			}
		}
	}
	return canvas
}

// bilinear samples data at fractional row fr, column fc with bilinear
// interpolation and edge clamping.
func bilinear(data [][]float64, fr, fc float64) float64 {
	rows := len(data)
	cols := len(data[0])
	r0 := int(math.Floor(fr))
	c0 := int(math.Floor(fc))
	r1 := r0 + 1
	c1 := c0 + 1
	clampi := func(v, hi int) int {
		if v < 0 {
			return 0
		}
		if v > hi {
			return hi
		}
		return v
	}
	r0 = clampi(r0, rows-1)
	r1 = clampi(r1, rows-1)
	c0 = clampi(c0, cols-1)
	c1 = clampi(c1, cols-1)
	dr := fr - math.Floor(fr)
	dc := fc - math.Floor(fc)
	top := data[r0][c0]*(1-dc) + data[r0][c1]*dc
	bot := data[r1][c0]*(1-dc) + data[r1][c1]*dc
	return top*(1-dr) + bot*dr
}

// marchingSquaresCell returns the iso-line segment(s) crossing the grid cell
// whose top-left corner is (r,c) at the given level. Each segment is
// {r1,c1,r2,c2} in fractional grid coordinates.
func marchingSquaresCell(data [][]float64, r, c int, level float64) [][4]float64 {
	// Corner values, clockwise from top-left.
	tl := data[r][c]
	tr := data[r][c+1]
	br := data[r+1][c+1]
	bl := data[r+1][c]
	// Edge crossing points (fractional grid coords), nil when the edge does not
	// cross the level.
	type pt struct {
		ok     bool
		rr, cc float64
	}
	interp := func(v0, v1 float64, r0, c0, r1, c1 float64) pt {
		if (v0 < level) == (v1 < level) {
			return pt{}
		}
		t := (level - v0) / (v1 - v0)
		return pt{true, r0 + (r1-r0)*t, c0 + (c1-c0)*t}
	}
	top := interp(tl, tr, float64(r), float64(c), float64(r), float64(c+1))
	right := interp(tr, br, float64(r), float64(c+1), float64(r+1), float64(c+1))
	bottom := interp(bl, br, float64(r+1), float64(c), float64(r+1), float64(c+1))
	left := interp(tl, bl, float64(r), float64(c), float64(r+1), float64(c))

	var crossings []pt
	for _, e := range []pt{top, right, bottom, left} {
		if e.ok {
			crossings = append(crossings, e)
		}
	}
	// A cell has 0, 2 or 4 crossings; pair them into segments in order.
	var out [][4]float64
	for i := 0; i+1 < len(crossings); i += 2 {
		a := crossings[i]
		b := crossings[i+1]
		out = append(out, [4]float64{a.rr, a.cc, b.rr, b.cc})
	}
	return out
}
