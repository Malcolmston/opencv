package plot

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// axes maps data coordinates into a rectangular pixel plot area inset from the
// canvas edges by four margins. The mapping matches [Plot2D]: x increases to the
// right and y increases upward (so larger data values map to smaller rows). It
// backs the additional plot types in this package.
type axes struct {
	width, height            int
	left, right, top, bottom int
	minX, maxX, minY, maxY   float64
}

// newAxes builds an axes for a canvas of the given size and margins and pins the
// data ranges, expanding any degenerate (zero-width) span so the mapping stays
// finite.
func newAxes(width, height, left, right, top, bottom int, minX, maxX, minY, maxY float64) *axes {
	if maxX == minX {
		minX -= 1
		maxX += 1
	}
	if maxY == minY {
		minY -= 1
		maxY += 1
	}
	return &axes{
		width: width, height: height,
		left: left, right: right, top: top, bottom: bottom,
		minX: minX, maxX: maxX, minY: minY, maxY: maxY,
	}
}

// area returns the inclusive pixel bounds of the drawable region.
func (a *axes) area() (x0, y0, x1, y1 int) {
	x0 = a.left
	y0 = a.top
	x1 = a.width - 1 - a.right
	y1 = a.height - 1 - a.bottom
	if x1 <= x0 {
		x1 = x0 + 1
	}
	if y1 <= y0 {
		y1 = y0 + 1
	}
	return x0, y0, x1, y1
}

// x maps a data x coordinate to a canvas column.
func (a *axes) x(dx float64) int {
	x0, _, x1, _ := a.area()
	t := (dx - a.minX) / (a.maxX - a.minX)
	return x0 + int(math.Round(t*float64(x1-x0)))
}

// y maps a data y coordinate to a canvas row (y increases upward).
func (a *axes) y(dy float64) int {
	_, y0, _, y1 := a.area()
	t := (dy - a.minY) / (a.maxY - a.minY)
	return y1 - int(math.Round(t*float64(y1-y0)))
}

// --- text -----------------------------------------------------------------

// glyphAdvance is the per-character horizontal step of the root package's 5×7
// bitmap font (5 px glyph + 1 px gap) before scaling.
const glyphAdvance = 6

// glyphRows is the height of the 5×7 bitmap font before scaling.
const glyphRows = 7

// TextSize returns the pixel width and height that [cv.PutText] uses to render s
// at the given integer scale, matching the built-in 5×7 bitmap font. The width
// is (6*len(s)-1)*scale for a non-empty string (the last glyph carries no
// trailing gap) and the height is 7*scale. An empty string has zero width.
func TextSize(s string, scale int) (w, h int) {
	if scale < 1 {
		scale = 1
	}
	n := len([]rune(s))
	if n == 0 {
		return 0, glyphRows * scale
	}
	return (glyphAdvance*n - 1) * scale, glyphRows * scale
}

// putTextCentered draws s so that it is horizontally centred on column cx, with
// baselineY as the text baseline (bottom of the glyphs).
func putTextCentered(m *cv.Mat, s string, cx, baselineY, scale int, color cv.Scalar) {
	w, _ := TextSize(s, scale)
	cv.PutText(m, s, cv.Point{X: cx - w/2, Y: baselineY}, scale, color)
}

// putTextVertical draws s as a top-to-bottom stack of glyphs anchored at column
// x, with the first glyph's top at topY. It is used for rotated-looking y-axis
// labels.
func putTextVertical(m *cv.Mat, s string, x, topY, scale int, color cv.Scalar) {
	if scale < 1 {
		scale = 1
	}
	for i, r := range s {
		baseline := topY + (i+1)*glyphRows*scale + i*2*scale
		cv.PutText(m, string(r), cv.Point{X: x, Y: baseline}, scale, color)
	}
}

// --- shared drawing --------------------------------------------------------

// fillBackground paints every pixel of m with color.
func fillBackground(m *cv.Mat, color cv.Scalar) {
	cv.Rectangle(m, cv.Point{X: 0, Y: 0}, cv.Point{X: m.Cols - 1, Y: m.Rows - 1}, color, cv.Filled)
}

// drawGridLines paints evenly spaced grid cells across the plot area.
func drawGridLines(m *cv.Mat, x0, y0, x1, y1, stepsX, stepsY int, color cv.Scalar) {
	if stepsX > 0 {
		for i := 0; i <= stepsX; i++ {
			x := x0 + int(math.Round(float64(i)/float64(stepsX)*float64(x1-x0)))
			cv.Line(m, cv.Point{X: x, Y: y0}, cv.Point{X: x, Y: y1}, color, 1)
		}
	}
	if stepsY > 0 {
		for i := 0; i <= stepsY; i++ {
			y := y0 + int(math.Round(float64(i)/float64(stepsY)*float64(y1-y0)))
			cv.Line(m, cv.Point{X: x0, Y: y}, cv.Point{X: x1, Y: y}, color, 1)
		}
	}
}

// drawFrameAxes draws the left and bottom axis lines of the plot area.
func drawFrameAxes(m *cv.Mat, x0, y0, x1, y1 int, color cv.Scalar) {
	cv.Line(m, cv.Point{X: x0, Y: y0}, cv.Point{X: x0, Y: y1}, color, 1)
	cv.Line(m, cv.Point{X: x0, Y: y1}, cv.Point{X: x1, Y: y1}, color, 1)
}

// LegendEntry is a single labelled colour swatch drawn by [DrawLegend].
type LegendEntry struct {
	// Label is the text shown next to the swatch.
	Label string
	// Color is the swatch fill colour, interpreted as RGB.
	Color cv.Scalar
}

// DrawLegend paints a boxed legend onto m with its top-left corner at (x, y).
// Each entry is drawn as a small filled colour swatch followed by its label in
// textColor at the given font scale. The legend is framed with a one-pixel
// border in textColor over a swatch of backgroundColor. It is a no-op when
// entries is empty.
func DrawLegend(m *cv.Mat, entries []LegendEntry, x, y, scale int, textColor, backgroundColor cv.Scalar) {
	if len(entries) == 0 {
		return
	}
	if scale < 1 {
		scale = 1
	}
	rowH := glyphRows*scale + 4*scale
	sw := glyphRows * scale // square swatch, same height as a glyph row block
	pad := 2 * scale
	// Widest label determines the box width.
	maxLabel := 0
	for _, e := range entries {
		if w, _ := TextSize(e.Label, scale); w > maxLabel {
			maxLabel = w
		}
	}
	boxW := pad + sw + pad + maxLabel + pad
	boxH := pad + len(entries)*rowH + pad
	cv.Rectangle(m, cv.Point{X: x, Y: y}, cv.Point{X: x + boxW, Y: y + boxH}, backgroundColor, cv.Filled)
	cv.Rectangle(m, cv.Point{X: x, Y: y}, cv.Point{X: x + boxW, Y: y + boxH}, textColor, 1)
	for i, e := range entries {
		ry := y + pad + i*rowH
		cv.Rectangle(m,
			cv.Point{X: x + pad, Y: ry},
			cv.Point{X: x + pad + sw, Y: ry + sw},
			e.Color, cv.Filled)
		cv.Rectangle(m,
			cv.Point{X: x + pad, Y: ry},
			cv.Point{X: x + pad + sw, Y: ry + sw},
			textColor, 1)
		baseline := ry + sw
		cv.PutText(m, e.Label, cv.Point{X: x + pad + sw + pad, Y: baseline}, scale, textColor)
	}
}

// --- statistics helpers ----------------------------------------------------

// sortedCopy returns a sorted copy of s (ascending) using insertion sort, which
// is deterministic and adequate for the small slices these plots handle.
func sortedCopy(s []float64) []float64 {
	out := append([]float64(nil), s...)
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j-1] > out[j]; j-- {
			out[j-1], out[j] = out[j], out[j-1]
		}
	}
	return out
}

// quantile returns the q-quantile (q in [0,1]) of an already-sorted slice using
// linear interpolation between order statistics. It panics if sorted is empty.
func quantile(sorted []float64, q float64) float64 {
	n := len(sorted)
	if n == 0 {
		panic("plot: quantile of empty slice")
	}
	if n == 1 {
		return sorted[0]
	}
	if q <= 0 {
		return sorted[0]
	}
	if q >= 1 {
		return sorted[n-1]
	}
	pos := q * float64(n-1)
	i := int(math.Floor(pos))
	f := pos - float64(i)
	return sorted[i] + (sorted[i+1]-sorted[i])*f
}

// meanStd returns the arithmetic mean and population standard deviation of s. It
// panics if s is empty.
func meanStd(s []float64) (mean, std float64) {
	if len(s) == 0 {
		panic("plot: meanStd of empty slice")
	}
	var sum float64
	for _, v := range s {
		sum += v
	}
	mean = sum / float64(len(s))
	var sq float64
	for _, v := range s {
		d := v - mean
		sq += d * d
	}
	std = math.Sqrt(sq / float64(len(s)))
	return mean, std
}

// minMax returns the smallest and largest values of s. It panics if s is empty.
func minMax(s []float64) (lo, hi float64) {
	if len(s) == 0 {
		panic("plot: minMax of empty slice")
	}
	lo, hi = s[0], s[0]
	for _, v := range s {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	return lo, hi
}
