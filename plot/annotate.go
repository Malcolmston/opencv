package plot

import (
	"strconv"

	cv "github.com/malcolmston/opencv"
)

// Annotation collects the axis labels, title, numeric tick marks and legend that
// [Plot2D.RenderAnnotated] paints over a base [Plot2D] chart. The zero value
// draws nothing extra; set only the fields you want.
type Annotation struct {
	// Title is centred along the top margin.
	Title string
	// XLabel is centred along the bottom margin; YLabel is stacked down the left
	// margin.
	XLabel, YLabel string
	// ShowTicks draws numeric tick labels along both axes.
	ShowTicks bool
	// TicksX and TicksY are the number of tick intervals per axis (each produces
	// count+1 labels). Zero falls back to 5.
	TicksX, TicksY int
	// Legend, when non-empty, is drawn inside the plot area's top-left corner.
	Legend []LegendEntry
	// TextColor is the colour of all annotation text and tick marks.
	TextColor cv.Scalar
	// TextScale magnifies the bitmap font (minimum 1).
	TextScale int
}

// RenderAnnotated renders the base [Plot2D] with [Plot2D.Render] and then paints
// the annotations from a onto the same canvas, returning it. Because the base
// chart already reserves the four margins, titles and axis labels land in that
// reserved space and tick labels sit just outside the plot area. The plot's data
// ranges are resolved by the underlying render, so tick values reflect the axis
// bounds actually used.
func (p *Plot2D) RenderAnnotated(a Annotation) *cv.Mat {
	canvas := p.Render()
	scale := a.TextScale
	if scale < 1 {
		scale = 1
	}
	col := a.TextColor
	x0, y0, x1, y1 := p.plotArea()

	if a.Title != "" {
		putTextCentered(canvas, a.Title, (x0+x1)/2, y0-2, scale, col)
	}
	if a.XLabel != "" {
		putTextCentered(canvas, a.XLabel, (x0+x1)/2, p.Height-2, scale, col)
	}
	if a.YLabel != "" {
		_, th := TextSize(a.YLabel, scale)
		topY := (y0+y1)/2 - (len(a.YLabel)*(th+2*scale))/2
		putTextVertical(canvas, a.YLabel, 2, topY, scale, col)
	}

	if a.ShowTicks {
		tx := a.TicksX
		if tx <= 0 {
			tx = 5
		}
		ty := a.TicksY
		if ty <= 0 {
			ty = 5
		}
		for i := 0; i <= tx; i++ {
			f := float64(i) / float64(tx)
			val := p.MinX + (p.MaxX-p.MinX)*f
			px := x0 + int(float64(x1-x0)*f)
			cv.Line(canvas, cv.Point{X: px, Y: y1}, cv.Point{X: px, Y: y1 + 4}, col, 1)
			putTextCentered(canvas, formatTick(val), px, y1+4+TextHeight(scale), scale, col)
		}
		for i := 0; i <= ty; i++ {
			f := float64(i) / float64(ty)
			val := p.MinY + (p.MaxY-p.MinY)*f
			py := y1 - int(float64(y1-y0)*f)
			cv.Line(canvas, cv.Point{X: x0 - 4, Y: py}, cv.Point{X: x0, Y: py}, col, 1)
			label := formatTick(val)
			w, h := TextSize(label, scale)
			cv.PutText(canvas, label, cv.Point{X: x0 - 6 - w, Y: py + h/2}, scale, col)
		}
	}

	if len(a.Legend) > 0 {
		DrawLegend(canvas, a.Legend, x0+6, y0+6, scale, col, p.BackgroundColor)
	}
	return canvas
}

// TextHeight returns the pixel height [cv.PutText] uses at the given scale (7*scale).
func TextHeight(scale int) int {
	if scale < 1 {
		scale = 1
	}
	return glyphRows * scale
}

// formatTick renders a tick value compactly: integers without a decimal point,
// otherwise trimmed to at most two fractional digits.
func formatTick(v float64) string {
	if v == float64(int64(v)) {
		return strconv.FormatInt(int64(v), 10)
	}
	s := strconv.FormatFloat(v, 'f', 2, 64)
	// Trim trailing zeros and any dangling decimal point.
	for len(s) > 0 && s[len(s)-1] == '0' {
		s = s[:len(s)-1]
	}
	if len(s) > 0 && s[len(s)-1] == '.' {
		s = s[:len(s)-1]
	}
	return s
}
