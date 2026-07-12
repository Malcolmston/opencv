package plot

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// defaultWedgeColors is the cyclic palette used for pie wedges and overlay
// series when the caller supplies none.
var defaultWedgeColors = []cv.Scalar{
	cv.NewScalar(31, 119, 180),
	cv.NewScalar(255, 127, 14),
	cv.NewScalar(44, 160, 44),
	cv.NewScalar(214, 39, 40),
	cv.NewScalar(148, 103, 189),
	cv.NewScalar(140, 86, 75),
	cv.NewScalar(227, 119, 194),
	cv.NewScalar(188, 189, 34),
	cv.NewScalar(23, 190, 207),
}

// PiePlot draws a pie chart whose wedge angles are proportional to the supplied
// values. Wedges start at 12 o'clock and proceed clockwise. Construct one with
// [NewPiePlot].
type PiePlot struct {
	// Values are the non-negative wedge magnitudes; they need not sum to one.
	Values []float64
	// Colors cycles through the wedge fill colours; defaults are used when nil.
	Colors []cv.Scalar
	// Labels, when non-nil, populate a legend on the right.
	Labels []string

	Width, Height int
	Background    cv.Scalar
	BorderColor   cv.Scalar
	TextColor     cv.Scalar
	TextScale     int
	// ShowLegend draws a labelled legend when Labels is set.
	ShowLegend bool
}

// NewPiePlot builds a [PiePlot] with default styling. It panics if values is
// empty, contains a negative entry, or sums to zero.
func NewPiePlot(values []float64) *PiePlot {
	if len(values) == 0 {
		panic("plot: NewPiePlot requires at least one value")
	}
	var sum float64
	for _, v := range values {
		if v < 0 {
			panic("plot: NewPiePlot values must be non-negative")
		}
		sum += v
	}
	if sum == 0 {
		panic("plot: NewPiePlot values must not sum to zero")
	}
	return &PiePlot{
		Values: append([]float64(nil), values...),
		Width:  480, Height: 480,
		Background:  cv.NewScalar(255, 255, 255),
		BorderColor: cv.NewScalar(255, 255, 255),
		TextColor:   cv.NewScalar(0, 0, 0),
		TextScale:   1,
		ShowLegend:  true,
	}
}

// SetSize sets the canvas dimensions and returns the plot for chaining.
func (p *PiePlot) SetSize(w, h int) *PiePlot { p.Width, p.Height = w, h; return p }

// SetColors sets the wedge colour cycle and returns the plot for chaining.
func (p *PiePlot) SetColors(colors []cv.Scalar) *PiePlot { p.Colors = colors; return p }

// SetLabels sets the legend labels and returns the plot for chaining.
func (p *PiePlot) SetLabels(labels []string) *PiePlot { p.Labels = labels; return p }

// wedgeColor returns the fill colour for wedge i, cycling the palette.
func (p *PiePlot) wedgeColor(i int) cv.Scalar {
	pal := p.Colors
	if len(pal) == 0 {
		pal = defaultWedgeColors
	}
	return pal[i%len(pal)]
}

// Render draws the pie chart onto a fresh three-channel RGB [cv.Mat] of size
// Height×Width and returns it. The pie is centred in the square region on the
// left; a legend (when enabled and Labels is set) occupies the right margin.
func (p *PiePlot) Render() *cv.Mat {
	if p.Width <= 0 || p.Height <= 0 {
		panic("plot: Render requires positive canvas dimensions")
	}
	canvas := cv.NewMat(p.Height, p.Width, 3)
	fillBackground(canvas, p.Background)

	legendW := 0
	if p.ShowLegend && len(p.Labels) > 0 {
		legendW = p.Width / 4
	}
	pieW := p.Width - legendW
	cx := pieW / 2
	cy := p.Height / 2
	radius := pieW / 2
	if p.Height/2 < radius {
		radius = p.Height / 2
	}
	radius = int(float64(radius) * 0.9)
	if radius < 1 {
		radius = 1
	}

	var total float64
	for _, v := range p.Values {
		total += v
	}
	// Start at -90° (12 o'clock) and sweep clockwise (increasing angle in image
	// coordinates where y grows downward).
	const steps = 720
	start := -math.Pi / 2
	for i, v := range p.Values {
		frac := v / total
		end := start + frac*2*math.Pi
		n := int(math.Ceil(frac * steps))
		if n < 1 {
			n = 1
		}
		poly := make([]cv.Point, 0, n+2)
		poly = append(poly, cv.Point{X: cx, Y: cy})
		for k := 0; k <= n; k++ {
			ang := start + (end-start)*float64(k)/float64(n)
			poly = append(poly, cv.Point{
				X: cx + int(math.Round(float64(radius)*math.Cos(ang))),
				Y: cy + int(math.Round(float64(radius)*math.Sin(ang))),
			})
		}
		cv.FillPoly(canvas, [][]cv.Point{poly}, p.wedgeColor(i))
		// Wedge border spokes.
		cv.Polylines(canvas, [][]cv.Point{poly}, true, p.BorderColor, 1)
		start = end
	}

	if legendW > 0 {
		entries := make([]LegendEntry, len(p.Labels))
		for i := range p.Labels {
			entries[i] = LegendEntry{Label: p.Labels[i], Color: p.wedgeColor(i)}
		}
		DrawLegend(canvas, entries, pieW+8, 10, p.TextScale, p.TextColor, p.Background)
	}
	return canvas
}
