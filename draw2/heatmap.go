package draw2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Palette selects a false-colour ramp for the colour-mapping routines.
type Palette int

const (
	// PaletteGray is a neutral black-to-white ramp.
	PaletteGray Palette = iota
	// PaletteJet is the classic blue-cyan-yellow-red ramp.
	PaletteJet
	// PaletteHot is a black-red-yellow-white ramp.
	PaletteHot
	// PaletteCool is a cyan-to-magenta ramp.
	PaletteCool
	// PaletteBone is a grey ramp with a cool blue tint in the shadows.
	PaletteBone
	// PaletteRainbow sweeps hue from red through green to blue.
	PaletteRainbow
	// PaletteViridis is the perceptually-uniform purple-green-yellow ramp.
	PaletteViridis
)

// clamp01 clamps v to the unit interval.
func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// PaletteColor maps a normalised intensity t in [0,1] through the given
// palette and returns the resulting opaque RGB colour. t is clamped to the
// valid range.
func PaletteColor(p Palette, t float64) cv.Scalar {
	t = clamp01(t)
	var r, g, b float64
	switch p {
	case PaletteGray:
		r, g, b = t, t, t
	case PaletteJet:
		r = clamp01(1.5 - math.Abs(4*t-3))
		g = clamp01(1.5 - math.Abs(4*t-2))
		b = clamp01(1.5 - math.Abs(4*t-1))
	case PaletteHot:
		r = clamp01(t / 0.375)
		g = clamp01((t - 0.375) / 0.375)
		b = clamp01((t - 0.75) / 0.25)
	case PaletteCool:
		r = t
		g = 1 - t
		b = 1
	case PaletteBone:
		if t < 0.75 {
			r = 0.875 * t
		} else {
			r = 1.375*t - 0.375
		}
		switch {
		case t < 0.375:
			g = 0.875 * t
		case t < 0.75:
			g = 1.2083333333333333*t - 0.125
		default:
			g = 0.875*t + 0.125
		}
		if t < 0.375 {
			b = 1.2083333333333333 * t
		} else {
			b = 0.875*t + 0.125
		}
	case PaletteRainbow:
		r, g, b = draw2hsv((1-t)*240, 1, 1)
	case PaletteViridis:
		r, g, b = draw2viridis(t)
	default:
		r, g, b = t, t, t
	}
	return cv.Scalar{r * 255, g * 255, b * 255, 255}
}

// draw2viridis interpolates the viridis ramp from a small anchor table.
func draw2viridis(t float64) (r, g, b float64) {
	anchors := [...][3]float64{
		{0.267004, 0.004874, 0.329415},
		{0.229739, 0.322361, 0.545706},
		{0.127568, 0.566949, 0.550556},
		{0.369214, 0.788888, 0.382914},
		{0.993248, 0.906157, 0.143936},
	}
	n := len(anchors) - 1
	x := t * float64(n)
	i := int(math.Floor(x))
	if i >= n {
		i = n - 1
	}
	f := x - float64(i)
	a := anchors[i]
	c := anchors[i+1]
	return a[0] + (c[0]-a[0])*f, a[1] + (c[1]-a[1])*f, a[2] + (c[2]-a[2])*f
}

// draw2hsv converts an HSV colour (hue in degrees, s and v in [0,1]) to RGB
// components in [0,1].
func draw2hsv(h, s, v float64) (r, g, b float64) {
	h = math.Mod(h, 360)
	if h < 0 {
		h += 360
	}
	c := v * s
	x := c * (1 - math.Abs(math.Mod(h/60, 2)-1))
	mm := v - c
	switch {
	case h < 60:
		r, g, b = c, x, 0
	case h < 120:
		r, g, b = x, c, 0
	case h < 180:
		r, g, b = 0, c, x
	case h < 240:
		r, g, b = 0, x, c
	case h < 300:
		r, g, b = x, 0, c
	default:
		r, g, b = c, 0, x
	}
	return r + mm, g + mm, b + mm
}

// Colorize maps a single-channel Mat to a new three-channel RGB Mat by passing
// each 8-bit intensity (scaled to [0,1]) through the given palette. It panics
// if src is not single-channel.
func Colorize(src *cv.Mat, p Palette) *cv.Mat {
	if src.Channels != 1 {
		panic("draw2: Colorize requires a single-channel Mat")
	}
	out := cv.NewMat(src.Rows, src.Cols, 3)
	for i := 0; i < src.Rows*src.Cols; i++ {
		col := PaletteColor(p, float64(src.Data[i])/255)
		out.Data[i*3+0] = draw2clamp8(col[0])
		out.Data[i*3+1] = draw2clamp8(col[1])
		out.Data[i*3+2] = draw2clamp8(col[2])
	}
	return out
}

// Heatmap renders a two-dimensional grid of scalar values as a three-channel
// RGB Mat. Values are linearly normalised over their own [min,max] range and
// mapped through the given palette. Each inner slice is one image row; rows are
// padded with the palette's zero colour if they are shorter than the widest
// row. It returns nil for an empty grid.
func Heatmap(values [][]float64, p Palette) *cv.Mat {
	rows := len(values)
	if rows == 0 {
		return nil
	}
	cols := 0
	minV, maxV := math.Inf(1), math.Inf(-1)
	for _, row := range values {
		if len(row) > cols {
			cols = len(row)
		}
		for _, v := range row {
			if v < minV {
				minV = v
			}
			if v > maxV {
				maxV = v
			}
		}
	}
	if cols == 0 {
		return nil
	}
	span := maxV - minV
	out := cv.NewMat(rows, cols, 3)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var t float64
			if x < len(values[y]) && span > 0 {
				t = (values[y][x] - minV) / span
			}
			col := PaletteColor(p, t)
			i := (y*cols + x) * 3
			out.Data[i+0] = draw2clamp8(col[0])
			out.Data[i+1] = draw2clamp8(col[1])
			out.Data[i+2] = draw2clamp8(col[2])
		}
	}
	return out
}

// ColorBar renders a vertical colour-bar legend of the given width and height
// into a new three-channel Mat, with the palette's maximum at the top and its
// minimum at the bottom. Both dimensions are clamped to a minimum of 1.
func ColorBar(width, height int, p Palette) *cv.Mat {
	if width < 1 {
		width = 1
	}
	if height < 1 {
		height = 1
	}
	out := cv.NewMat(height, width, 3)
	for y := 0; y < height; y++ {
		t := 1 - float64(y)/float64(height-1)
		if height == 1 {
			t = 1
		}
		col := PaletteColor(p, t)
		for x := 0; x < width; x++ {
			i := (y*width + x) * 3
			out.Data[i+0] = draw2clamp8(col[0])
			out.Data[i+1] = draw2clamp8(col[1])
			out.Data[i+2] = draw2clamp8(col[2])
		}
	}
	return out
}
