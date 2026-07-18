package draw2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// draw2clamp8 rounds v to the nearest integer and clamps it to the 0..255
// range of an 8-bit sample.
func draw2clamp8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v + 0.5)
}

// draw2inBounds reports whether pixel (x, y) lies inside m.
func draw2inBounds(m *cv.Mat, x, y int) bool {
	return x >= 0 && x < m.Cols && y >= 0 && y < m.Rows
}

// draw2index returns the flat Data offset of the first sample of pixel (x, y).
func draw2index(m *cv.Mat, x, y int) int {
	return (y*m.Cols + x) * m.Channels
}

// draw2set writes color into pixel (x, y), replacing whatever is there. It
// silently ignores out-of-range coordinates so primitives can clip. Only the
// first m.Channels components of color are used.
func draw2set(m *cv.Mat, x, y int, color cv.Scalar) {
	if !draw2inBounds(m, x, y) {
		return
	}
	i := draw2index(m, x, y)
	for c := 0; c < m.Channels; c++ {
		m.Data[i+c] = draw2clamp8(color[c])
	}
}

// draw2blend alpha-composites color over pixel (x, y) with coverage a in
// [0,1]: out = (1-a)*existing + a*color. Out-of-range coordinates and
// non-positive coverage are ignored.
func draw2blend(m *cv.Mat, x, y int, color cv.Scalar, a float64) {
	if a <= 0 || !draw2inBounds(m, x, y) {
		return
	}
	if a > 1 {
		a = 1
	}
	i := draw2index(m, x, y)
	for c := 0; c < m.Channels; c++ {
		existing := float64(m.Data[i+c])
		m.Data[i+c] = draw2clamp8(existing*(1-a) + color[c]*a)
	}
}

// draw2disc paints a filled disc of the given radius centred on (cx, cy). It
// is the thickness primitive shared by several outline routines.
func draw2disc(m *cv.Mat, cx, cy, radius int, color cv.Scalar) {
	if radius <= 0 {
		draw2set(m, cx, cy, color)
		return
	}
	r2 := radius * radius
	for dy := -radius; dy <= radius; dy++ {
		for dx := -radius; dx <= radius; dx++ {
			if dx*dx+dy*dy <= r2 {
				draw2set(m, cx+dx, cy+dy, color)
			}
		}
	}
}

// draw2fpart returns the fractional part of x.
func draw2fpart(x float64) float64 { return x - math.Floor(x) }

// draw2rfpart returns one minus the fractional part of x.
func draw2rfpart(x float64) float64 { return 1 - draw2fpart(x) }

// draw2round rounds to the nearest integer.
func draw2round(x float64) int { return int(math.Floor(x + 0.5)) }

func draw2absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func draw2signInt(v int) int {
	switch {
	case v > 0:
		return 1
	case v < 0:
		return -1
	default:
		return 0
	}
}

func draw2minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func draw2maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
