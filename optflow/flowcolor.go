package optflow

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// colorWheel is the Middlebury optical-flow colour wheel: a sequence of RGB
// anchor colours spanning red→yellow→green→cyan→blue→magenta→red. Flow direction
// selects a hue by interpolating between adjacent anchors; flow magnitude scales
// saturation from white (still) to the full colour (fast). The segment counts
// (15,6,4,11,13,6) are those of the reference Middlebury implementation.
var colorWheel = buildColorWheel()

func buildColorWheel() [][3]float64 {
	const (
		ry = 15
		yg = 6
		gc = 4
		cb = 11
		bm = 13
		mr = 6
	)
	wheel := make([][3]float64, 0, ry+yg+gc+cb+bm+mr)
	add := func(r, g, b float64) { wheel = append(wheel, [3]float64{r, g, b}) }
	for i := 0; i < ry; i++ {
		add(255, 255*float64(i)/ry, 0)
	}
	for i := 0; i < yg; i++ {
		add(255-255*float64(i)/yg, 255, 0)
	}
	for i := 0; i < gc; i++ {
		add(0, 255, 255*float64(i)/gc)
	}
	for i := 0; i < cb; i++ {
		add(0, 255-255*float64(i)/cb, 255)
	}
	for i := 0; i < bm; i++ {
		add(255*float64(i)/bm, 0, 255)
	}
	for i := 0; i < mr; i++ {
		add(255, 0, 255-255*float64(i)/mr)
	}
	return wheel
}

// FlowToColor renders a flow field as a three-channel RGB image using the
// Middlebury colour-wheel convention: the hue of each pixel encodes the flow
// direction and the colour saturation encodes the magnitude, normalised by the
// largest magnitude in the field so the visualisation always uses the full
// range. Zero flow maps to white. The returned cv.Mat has the same dimensions
// as the field and exactly three channels (RGB).
//
// The field must be non-empty. The mapping is deterministic and independent of
// image content beyond the flow values themselves.
func FlowToColor(flow *FlowField) *cv.Mat {
	if flow == nil || flow.Rows <= 0 || flow.Cols <= 0 {
		panic("optflow: FlowToColor requires a non-empty flow field")
	}
	rows, cols := flow.Rows, flow.Cols
	out := cv.NewMat(rows, cols, 3)

	maxRad := flow.MaxMagnitude()
	if maxRad <= 0 {
		maxRad = 1 // avoid division by zero; field is entirely still → white
	}
	ncols := len(colorWheel)

	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := (y*cols + x) * 2
			u := flow.Data[i]
			v := flow.Data[i+1]
			rad := math.Hypot(u, v) / maxRad

			// Angle in [0,1): atan2 mapped from [-pi,pi] onto the wheel.
			a := math.Atan2(-v, -u) / math.Pi // [-1,1]
			fk := (a + 1) / 2 * float64(ncols-1)
			k0 := int(math.Floor(fk))
			k1 := (k0 + 1) % ncols
			f := fk - float64(k0)
			if k0 >= ncols {
				k0 = ncols - 1
			}

			var rgb [3]float64
			for c := 0; c < 3; c++ {
				col0 := colorWheel[k0][c] / 255.0
				col1 := colorWheel[k1][c] / 255.0
				col := (1-f)*col0 + f*col1
				if rad <= 1 {
					// Increase saturation toward the colour with radius.
					col = 1 - rad*(1-col)
				} else {
					// Beyond the normalising radius, darken slightly.
					col *= 0.75
				}
				rgb[c] = col * 255.0
			}
			set3(out, y, x, rgb[0], rgb[1], rgb[2])
		}
	}
	return out
}

// set3 writes a rounded, clamped RGB triple into a 3-channel Mat pixel.
func set3(m *cv.Mat, y, x int, r, g, b float64) {
	m.Set(y, x, 0, clampU8(r))
	m.Set(y, x, 1, clampU8(g))
	m.Set(y, x, 2, clampU8(b))
}

// clampU8 rounds to nearest and clamps into [0,255].
func clampU8(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}
