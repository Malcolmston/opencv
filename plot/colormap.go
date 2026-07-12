package plot

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Colormap selects one of the built-in 256-entry colour lookup tables used by
// [ApplyColorMap]. Each colormap maps an 8-bit intensity (0..255) to an RGB
// colour; the documented endpoint colours are the values at intensity 0 and 255.
type Colormap int

const (
	// ColormapJet is the classic rainbow map: 0 is dark blue (0,0,128) and 255
	// is dark red (128,0,0), passing through cyan, green and yellow.
	ColormapJet Colormap = iota
	// ColormapHot runs black-red-yellow-white: 0 is black (0,0,0) and 255 is
	// white (255,255,255).
	ColormapHot
	// ColormapCool runs cyan to magenta: 0 is cyan (0,255,255) and 255 is
	// magenta (255,0,255).
	ColormapCool
	// ColormapBone is a grayscale with a cool blue tint: 0 is black (0,0,0) and
	// 255 is white (255,255,255).
	ColormapBone
	// ColormapHSV cycles through the full hue circle at full saturation and
	// value: 0 is red (255,0,0) and 255 wraps back to red (255,0,0).
	ColormapHSV
	// ColormapViridis is the perceptually uniform viridis map: 0 is dark purple
	// (68,1,84) and 255 is yellow (253,231,37).
	ColormapViridis
	// ColormapPlasma is the perceptually uniform plasma map: 0 is dark blue
	// (13,8,135) and 255 is yellow (240,249,33).
	ColormapPlasma
	// ColormapGrayscale is the identity map: 0 is black (0,0,0) and 255 is white
	// (255,255,255), each channel equal to the input intensity.
	ColormapGrayscale
)

// clamp01 limits t to the closed interval [0,1].
func clamp01(t float64) float64 {
	if t < 0 {
		return 0
	}
	if t > 1 {
		return 1
	}
	return t
}

// u8 scales a fraction in [0,1] to an 8-bit sample with round-half-up.
func u8(f float64) uint8 {
	v := math.Round(clamp01(f) * 255)
	return uint8(v)
}

// ColormapTable returns the 256-entry lookup table for the given built-in
// colormap. Entry i is the [3]uint8 RGB colour for intensity i. It panics if cm
// is not a recognised colormap.
func ColormapTable(cm Colormap) [][3]uint8 {
	switch cm {
	case ColormapJet:
		return buildTable(jetColor)
	case ColormapHot:
		return buildTable(hotColor)
	case ColormapCool:
		return buildTable(coolColor)
	case ColormapBone:
		return buildTable(boneColor)
	case ColormapHSV:
		return buildTable(hsvColor)
	case ColormapViridis:
		return buildTable(anchorColor(viridisAnchors))
	case ColormapPlasma:
		return buildTable(anchorColor(plasmaAnchors))
	case ColormapGrayscale:
		return buildTable(grayColor)
	default:
		panic("plot: ColormapTable unknown colormap")
	}
}

// buildTable samples fn at the 256 intensities to produce a lookup table.
func buildTable(fn func(t float64) (r, g, b float64)) [][3]uint8 {
	table := make([][3]uint8, 256)
	for i := 0; i < 256; i++ {
		t := float64(i) / 255
		r, g, b := fn(t)
		table[i] = [3]uint8{u8(r), u8(g), u8(b)}
	}
	return table
}

// grayColor is the identity grayscale ramp.
func grayColor(t float64) (r, g, b float64) { return t, t, t }

// jetColor is the classic jet rainbow ramp.
func jetColor(t float64) (r, g, b float64) {
	four := 4 * t
	r = clamp01(1.5 - math.Abs(four-3))
	g = clamp01(1.5 - math.Abs(four-2))
	b = clamp01(1.5 - math.Abs(four-1))
	return r, g, b
}

// hotColor ramps black -> red -> yellow -> white.
func hotColor(t float64) (r, g, b float64) {
	return clamp01(3 * t), clamp01(3*t - 1), clamp01(3*t - 2)
}

// coolColor ramps cyan -> magenta.
func coolColor(t float64) (r, g, b float64) { return t, 1 - t, 1 }

// boneColor is grayscale mixed with a blue-tinted hot ramp (MATLAB's bone).
func boneColor(t float64) (r, g, b float64) {
	r = (7*t + clamp01(3*t-2)) / 8
	g = (7*t + clamp01(3*t-1)) / 8
	b = (7*t + clamp01(3*t)) / 8
	return r, g, b
}

// hsvColor sweeps the hue circle at full saturation and value.
func hsvColor(t float64) (r, g, b float64) {
	h := clamp01(t) * 6 // sector in [0,6]
	i := math.Floor(h)
	f := h - i
	switch int(i) % 6 {
	case 0:
		return 1, f, 0
	case 1:
		return 1 - f, 1, 0
	case 2:
		return 0, 1, f
	case 3:
		return 0, 1 - f, 1
	case 4:
		return f, 0, 1
	default:
		return 1, 0, 1 - f
	}
}

// viridisAnchors and plasmaAnchors are evenly spaced control colours (at
// fractions 0, 1/4, 1/2, 3/4, 1) that are linearly interpolated to build the
// perceptually uniform maps. The endpoints are reproduced exactly.
var viridisAnchors = [][3]float64{
	{68, 1, 84},
	{59, 82, 139},
	{33, 145, 140},
	{94, 201, 98},
	{253, 231, 37},
}

var plasmaAnchors = [][3]float64{
	{13, 8, 135},
	{126, 3, 168},
	{204, 71, 120},
	{248, 149, 64},
	{240, 249, 33},
}

// anchorColor returns a ramp function that linearly interpolates the given
// 0..255 anchor colours across t in [0,1].
func anchorColor(anchors [][3]float64) func(t float64) (r, g, b float64) {
	segs := len(anchors) - 1
	return func(t float64) (r, g, b float64) {
		t = clamp01(t)
		pos := t * float64(segs)
		i := int(math.Floor(pos))
		if i >= segs {
			i = segs - 1
		}
		f := pos - float64(i)
		a := anchors[i]
		bb := anchors[i+1]
		lerp := func(x, y float64) float64 { return (x + (y-x)*f) / 255 }
		return lerp(a[0], bb[0]), lerp(a[1], bb[1]), lerp(a[2], bb[2])
	}
}

// ApplyColorMap recolours a single-channel image through the given built-in
// colormap and returns a fresh three-channel RGB [cv.Mat]: output pixel (y,x)
// takes the colormap entry indexed by the input intensity at (y,x). It panics
// if gray is not single-channel or cm is unknown.
//
// The colour convention is RGB (channel 0 red, 1 green, 2 blue); OpenCV's
// applyColorMap returns BGR instead.
func ApplyColorMap(gray *cv.Mat, cm Colormap) *cv.Mat {
	return ApplyCustomColorMap(gray, ColormapTable(cm))
}

// ApplyCustomColorMap recolours a single-channel image through a caller-supplied
// 256-entry RGB table and returns a fresh three-channel [cv.Mat]. It panics if
// gray is not single-channel or table does not have exactly 256 entries.
func ApplyCustomColorMap(gray *cv.Mat, table [][3]uint8) *cv.Mat {
	if gray.Channels != 1 {
		panic("plot: ApplyCustomColorMap requires a single-channel image")
	}
	if len(table) != 256 {
		panic("plot: ApplyCustomColorMap requires a 256-entry table")
	}
	dst := cv.NewMat(gray.Rows, gray.Cols, 3)
	for p := 0; p < gray.Total(); p++ {
		c := table[gray.Data[p]]
		dst.Data[p*3+0] = c[0]
		dst.Data[p*3+1] = c[1]
		dst.Data[p*3+2] = c[2]
	}
	return dst
}

// LUT applies a per-sample lookup table to every sample of img and returns a
// fresh [cv.Mat] of the same shape, mirroring cv2.LUT: output sample s becomes
// table[s]. The table must have exactly 256 entries and is applied identically
// to every channel. It panics if table is not 256 entries long.
func LUT(img *cv.Mat, table []uint8) *cv.Mat {
	if len(table) != 256 {
		panic("plot: LUT requires a 256-entry table")
	}
	dst := cv.NewMat(img.Rows, img.Cols, img.Channels)
	for i, s := range img.Data {
		dst.Data[i] = table[s]
	}
	return dst
}
