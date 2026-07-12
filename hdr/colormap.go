package hdr

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ColorMap selects a pseudo-colour palette for visualising a scalar field such
// as a radiance or luminance map, mirroring OpenCV's applyColorMap. Each map is
// a function from a normalised value t in [0,1] to an RGB triple.
type ColorMap int

const (
	// ColorMapGray is a plain grayscale ramp.
	ColorMapGray ColorMap = iota
	// ColorMapJet is the classic blue-cyan-yellow-red rainbow.
	ColorMapJet
	// ColorMapHot is a black-red-yellow-white heat ramp.
	ColorMapHot
	// ColorMapInferno is a perceptually smoother black-purple-orange-yellow ramp.
	ColorMapInferno
)

// colorMapLookup returns the RGB triple (each in [0,255]) for a normalised value
// t in [0,1] under the given colour map.
func colorMapLookup(cm ColorMap, t float64) (r, g, b uint8) {
	t = clamp01(t)
	switch cm {
	case ColorMapJet:
		return jetColor(t)
	case ColorMapHot:
		return hotColor(t)
	case ColorMapInferno:
		return infernoColor(t)
	default:
		v := clamp8(t * 255)
		return v, v, v
	}
}

// jetColor implements the standard "jet" rainbow ramp.
func jetColor(t float64) (uint8, uint8, uint8) {
	// Piecewise-linear red/green/blue channels of the jet colormap.
	r := clamp01(1.5 - math.Abs(4*t-3))
	g := clamp01(1.5 - math.Abs(4*t-2))
	b := clamp01(1.5 - math.Abs(4*t-1))
	return clamp8(r * 255), clamp8(g * 255), clamp8(b * 255)
}

// hotColor implements the black-red-yellow-white heat ramp.
func hotColor(t float64) (uint8, uint8, uint8) {
	r := clamp01(t / 0.375)
	g := clamp01((t - 0.375) / 0.375)
	b := clamp01((t - 0.75) / 0.25)
	return clamp8(r * 255), clamp8(g * 255), clamp8(b * 255)
}

// infernoColor is a smooth polynomial approximation of the inferno palette.
func infernoColor(t float64) (uint8, uint8, uint8) {
	// Simple cubic-ish approximation: dark purple to bright yellow.
	r := clamp01(-0.2 + 2.2*t - 0.7*t*t)
	g := clamp01(-0.1 + 0.3*t + 0.8*t*t)
	b := clamp01(0.3 + 1.6*t - 3.0*t*t + 1.3*t*t*t)
	return clamp8(r * 255), clamp8(g * 255), clamp8(b * 255)
}

// ApplyColorMap turns a single-channel [cv.FloatMat] into a three-channel 8-bit
// pseudo-colour image. The field is linearly normalised from its own min/max
// onto [0,1] before the palette is applied, so the full colour range is always
// used. A constant field maps entirely to the palette's low end.
func ApplyColorMap(f *cv.FloatMat, cm ColorMap) *cv.Mat {
	if f == nil || len(f.Data) == 0 {
		panic("hdr: ApplyColorMap on nil or empty FloatMat")
	}
	minV, maxV := math.Inf(1), math.Inf(-1)
	for _, v := range f.Data {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	rng := maxV - minV
	if rng <= 0 {
		rng = 1
	}
	out := cv.NewMat(f.Rows, f.Cols, 3)
	for i, v := range f.Data {
		r, g, b := colorMapLookup(cm, (v-minV)/rng)
		out.Data[i*3+0] = r
		out.Data[i*3+1] = g
		out.Data[i*3+2] = b
	}
	return out
}

// Visualize renders the radiance map's luminance as a false-colour image for
// inspection. The luminance is taken in the log domain (so a wide dynamic range
// is legible), normalised across the image, and mapped through the given
// palette. It is a convenience wrapper over [ApplyColorMap].
func (r *Radiance) Visualize(cm ColorMap) *cv.Mat {
	const eps = 1e-6
	lum := r.luminance()
	logLum := cv.NewFloatMat(r.Rows, r.Cols)
	for i, v := range lum.data {
		if v < 0 {
			v = 0
		}
		logLum.Data[i] = math.Log(v + eps)
	}
	return ApplyColorMap(logLum, cm)
}
