package plot

import cv "github.com/malcolmston/opencv"

// The additional built-in colormaps, continuing the [Colormap] enumeration after
// [ColormapGrayscale]. They round out the set of OpenCV COLORMAP_* tables that
// were not already provided by [ColormapTable]. Apply any of them with
// [Colorize] or fetch their lookup table with [Table]; the plain [ApplyColorMap]
// only knows the original eight.
const (
	// ColormapAutumn ramps red to yellow: 0 is red (255,0,0) and 255 is yellow
	// (255,255,0).
	ColormapAutumn Colormap = iota + 8
	// ColormapWinter ramps blue to spring-green: 0 is blue (0,0,255) and 255 is
	// (0,255,128).
	ColormapWinter
	// ColormapSummer ramps green to yellow at constant low blue: 0 is (0,128,102)
	// and 255 is (255,255,102).
	ColormapSummer
	// ColormapSpring ramps magenta to yellow: 0 is magenta (255,0,255) and 255 is
	// yellow (255,255,0).
	ColormapSpring
	// ColormapOcean ramps dark teal to white through blues: 0 is (0,32,64) and
	// 255 is white (255,255,255).
	ColormapOcean
	// ColormapRainbow sweeps the spectrum red to magenta: 0 is red (255,0,0) and
	// 255 is magenta (255,0,255).
	ColormapRainbow
	// ColormapPink is a warm sepia ramp: 0 is (30,15,15) and 255 is white
	// (255,255,255).
	ColormapPink
	// ColormapParula is MATLAB's parula: 0 is dark blue (53,42,135) and 255 is
	// yellow (249,251,21).
	ColormapParula
	// ColormapMagma is the perceptually uniform magma map: 0 is near-black
	// (0,0,4) and 255 is pale yellow (252,253,191).
	ColormapMagma
	// ColormapInferno is the perceptually uniform inferno map: 0 is near-black
	// (0,0,4) and 255 is pale yellow (252,255,164).
	ColormapInferno
	// ColormapCividis is the colour-vision-deficiency-friendly cividis map: 0 is
	// dark blue (0,32,76) and 255 is yellow (255,233,69).
	ColormapCividis
	// ColormapTwilight is the cyclic twilight map: both endpoints are the same
	// pale lavender (226,217,226).
	ColormapTwilight
	// ColormapTurbo is Google's turbo rainbow: 0 is dark indigo (48,18,59) and
	// 255 is dark red (122,4,3).
	ColormapTurbo
)

// autumnColor ramps red to yellow.
func autumnColor(t float64) (r, g, b float64) { return 1, t, 0 }

// winterColor ramps blue to spring-green.
func winterColor(t float64) (r, g, b float64) { return 0, t, 1 - t/2 }

// summerColor ramps green to yellow at constant low blue.
func summerColor(t float64) (r, g, b float64) { return t, 0.5 + 0.5*t, 0.4 }

// springColor ramps magenta to yellow.
func springColor(t float64) (r, g, b float64) { return 1, t, 1 - t }

// rainbowColor sweeps the visible spectrum from red to magenta by reusing the
// hue circle over five of its six sectors.
func rainbowColor(t float64) (r, g, b float64) { return hsvColor(clamp01(t) * 5.0 / 6.0) }

// oceanAnchors, pinkAnchors and the perceptual anchor sets are evenly spaced
// control colours (0..255) linearly interpolated by anchorColor.
var oceanAnchors = [][3]float64{
	{0, 32, 64}, {0, 64, 128}, {0, 128, 160}, {100, 180, 200}, {255, 255, 255},
}

var pinkAnchors = [][3]float64{
	{30, 15, 15}, {150, 110, 110}, {200, 170, 150}, {230, 215, 190}, {255, 255, 255},
}

var parulaAnchors = [][3]float64{
	{53, 42, 135}, {4, 130, 200}, {57, 178, 116}, {210, 190, 60}, {249, 251, 21},
}

var magmaAnchors = [][3]float64{
	{0, 0, 4}, {81, 18, 124}, {183, 55, 121}, {252, 137, 97}, {252, 253, 191},
}

var infernoAnchors = [][3]float64{
	{0, 0, 4}, {87, 16, 110}, {188, 55, 84}, {249, 142, 9}, {252, 255, 164},
}

var cividisAnchors = [][3]float64{
	{0, 32, 76}, {45, 73, 110}, {124, 123, 120}, {190, 183, 96}, {255, 233, 69},
}

var twilightAnchors = [][3]float64{
	{226, 217, 226}, {58, 79, 164}, {17, 32, 60}, {170, 50, 80}, {226, 217, 226},
}

var turboAnchors = [][3]float64{
	{48, 18, 59}, {62, 155, 254}, {48, 222, 139}, {225, 220, 55}, {122, 4, 3},
}

// Table returns the 256-entry RGB lookup table for any built-in [Colormap],
// including both the original maps served by [ColormapTable] and the additional
// maps declared in this file (autumn, winter, summer, spring, ocean, rainbow,
// pink, parula, magma, inferno, cividis, twilight and turbo). Entry i is the
// [3]uint8 colour for intensity i. It panics if cm is not a recognised colormap.
func Table(cm Colormap) [][3]uint8 {
	switch cm {
	case ColormapAutumn:
		return buildTable(autumnColor)
	case ColormapWinter:
		return buildTable(winterColor)
	case ColormapSummer:
		return buildTable(summerColor)
	case ColormapSpring:
		return buildTable(springColor)
	case ColormapOcean:
		return buildTable(anchorColor(oceanAnchors))
	case ColormapRainbow:
		return buildTable(rainbowColor)
	case ColormapPink:
		return buildTable(anchorColor(pinkAnchors))
	case ColormapParula:
		return buildTable(anchorColor(parulaAnchors))
	case ColormapMagma:
		return buildTable(anchorColor(magmaAnchors))
	case ColormapInferno:
		return buildTable(anchorColor(infernoAnchors))
	case ColormapCividis:
		return buildTable(anchorColor(cividisAnchors))
	case ColormapTwilight:
		return buildTable(anchorColor(twilightAnchors))
	case ColormapTurbo:
		return buildTable(anchorColor(turboAnchors))
	default:
		// Delegates to the original eight (and panics on a truly unknown value).
		return ColormapTable(cm)
	}
}

// Colorize recolours a single-channel image through any built-in [Colormap]
// (original or additional) and returns a fresh three-channel RGB [cv.Mat]. It is
// the general-purpose counterpart to [ApplyColorMap], which only recognises the
// original eight maps. It panics if gray is not single-channel or cm is unknown.
func Colorize(gray *cv.Mat, cm Colormap) *cv.Mat {
	return ApplyCustomColorMap(gray, Table(cm))
}
