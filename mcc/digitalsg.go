package mcc

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// DigitalSG geometry: the X-Rite ColorChecker Digital SG is a 10-row by
// 14-column array of 140 patches.
const (
	sgRows    = 10
	sgCols    = 14
	sgPatches = sgRows * sgCols // 140
)

// sgChart holds the assembled DigitalSG reference patches, built once at
// package initialisation.
var sgChart []Patch

// sgRGB and sgNames hold the DigitalSG reference sRGB values and patch names in
// row-major order (patch index = row*sgCols + col).
var (
	sgRGB   [sgPatches][3]uint8
	sgNames [sgPatches]string
)

func init() {
	buildDigitalSG()
	sgChart = make([]Patch, sgPatches)
	for i := range sgRGB {
		sgChart[i] = Patch{
			Name: sgNames[i],
			RGB:  sgRGB[i],
			Lab:  RGBToLab(sgRGB[i][0], sgRGB[i][1], sgRGB[i][2]),
		}
	}
}

// buildDigitalSG populates the 140-patch DigitalSG reference. It reproduces the
// real 10x14 geometry with a neutral gray frame around the border, the classic
// 24-patch GretagMacbeth ColorChecker embedded as a contiguous 4x6 block (using
// the same values as [Macbeth24], so the embedded colors are anchored to real
// reference data), and a deterministic Lab-space sweep of saturated hues filling
// the remaining interior cells to span the color gamut. Every value is a real,
// usable color; the reference is intended for detection, rendering and
// correction tests rather than as a substitute for a factory-measured SG chart.
func buildDigitalSG() {
	for idx := 0; idx < sgPatches; idx++ {
		r := idx / sgCols
		c := idx % sgCols
		if r == 0 || r == sgRows-1 || c == 0 || c == sgCols-1 {
			// Neutral border frame: a repeating 9-step gray ramp, with the four
			// physical corners forced to white.
			t := float64((r*sgCols+c)%9) / 8
			g := clampToUint8(30 + t*215)
			sgRGB[idx] = [3]uint8{g, g, g}
			sgNames[idx] = fmt.Sprintf("frame gray (%d,%d)", r, c)
			continue
		}
		// Interior sweep: vary lightness with the row and hue with the column,
		// generating in-gamut saturated colors via Lab->sRGB.
		l := 35 + 50*float64(r-1)/float64(sgRows-3)
		hue := 360 * float64(c-1) / float64(sgCols-2)
		chroma := 45.0
		lab := [3]float64{l, chroma * math.Cos(hue*math.Pi/180), chroma * math.Sin(hue*math.Pi/180)}
		rgb := LabToRGB(lab, WhiteD65)
		sgRGB[idx] = rgb
		sgNames[idx] = fmt.Sprintf("sweep (%d,%d)", r, c)
	}
	// Force the four corners to white.
	for _, idx := range []int{0, sgCols - 1, (sgRows - 1) * sgCols, sgRows*sgCols - 1} {
		sgRGB[idx] = [3]uint8{245, 245, 245}
		sgNames[idx] = "frame white"
	}
	// Embed the classic 24-patch ColorChecker as a 4x6 block at rows 3..6,
	// columns 4..9 (entirely inside the interior region).
	for i := 0; i < 24; i++ {
		rr := 3 + i/6
		cc := 4 + i%6
		idx := rr*sgCols + cc
		sgRGB[idx] = macbethRGB[i]
		sgNames[idx] = "classic " + patchNames[i]
	}
}

// DigitalSGRows returns the number of patch rows in the ColorChecker Digital SG
// chart (10).
func DigitalSGRows() int { return sgRows }

// DigitalSGCols returns the number of patch columns in the ColorChecker Digital
// SG chart (14).
func DigitalSGCols() int { return sgCols }

// DigitalSGNumPatches returns the number of patches in the ColorChecker Digital
// SG chart (140).
func DigitalSGNumPatches() int { return sgPatches }

// DigitalSGReference returns the 140 reference patches of the ColorChecker
// Digital SG chart in row-major grid order (patch index = row*14 + col). The
// returned slice is a fresh copy that callers may modify freely. See
// [buildDigitalSG] for how the reference is constructed.
func DigitalSGReference() []Patch {
	out := make([]Patch, len(sgChart))
	copy(out, sgChart)
	return out
}

// DigitalSGReferenceRGB returns the DigitalSG reference sRGB values as float64
// triples in the 0..255 range and grid order, ready to feed to
// [TrainColorCorrection] as the target colors.
func DigitalSGReferenceRGB() [][3]float64 {
	out := make([][3]float64, len(sgChart))
	for i, p := range sgChart {
		out[i] = [3]float64{float64(p.RGB[0]), float64(p.RGB[1]), float64(p.RGB[2])}
	}
	return out
}

// DigitalSGReferenceLab returns the DigitalSG reference CIE L*a*b* (D65) values
// in grid order.
func DigitalSGReferenceLab() [][3]float64 {
	out := make([][3]float64, len(sgChart))
	for i, p := range sgChart {
		out[i] = p.Lab
	}
	return out
}

// RenderSGChart synthesizes a canonical image of the ColorChecker Digital SG
// chart, drawing each of the 140 patches as a solid patchPx-by-patchPx square
// separated by gapPx-wide black gaps, exactly as [RenderChart] does for the
// 24-patch charts. It panics if patchPx or gapPx is not positive.
func RenderSGChart(patchPx, gapPx int) *cv.Mat {
	if patchPx <= 0 || gapPx <= 0 {
		panic("mcc: RenderSGChart requires positive patchPx and gapPx")
	}
	w := sgCols*patchPx + (sgCols+1)*gapPx
	h := sgRows*patchPx + (sgRows+1)*gapPx
	img := cv.NewMat(h, w, 3)
	for r := 0; r < sgRows; r++ {
		for c := 0; c < sgCols; c++ {
			p := sgChart[r*sgCols+c]
			x0 := gapPx + c*(patchPx+gapPx)
			y0 := gapPx + r*(patchPx+gapPx)
			for y := 0; y < patchPx; y++ {
				for x := 0; x < patchPx; x++ {
					img.SetPixel(y0+y, x0+x, []uint8{p.RGB[0], p.RGB[1], p.RGB[2]})
				}
			}
		}
	}
	return img
}

// DigitalSGOuterQuad returns the four outer corners of the 140-patch array in an
// image rendered by [RenderSGChart] with the same parameters, ordered top-left,
// top-right, bottom-right, bottom-left — the quad to pass to [SampleSGWithHint].
func DigitalSGOuterQuad(patchPx, gapPx int) [4]cv.Point {
	x0 := gapPx
	y0 := gapPx
	x1 := gapPx + sgCols*patchPx + (sgCols-1)*gapPx - 1
	y1 := gapPx + sgRows*patchPx + (sgRows-1)*gapPx - 1
	return [4]cv.Point{
		{X: x0, Y: y0},
		{X: x1, Y: y0},
		{X: x1, Y: y1},
		{X: x0, Y: y1},
	}
}

// CCheckerSG is a sampled ColorChecker Digital SG chart: the four image points
// framing the 140-patch array and the measured sRGB color of every patch in
// grid order, mirroring [CChecker] for the larger chart.
type CCheckerSG struct {
	// Corners are the four image points the chart was sampled from, ordered
	// top-left, top-right, bottom-right, bottom-left.
	Corners [4]cv.Point
	// Measured holds the measured sRGB color of each of the 140 patches in grid
	// order (patch index = row*14 + col).
	Measured [][3]float64
}

// SampleSGWithHint samples a ColorChecker Digital SG chart directly from the
// four outer corners of its patch array, supplied top-left, top-right,
// bottom-right, bottom-left. Like [CCheckerDetector.DetectWithHint] it skips any
// search and is the robust path when the chart's location is already known. img
// must be a usable Mat (RGB, or single-channel which is promoted). It returns
// false only if img is unusable.
func SampleSGWithHint(img *cv.Mat, outerQuad [4]cv.Point) (*CCheckerSG, bool) {
	if img == nil || img.Empty() {
		return nil, false
	}
	rgb := toRGB(img)
	src := [4]cv.Point{
		{X: 0, Y: 0},
		{X: sgCols * canonS, Y: 0},
		{X: sgCols * canonS, Y: sgRows * canonS},
		{X: 0, Y: sgRows * canonS},
	}
	h := cv.GetPerspectiveTransform(src, outerQuad)
	measured := sampleChart(rgb, h, sgRows, sgCols)
	return &CCheckerSG{Corners: outerQuad, Measured: measured}, true
}

// MeasuredRGB returns a copy of the measured per-patch colors in grid order.
func (c *CCheckerSG) MeasuredRGB() [][3]float64 {
	out := make([][3]float64, len(c.Measured))
	copy(out, c.Measured)
	return out
}

// Reference returns the DigitalSG reference patches.
func (c *CCheckerSG) Reference() []Patch { return DigitalSGReference() }

// PatchErrors returns the CIE76 Delta E between each measured patch color and
// its DigitalSG reference, in grid order.
func (c *CCheckerSG) PatchErrors() []float64 {
	out := make([]float64, len(c.Measured))
	for i, m := range c.Measured {
		out[i] = DeltaE76(rgbToLabF(m[0], m[1], m[2]), sgChart[i].Lab)
	}
	return out
}

// MeanError returns the mean of [CCheckerSG.PatchErrors].
func (c *CCheckerSG) MeanError() float64 { return mean(c.PatchErrors()) }

// MaxError returns the largest of [CCheckerSG.PatchErrors].
func (c *CCheckerSG) MaxError() float64 {
	m := 0.0
	for _, e := range c.PatchErrors() {
		if e > m {
			m = e
		}
	}
	return m
}
