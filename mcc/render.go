package mcc

import cv "github.com/malcolmston/opencv"

// RenderChart synthesizes a canonical image of the given chart. Each of the
// chart's patches is drawn as a solid patchPx-by-patchPx square filled with its
// reference sRGB color; squares are separated from each other, and from the
// image edge, by a gapPx-wide black gap. The result is a three-channel RGB
// [cv.Mat] whose overall size is
//
//	width  = cols*patchPx + (cols+1)*gapPx
//	height = rows*patchPx + (rows+1)*gapPx
//
// The black gaps make patches individually detectable, so a rendered chart is
// valid input for [CCheckerDetector.Detect] as well as convenient ground truth
// for tests. It panics if patchPx or gapPx is not positive.
func RenderChart(t CheckerType, patchPx, gapPx int) *cv.Mat {
	if patchPx <= 0 || gapPx <= 0 {
		panic("mcc: RenderChart requires positive patchPx and gapPx")
	}
	rows, cols := t.Rows(), t.Cols()
	ref := ReferenceChart(t)
	w := cols*patchPx + (cols+1)*gapPx
	h := rows*patchPx + (rows+1)*gapPx
	img := cv.NewMat(h, w, 3)
	// The Mat starts zero-filled (black), which already provides the gaps and
	// border; only the patch interiors need painting.
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			p := ref[r*cols+c]
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

// ChartOuterQuad returns the four outer corners of the patch array in a chart
// rendered by [RenderChart] with the same parameters, ordered top-left,
// top-right, bottom-right, bottom-left. It is the quad to pass to
// [CCheckerDetector.DetectWithHint] for such an image, and a building block for
// tests that warp a rendered chart by a known homography.
func ChartOuterQuad(t CheckerType, patchPx, gapPx int) [4]cv.Point {
	rows, cols := t.Rows(), t.Cols()
	x0 := gapPx
	y0 := gapPx
	x1 := gapPx + cols*patchPx + (cols-1)*gapPx
	y1 := gapPx + rows*patchPx + (rows-1)*gapPx
	// x1,y1 are the exclusive right/bottom edges of the patch array; step back
	// one pixel to land on the last patch pixel.
	x1--
	y1--
	return [4]cv.Point{
		{X: x0, Y: y0},
		{X: x1, Y: y0},
		{X: x1, Y: y1},
		{X: x0, Y: y1},
	}
}
