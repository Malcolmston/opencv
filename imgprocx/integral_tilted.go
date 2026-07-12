package imgprocx

import cv "github.com/malcolmston/opencv"

// IntegralTilted computes the tilted (45°-rotated) summed-area table of img,
// mirroring the tilted output of cv::integral. The returned table has size
// (Rows+1)×(Cols+1) and follows OpenCV's definition
//
//	tilted[Y][X] = Σ intensity(y, x)   over y < Y with |x - X + 1| <= Y - y - 1,
//
// where a pixel's intensity is the sum of its channel samples (as in
// [IntegralImage]). Each cell is thus the sum over an upward-pointing 45° triangle
// whose apex sits just above (X-1, Y-1). Reading a cell is O(1) via [TiltedSum].
//
// It is filled in a single pass with the recurrence
//
//	tilted[Y][X] = tilted[Y-1][X-1] + tilted[Y-1][X+1] - tilted[Y-2][X]
//	             + intensity(Y-1, X-1) + intensity(Y-2, X-1),
//
// treating out-of-range table cells and pixels as zero. Tilted integral images
// let Haar-like features be evaluated at 45° orientations in constant time.
func IntegralTilted(img *cv.Mat) [][]float64 {
	rows, cols := img.Rows, img.Cols
	// A triangle apexed at column X-1 can cover in-bounds pixels for X anywhere
	// in [2-rows, cols+rows-1], so the working table is padded by `off` columns
	// on each side; internal index xi maps to image column x = xi - off. Only
	// columns x in 0..cols are returned.
	off := rows + 1
	wide := cols + 1 + 2*off
	work := make([][]float64, rows+1)
	for i := range work {
		work[i] = make([]float64, wide)
	}
	// tAt reads the working table by internal index, 0 outside the buffer.
	tAt := func(y, xi int) float64 {
		if y < 0 || xi < 0 || xi >= wide {
			return 0
		}
		return work[y][xi]
	}
	// pAt reads pixel intensity by image column, 0 for out-of-range pixels.
	pAt := func(y, x int) float64 {
		if y < 0 || y >= rows || x < 0 || x >= cols {
			return 0
		}
		return pixelSum(img, y, x)
	}
	for y := 1; y <= rows; y++ {
		for xi := 0; xi < wide; xi++ {
			x := xi - off
			work[y][xi] = tAt(y-1, xi-1) + tAt(y-1, xi+1) - tAt(y-2, xi) +
				pAt(y-1, x-1) + pAt(y-2, x-1)
		}
	}
	tilted := make([][]float64, rows+1)
	for i := range tilted {
		tilted[i] = make([]float64, cols+1)
		copy(tilted[i], work[i][off:off+cols+1])
	}
	return tilted
}

// TiltedSum returns tilted[y][x] from a table produced by [IntegralTilted]: the
// total intensity over the upward 45° triangle with apex just above (x-1, y-1).
// It is a bounds-checked convenience accessor; x and y are table coordinates
// (0 <= x <= Cols, 0 <= y <= Rows) and it panics if they fall outside the table.
func TiltedSum(tilted [][]float64, x, y int) float64 {
	if y < 0 || y >= len(tilted) || x < 0 || x >= len(tilted[0]) {
		panic("imgprocx: TiltedSum coordinates out of range")
	}
	return tilted[y][x]
}
