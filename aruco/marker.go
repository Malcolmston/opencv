package aruco

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// GenerateMarker renders the marker with the given id from dict into a new
// single-channel [cv.Mat] whose side is sidePixels pixels. The image is the
// full marker: a solid black border one cell thick around the id's inner grid,
// with white cells drawn as 255 and black cells as 0.
//
// The marker is drawn by mapping each output pixel to its cell, so any
// sidePixels at least as large as the marker's cell count is accepted and the
// returned Mat is exactly sidePixels by sidePixels. For crisp, evenly sized
// cells choose a sidePixels that is a multiple of BitsPerSide()+2.
//
// It panics if dict is nil, id is out of range, or sidePixels is smaller than
// the marker's cell count (BitsPerSide()+2).
func GenerateMarker(dict *Dictionary, id, sidePixels int) *cv.Mat {
	if dict == nil {
		panic("aruco: GenerateMarker nil dictionary")
	}
	grid := dict.bits(id)
	if grid == nil {
		panic(fmt.Sprintf("aruco: GenerateMarker id %d out of range [0,%d)", id, dict.Size()))
	}
	side := dict.bitsPerSide
	cells := side + 2 // include the one-cell black border on every side
	if sidePixels < cells {
		panic(fmt.Sprintf("aruco: GenerateMarker sidePixels %d too small for a %d-cell marker", sidePixels, cells))
	}

	out := cv.NewMat(sidePixels, sidePixels, 1)
	for y := 0; y < sidePixels; y++ {
		cy := y * cells / sidePixels
		for x := 0; x < sidePixels; x++ {
			cx := x * cells / sidePixels
			if cellIsWhite(grid, side, cy, cx) {
				out.Set(y, x, 0, 255)
			}
			// Border and black cells stay 0 (the Mat is zero-initialised).
		}
	}
	return out
}

// cellIsWhite reports whether cell (row, col) of the full marker grid is white.
// row and col index the bordered grid, so the outer ring (row/col 0 or cells-1)
// is always black, and interior cells consult the inner bit grid.
func cellIsWhite(grid []byte, side, row, col int) bool {
	ir := row - 1
	ic := col - 1
	if ir < 0 || ic < 0 || ir >= side || ic >= side {
		return false // border cell
	}
	return grid[ir*side+ic] == 1
}
