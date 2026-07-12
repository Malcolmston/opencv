package datamatrix

import cv "github.com/malcolmston/opencv"

// This file lays a full interleaved codeword stream into the physical module
// grid of any supported symbol (square or rectangular, single- or multi-region)
// and reads it back. The ECC200 "utah" bit placement (buildPlacement) operates
// on the logical mapping matrix that spans every data region; mapToModule then
// tiles that logical grid into the physical symbol, inserting the two
// finder/timing modules that frame each data region.

// EncodedSymbol is a fully laid-out ECC200 symbol of any supported size. Unlike
// the square-only [Symbol], it carries independent row and column counts so it
// can represent rectangular symbols. Modules[r][c] is true where a module is
// dark; row 0 is the top edge and column 0 the left edge.
type EncodedSymbol struct {
	// Rows is the symbol height in modules, finder patterns included.
	Rows int
	// Cols is the symbol width in modules, finder patterns included.
	Cols int
	// SizeName is the symbol size as "rowsxcols", e.g. "16x48".
	SizeName string
	// Modules holds the module colours: true = dark.
	Modules [][]bool
}

// newGrid allocates a blank module grid of the given dimensions.
func newGrid(rows, cols int) [][]bool {
	g := make([][]bool, rows)
	for r := range g {
		g[r] = make([]bool, cols)
	}
	return g
}

// drawRegionFinders draws the solid "L" (left column and bottom row) and the
// alternating timing pattern (top row and right column) that frames every data
// region of the symbol.
func drawRegionFinders(grid [][]bool, info symbolInfo) {
	for rr := 0; rr < info.vertRegions(); rr++ {
		for rc := 0; rc < info.horizRegions(); rc++ {
			row0 := rr * (info.regH + 2)
			col0 := rc * (info.regW + 2)
			h := info.regH + 2
			w := info.regW + 2
			for i := 0; i < h; i++ {
				grid[row0+i][col0] = true         // left solid
				grid[row0+i][col0+w-1] = i%2 == 1 // right timing
			}
			for j := 0; j < w; j++ {
				grid[row0+h-1][col0+j] = true // bottom solid
				grid[row0][col0+j] = j%2 == 0 // top timing
			}
		}
	}
}

// mapToModule converts a logical mapping-matrix coordinate to the physical
// module coordinate for the given symbol, accounting for the per-region
// finder/timing frame.
func mapToModule(info symbolInfo, mr, mc int) (row, col int) {
	regRow := mr / info.regH
	regCol := mc / info.regW
	dr := mr % info.regH
	dc := mc % info.regW
	row = regRow*(info.regH+2) + 1 + dr
	col = regCol*(info.regW+2) + 1 + dc
	return row, col
}

// placeCodewords lays the full codeword stream (length info.totalCW) into a new
// physical module grid and returns it.
func placeCodewords(info symbolInfo, full []int) [][]bool {
	grid := newGrid(info.symbolRows(), info.symbolCols())
	drawRegionFinders(grid, info)
	pl := buildPlacement(info.mapRows(), info.mapCols(), info.totalCW())
	for k, v := range full {
		for b := 0; b < 8; b++ {
			bit := (v >> (7 - b)) & 1
			p := pl.pos[k][b]
			row, col := mapToModule(info, p.Row, p.Col)
			grid[row][col] = bit == 1
		}
	}
	for _, f := range pl.fixed {
		row, col := mapToModule(info, f.Row, f.Col)
		grid[row][col] = true
	}
	return grid
}

// readCodewordStream reads the full codeword stream out of a physical module
// grid using the same placement.
func readCodewordStream(info symbolInfo, grid [][]bool) []int {
	pl := buildPlacement(info.mapRows(), info.mapCols(), info.totalCW())
	full := make([]int, info.totalCW())
	for k := 0; k < info.totalCW(); k++ {
		v := 0
		for b := 0; b < 8; b++ {
			p := pl.pos[k][b]
			row, col := mapToModule(info, p.Row, p.Col)
			v <<= 1
			if grid[row][col] {
				v |= 1
			}
		}
		full[k] = v
	}
	return full
}

// Render rasterises the symbol into a Mat according to opts, matching the
// conventions of [EncodeWithOptions]: dark modules are black on a white
// background, surrounded by the requested quiet zone.
func (e *EncodedSymbol) Render(opts Options) *cv.Mat {
	opts = opts.normalized()
	mp := opts.ModulePixels
	qz := opts.QuietZoneModules
	ch := opts.Channels
	h := (e.Rows + 2*qz) * mp
	w := (e.Cols + 2*qz) * mp
	m := cv.NewMat(h, w, ch)
	m.SetTo(255)
	for r := 0; r < e.Rows; r++ {
		for c := 0; c < e.Cols; c++ {
			if !e.Modules[r][c] {
				continue
			}
			y0 := (r + qz) * mp
			x0 := (c + qz) * mp
			for dy := 0; dy < mp; dy++ {
				for dx := 0; dx < mp; dx++ {
					base := ((y0+dy)*m.Cols + (x0 + dx)) * ch
					for cc := 0; cc < ch; cc++ {
						m.Data[base+cc] = 0
					}
				}
			}
		}
	}
	return m
}

// finderMatchesLayout reports whether grid carries the correct finder and
// timing pattern for every data region of info.
func finderMatchesLayout(grid [][]bool, info symbolInfo) bool {
	if len(grid) != info.symbolRows() {
		return false
	}
	for _, row := range grid {
		if len(row) != info.symbolCols() {
			return false
		}
	}
	for rr := 0; rr < info.vertRegions(); rr++ {
		for rc := 0; rc < info.horizRegions(); rc++ {
			row0 := rr * (info.regH + 2)
			col0 := rc * (info.regW + 2)
			h := info.regH + 2
			w := info.regW + 2
			for i := 0; i < h; i++ {
				if !grid[row0+i][col0] {
					return false
				}
				if grid[row0+i][col0+w-1] != (i%2 == 1) {
					return false
				}
			}
			for j := 0; j < w; j++ {
				if !grid[row0+h-1][col0+j] {
					return false
				}
				if grid[row0][col0+j] != (j%2 == 0) {
					return false
				}
			}
		}
	}
	return true
}
