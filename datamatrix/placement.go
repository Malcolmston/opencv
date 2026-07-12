package datamatrix

// This file implements the ECC200 "utah" module-placement algorithm from
// ISO/IEC 16022 Annex F. Rather than placing bits directly, buildPlacement
// returns, for every codeword, the eight (row, column) coordinates in the
// mapping matrix that hold its eight bits (most-significant bit first). The
// encoder writes bits into those coordinates and the decoder reads them back,
// so the two directions are guaranteed to be exact inverses.

// bitPos identifies a single module position in the mapping matrix.
type bitPos struct {
	Row, Col int
}

// placement is the result of running the ECC200 placement algorithm for a
// particular mapping-matrix size.
type placement struct {
	nrow, ncol int
	// pos[k] holds the eight module coordinates of codeword k, ordered from
	// bit 1 (most significant, value 0x80) to bit 8 (least significant).
	pos [][8]bitPos
	// fixed lists module coordinates that carry the fixed corner pattern
	// (always dark) used when the mapping matrix is not a multiple of eight.
	fixed []bitPos
}

// buildPlacement runs the placement algorithm for a mapping matrix of nrow x
// ncol modules that stores totalCW codewords.
func buildPlacement(nrow, ncol, totalCW int) placement {
	p := placement{
		nrow: nrow,
		ncol: ncol,
		pos:  make([][8]bitPos, totalCW),
	}
	occupied := make([]bool, nrow*ncol)

	// module records the coordinate of bit (1..8) of codeword chr (1-based),
	// applying the wrap-around rules from the specification.
	module := func(row, col, chr, bit int) {
		if row < 0 {
			row += nrow
			col += 4 - ((nrow + 4) % 8)
		}
		if col < 0 {
			col += ncol
			row += 4 - ((ncol + 4) % 8)
		}
		occupied[row*ncol+col] = true
		p.pos[chr-1][bit-1] = bitPos{Row: row, Col: col}
	}

	utah := func(row, col, chr int) {
		module(row-2, col-2, chr, 1)
		module(row-2, col-1, chr, 2)
		module(row-1, col-2, chr, 3)
		module(row-1, col-1, chr, 4)
		module(row-1, col, chr, 5)
		module(row, col-2, chr, 6)
		module(row, col-1, chr, 7)
		module(row, col, chr, 8)
	}

	corner1 := func(chr int) {
		module(nrow-1, 0, chr, 1)
		module(nrow-1, 1, chr, 2)
		module(nrow-1, 2, chr, 3)
		module(0, ncol-2, chr, 4)
		module(0, ncol-1, chr, 5)
		module(1, ncol-1, chr, 6)
		module(2, ncol-1, chr, 7)
		module(3, ncol-1, chr, 8)
	}

	corner2 := func(chr int) {
		module(nrow-3, 0, chr, 1)
		module(nrow-2, 0, chr, 2)
		module(nrow-1, 0, chr, 3)
		module(0, ncol-4, chr, 4)
		module(0, ncol-3, chr, 5)
		module(0, ncol-2, chr, 6)
		module(0, ncol-1, chr, 7)
		module(1, ncol-1, chr, 8)
	}

	corner3 := func(chr int) {
		module(nrow-3, 0, chr, 1)
		module(nrow-2, 0, chr, 2)
		module(nrow-1, 0, chr, 3)
		module(0, ncol-2, chr, 4)
		module(0, ncol-1, chr, 5)
		module(1, ncol-1, chr, 6)
		module(2, ncol-1, chr, 7)
		module(3, ncol-1, chr, 8)
	}

	corner4 := func(chr int) {
		module(nrow-1, 0, chr, 1)
		module(nrow-1, ncol-1, chr, 2)
		module(0, ncol-3, chr, 3)
		module(0, ncol-2, chr, 4)
		module(0, ncol-1, chr, 5)
		module(1, ncol-3, chr, 6)
		module(1, ncol-2, chr, 7)
		module(1, ncol-1, chr, 8)
	}

	chr := 1
	row, col := 4, 0
	for {
		// Special corner cases.
		if row == nrow && col == 0 {
			corner1(chr)
			chr++
		}
		if row == nrow-2 && col == 0 && ncol%4 != 0 {
			corner2(chr)
			chr++
		}
		if row == nrow-2 && col == 0 && ncol%8 == 4 {
			corner3(chr)
			chr++
		}
		if row == nrow+4 && col == 2 && ncol%8 == 0 {
			corner4(chr)
			chr++
		}
		// Sweep upward diagonally.
		for {
			if row < nrow && col >= 0 && !occupied[row*ncol+col] {
				utah(row, col, chr)
				chr++
			}
			row -= 2
			col += 2
			if row < 0 || col >= ncol {
				break
			}
		}
		row++
		col += 3
		// Sweep downward diagonally.
		for {
			if row >= 0 && col < ncol && !occupied[row*ncol+col] {
				utah(row, col, chr)
				chr++
			}
			row += 2
			col -= 2
			if row >= nrow || col < 0 {
				break
			}
		}
		row += 3
		col++
		if row >= nrow && col >= ncol {
			break
		}
	}

	// Fixed pattern in the bottom-right corner when the last module is unused.
	if !occupied[nrow*ncol-1] {
		occupied[nrow*ncol-1] = true
		occupied[nrow*ncol-ncol-2] = true
		p.fixed = []bitPos{
			{Row: nrow - 1, Col: ncol - 1},
			{Row: nrow - 2, Col: ncol - 2},
		}
	}

	return p
}
