package datamatrix

import "fmt"

// This file adds the full ECC200 symbol-attribute table (ISO/IEC 16022 Annex E)
// used by the extended codec in this package: every square size from 10x10 to
// 132x132 and the six standard rectangular sizes. It complements the small
// single-region symbolSpec table (symbol.go) that the original ASCII-only codec
// uses; the two coexist and describe the same six small square symbols
// identically.
//
// A symbol is divided into one or more identical square-ish data regions, each
// framed by its own finder/timing pattern. The bit-placement "mapping matrix"
// spans every data region as one logical grid; the physical module grid tiles
// those regions with two extra finder modules per region on each axis.

// symbolInfo describes one ECC200 symbol size.
type symbolInfo struct {
	rectangular bool
	dataCW      int // number of data codewords
	eccCW       int // number of error-correction codewords
	regW        int // data-region width in modules (columns), finder excluded
	regH        int // data-region height in modules (rows), finder excluded
	dataRegions int // number of data regions (1, 2, 4, 16 or 36)
	rsBlockData int // data codewords per interleaved Reed-Solomon block
	rsBlockErr  int // error codewords per interleaved Reed-Solomon block
}

// symbolTable lists every supported symbol in ascending data-capacity order.
// Square and rectangular symbols are interleaved by capacity; selection filters
// by the requested shape. Values are taken directly from ISO/IEC 16022.
var symbolTable = []symbolInfo{
	// Square symbols.
	{false, 3, 5, 8, 8, 1, 3, 5},
	{true, 5, 7, 16, 6, 1, 5, 7},            // 8x18
	{false, 5, 7, 10, 10, 1, 5, 7},          // 12x12
	{true, 10, 11, 14, 6, 2, 10, 11},        // 8x32
	{false, 8, 10, 12, 12, 1, 8, 10},        // 14x14
	{true, 16, 14, 24, 10, 1, 16, 14},       // 12x26
	{false, 12, 12, 14, 14, 1, 12, 12},      // 16x16
	{true, 22, 18, 16, 10, 2, 22, 18},       // 12x36
	{false, 18, 14, 16, 16, 1, 18, 14},      // 18x18
	{true, 32, 24, 16, 14, 2, 32, 24},       // 16x36
	{false, 22, 18, 18, 18, 1, 22, 18},      // 20x20
	{true, 49, 28, 22, 14, 2, 49, 28},       // 16x48
	{false, 30, 20, 20, 20, 1, 30, 20},      // 22x22
	{false, 36, 24, 22, 22, 1, 36, 24},      // 24x24
	{false, 44, 28, 24, 24, 1, 44, 28},      // 26x26
	{false, 62, 36, 14, 14, 4, 62, 36},      // 32x32
	{false, 86, 42, 16, 16, 4, 86, 42},      // 36x36
	{false, 114, 48, 18, 18, 4, 114, 48},    // 40x40
	{false, 144, 56, 20, 20, 4, 144, 56},    // 44x44
	{false, 174, 68, 22, 22, 4, 174, 68},    // 48x48
	{false, 204, 84, 24, 24, 4, 102, 42},    // 52x52 (2 blocks)
	{false, 280, 112, 14, 14, 16, 140, 56},  // 64x64 (2 blocks)
	{false, 368, 144, 16, 16, 16, 92, 36},   // 72x72 (4 blocks)
	{false, 456, 192, 18, 18, 16, 114, 48},  // 80x80 (4 blocks)
	{false, 576, 224, 20, 20, 16, 144, 56},  // 88x88 (4 blocks)
	{false, 696, 272, 22, 22, 16, 174, 68},  // 96x96 (4 blocks)
	{false, 816, 336, 24, 24, 16, 136, 56},  // 104x104 (6 blocks)
	{false, 1050, 408, 18, 18, 36, 175, 68}, // 120x120 (6 blocks)
	{false, 1304, 496, 20, 20, 36, 163, 62}, // 132x132 (8 blocks)
}

// horizRegions returns the number of data regions along the horizontal axis.
func (s symbolInfo) horizRegions() int {
	switch s.dataRegions {
	case 1:
		return 1
	case 2, 4:
		return 2
	case 16:
		return 4
	case 36:
		return 6
	default:
		panic(fmt.Sprintf("datamatrix: unsupported dataRegions %d", s.dataRegions))
	}
}

// vertRegions returns the number of data regions along the vertical axis.
func (s symbolInfo) vertRegions() int {
	switch s.dataRegions {
	case 1, 2:
		return 1
	case 4:
		return 2
	case 16:
		return 4
	case 36:
		return 6
	default:
		panic(fmt.Sprintf("datamatrix: unsupported dataRegions %d", s.dataRegions))
	}
}

// symbolCols returns the full symbol width in modules (finder patterns included).
func (s symbolInfo) symbolCols() int { return s.horizRegions() * (s.regW + 2) }

// symbolRows returns the full symbol height in modules (finder patterns included).
func (s symbolInfo) symbolRows() int { return s.vertRegions() * (s.regH + 2) }

// mapCols returns the number of columns in the logical mapping matrix.
func (s symbolInfo) mapCols() int { return s.horizRegions() * s.regW }

// mapRows returns the number of rows in the logical mapping matrix.
func (s symbolInfo) mapRows() int { return s.vertRegions() * s.regH }

// totalCW returns the total number of codewords (data plus error correction).
func (s symbolInfo) totalCW() int { return s.dataCW + s.eccCW }

// blockCount returns the number of interleaved Reed-Solomon blocks.
func (s symbolInfo) blockCount() int { return s.dataCW / s.rsBlockData }

// name returns the symbol size as "rowsxcols", e.g. "16x48".
func (s symbolInfo) name() string { return fmt.Sprintf("%dx%d", s.symbolRows(), s.symbolCols()) }

// SizePreference selects which symbol shapes [EncodeText] may choose from.
type SizePreference int

const (
	// SizeAuto picks the smallest square symbol that fits (the ECC200 default).
	SizeAuto SizePreference = iota
	// SizeSquareOnly restricts selection to square symbols.
	SizeSquareOnly
	// SizeRectangleOnly restricts selection to the six rectangular symbols.
	SizeRectangleOnly
)

// chooseSymbol returns the smallest symbol whose data capacity is at least need
// and whose shape satisfies pref, and reports whether one exists.
func chooseSymbol(need int, pref SizePreference) (symbolInfo, bool) {
	best, found := symbolInfo{}, false
	for _, s := range symbolTable {
		switch pref {
		case SizeSquareOnly, SizeAuto:
			if s.rectangular {
				continue
			}
		case SizeRectangleOnly:
			if !s.rectangular {
				continue
			}
		}
		if s.dataCW < need {
			continue
		}
		if !found || s.dataCW < best.dataCW {
			best, found = s, true
		}
	}
	return best, found
}

// symbolByDimensions returns the symbol whose full module dimensions are
// rows x cols, and reports whether one exists.
func symbolByDimensions(rows, cols int) (symbolInfo, bool) {
	for _, s := range symbolTable {
		if s.symbolRows() == rows && s.symbolCols() == cols {
			return s, true
		}
	}
	return symbolInfo{}, false
}

// blockDataLen returns the number of data codewords in interleaved block b
// (0-based). For every supported symbol the blocks are of equal size.
func (s symbolInfo) blockDataLen(b int) int {
	_ = b
	return s.dataCW / s.blockCount()
}

// blockErrLen returns the number of error codewords in interleaved block b
// (0-based).
func (s symbolInfo) blockErrLen(b int) int {
	_ = b
	return s.eccCW / s.blockCount()
}
