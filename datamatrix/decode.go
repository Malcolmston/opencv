package datamatrix

import (
	"strings"

	cv "github.com/malcolmston/opencv"
)

// decodeASCII converts ECC200 ASCII-encodation data codewords back into text.
// Decoding stops at the end-of-message codeword (129); unused (zero) codewords
// are ignored.
func decodeASCII(cw []int) (string, error) {
	var sb strings.Builder
	for _, c := range cw {
		switch {
		case c == 129:
			return sb.String(), nil // end of message / padding
		case c == 0:
			// Unused module position; ignore.
		case c >= 1 && c <= 128:
			sb.WriteByte(byte(c - 1))
		case c >= 130 && c <= 229:
			v := c - 130
			sb.WriteByte(byte('0' + v/10))
			sb.WriteByte(byte('0' + v%10))
		default:
			// 230-241 select C40/Text/X12/EDIFACT/Base256 modes, none of
			// which are implemented here.
			return "", errUnsupportedMode
		}
	}
	return sb.String(), nil
}

// readCodewords reads the totalCW codewords out of a symbol's mapping matrix
// using the ECC200 placement.
func readCodewords(sym *Symbol, spec symbolSpec) []int {
	pl := buildPlacement(spec.MappingSize(), spec.MappingSize(), spec.TotalCW())
	full := make([]int, spec.TotalCW())
	for k := 0; k < spec.TotalCW(); k++ {
		v := 0
		for b := 0; b < 8; b++ {
			pos := pl.pos[k][b]
			v <<= 1
			if sym.getMapping(pos.Row, pos.Col) {
				v |= 1
			}
		}
		full[k] = v
	}
	return full
}

// decodeSymbol reads, error-corrects and decodes a fully-sampled symbol.
func decodeSymbol(sym *Symbol) (string, error) {
	spec, ok := symbolBySize(sym.Size)
	if !ok {
		return "", errBadMatrix
	}
	full := readCodewords(sym, spec)
	corrected, _, err := rsCorrect(full, spec.ECCW)
	if err != nil {
		return "", err
	}
	return decodeASCII(corrected[:spec.DataCW])
}

// DecodeMatrix decodes a fully-sampled square module grid, where modules[r][c]
// is true for a dark module. The grid must include the finder/timing border
// and be one of the supported square sizes. It runs Reed-Solomon error
// correction before decoding, so a grid with a recoverable number of flipped
// data modules still decodes correctly.
func DecodeMatrix(modules [][]bool) (string, error) {
	n := len(modules)
	if n == 0 {
		return "", errBadMatrix
	}
	for _, row := range modules {
		if len(row) != n {
			return "", errBadMatrix
		}
	}
	if _, ok := symbolBySize(n); !ok {
		return "", errBadMatrix
	}
	return decodeSymbol(&Symbol{Size: n, Modules: modules})
}

// isDark reports whether the channel-0 sample of pixel (y, x) is dark.
func isDark(m *cv.Mat, y, x int) bool {
	if y < 0 || y >= m.Rows || x < 0 || x >= m.Cols {
		return false
	}
	return m.Data[(y*m.Cols+x)*m.Channels] < 128
}

// darkBoundingBox returns the inclusive bounding box of all dark pixels and
// whether any were found. Because the solid finder pattern touches all four
// symbol edges, this box tightly frames the symbol and excludes the white
// quiet zone.
func darkBoundingBox(m *cv.Mat) (minX, minY, maxX, maxY int, ok bool) {
	minX, minY = m.Cols, m.Rows
	maxX, maxY = -1, -1
	for y := 0; y < m.Rows; y++ {
		row := y * m.Cols * m.Channels
		for x := 0; x < m.Cols; x++ {
			if m.Data[row+x*m.Channels] < 128 {
				if x < minX {
					minX = x
				}
				if x > maxX {
					maxX = x
				}
				if y < minY {
					minY = y
				}
				if y > maxY {
					maxY = y
				}
			}
		}
	}
	return minX, minY, maxX, maxY, maxX >= 0
}

// sampleGrid samples an n x n module grid at module centres inside the given
// pixel bounding box.
func sampleGrid(m *cv.Mat, minX, minY, w, h, n int) [][]bool {
	grid := make([][]bool, n)
	for r := 0; r < n; r++ {
		grid[r] = make([]bool, n)
		py := minY + int((float64(r)+0.5)*float64(h)/float64(n))
		for c := 0; c < n; c++ {
			px := minX + int((float64(c)+0.5)*float64(w)/float64(n))
			grid[r][c] = isDark(m, py, px)
		}
	}
	return grid
}

// finderMatches checks that grid carries the ECC200 finder and timing pattern:
// a solid left column and bottom row, an alternating top row (dark on even
// columns) and an alternating right column (dark on odd rows).
func finderMatches(grid [][]bool) bool {
	n := len(grid)
	for i := 0; i < n; i++ {
		if !grid[i][0] { // left solid
			return false
		}
		if !grid[n-1][i] { // bottom solid
			return false
		}
		if grid[0][i] != (i%2 == 0) { // top timing
			return false
		}
		if grid[i][n-1] != (i%2 == 1) { // right timing
			return false
		}
	}
	return true
}

// DetectAndDecode locates a single Data Matrix symbol in the image, samples its
// module grid, and decodes it. The image may be scaled by an integer factor and
// surrounded by a white quiet zone. Modules are expected dark on a light
// background (as produced by [Encode]). Detection relies on the solid finder
// pattern framing the symbol; see the package documentation for the tolerances
// and DEFERRED capabilities.
func DetectAndDecode(m *cv.Mat) (string, error) {
	if m == nil || m.Empty() {
		return "", errNotFound
	}
	minX, minY, maxX, maxY, ok := darkBoundingBox(m)
	if !ok {
		return "", errNotFound
	}
	w := maxX - minX + 1
	h := maxY - minY + 1

	var lastErr error = errNotFound
	for _, spec := range symbolSpecs {
		n := spec.Size
		if w < n || h < n {
			continue // fewer pixels than modules: cannot sample reliably
		}
		grid := sampleGrid(m, minX, minY, w, h, n)
		if !finderMatches(grid) {
			continue
		}
		s, err := decodeSymbol(&Symbol{Size: n, Modules: grid})
		if err != nil {
			lastErr = err
			continue
		}
		return s, nil
	}
	return "", lastErr
}
