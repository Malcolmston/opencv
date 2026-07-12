package barcode

import (
	"errors"
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// This file implements a QR Code symbol builder and matrix decoder for a
// practical subset of ISO/IEC 18004: versions 1-4 (21x21 to 33x33 modules),
// byte (8-bit) encoding mode, and error-correction level L. Within that scope
// the symbols are standards-compliant — finder patterns, timing patterns, the
// version-2+ alignment pattern, BCH-protected format information, a
// Reed-Solomon error-correction block and data masking all follow the
// specification — so the codes this package produces are also readable by
// conformant third-party QR scanners, and (equally important for this package)
// [QRDetectAndDecode] round-trips them.
//
// The encoder and decoder form a matched pair: what QREncode writes,
// QRDetectAndDecode reads. Larger versions, other encoding modes (numeric,
// alphanumeric, Kanji) and other ECC levels are deferred; see the package
// documentation.

// QR rendering parameters. Every module is drawn as an moduleScale x moduleScale
// block of pixels surrounded by a quietZone-module light border, matching the
// specification's recommended 4-module quiet zone.
const (
	moduleScale = 8
	quietZone   = 4

	byteMode  = 0x4 // 4-bit mode indicator for byte mode
	eclFormat = 1   // format-info ECC-level field for level L
)

// eccCodewordsL gives the number of Reed-Solomon error-correction codewords for
// level-L symbols of versions 1-4, each of which uses a single EC block.
var eccCodewordsL = map[int]int{1: 7, 2: 10, 3: 15, 4: 20}

// qrCode is a fully realised QR symbol as a square grid of modules. A true
// module is dark. isFunc marks the function-pattern modules (finder, timing,
// alignment, format and the dark module) that do not carry data and are never
// masked.
type qrCode struct {
	version int
	size    int
	modules [][]bool
	isFunc  [][]bool
}

// numRawDataModules returns the number of data-carrying modules of a symbol
// version before error correction, i.e. the module count minus the function
// patterns. Dividing by 8 gives the total codeword capacity.
func numRawDataModules(version int) int {
	result := (16*version+128)*version + 64
	if version >= 2 {
		numAlign := version/7 + 2
		result -= (25*numAlign-10)*numAlign - 55
	}
	return result
}

// totalCodewords returns the total number of 8-bit codewords in a symbol.
func totalCodewords(version int) int {
	return numRawDataModules(version) / 8
}

// dataCodewords returns the number of message (non-EC) codewords available at
// ECC level L.
func dataCodewords(version int) int {
	return totalCodewords(version) - eccCodewordsL[version]
}

// maxByteCapacity is the largest byte-mode payload (in bytes) a version holds,
// accounting for the 4-bit mode indicator and 8-bit character count.
func maxByteCapacity(version int) int {
	return dataCodewords(version) - 2
}

// newQRTemplate builds a blank symbol of the given version with all function
// patterns drawn and reserved, ready for data placement.
func newQRTemplate(version int) *qrCode {
	size := version*4 + 17
	q := &qrCode{version: version, size: size}
	q.modules = make([][]bool, size)
	q.isFunc = make([][]bool, size)
	for i := range q.modules {
		q.modules[i] = make([]bool, size)
		q.isFunc[i] = make([]bool, size)
	}
	q.drawFunctionPatterns()
	return q
}

// set writes a module value and marks it as a function module.
func (q *qrCode) setFunc(row, col int, dark bool) {
	q.modules[row][col] = dark
	q.isFunc[row][col] = true
}

func (q *qrCode) drawFunctionPatterns() {
	size := q.size
	// Timing patterns fill row 6 and column 6; finders overwrite their ends.
	for i := 0; i < size; i++ {
		q.setFunc(6, i, i%2 == 0)
		q.setFunc(i, 6, i%2 == 0)
	}
	// Three finder patterns with separators at the corners.
	q.drawFinder(3, 3)
	q.drawFinder(3, size-4)
	q.drawFinder(size-4, 3)
	// A single alignment pattern for versions 2-4, near the bottom-right.
	if q.version >= 2 {
		q.drawAlignment(size-7, size-7)
	}
	// Reserve format-information modules and set the fixed dark module.
	for _, p := range formatPositionsCopy1(size) {
		q.isFunc[p[0]][p[1]] = true
	}
	for _, p := range formatPositionsCopy2(size) {
		q.isFunc[p[0]][p[1]] = true
	}
	q.setFunc(size-8, 8, true) // dark module
}

// drawFinder renders a 7x7 finder pattern (plus its 1-module separator) centred
// on (cr, cc) using the Chebyshev-distance rings of the specification.
func (q *qrCode) drawFinder(cr, cc int) {
	for dr := -4; dr <= 4; dr++ {
		for dc := -4; dc <= 4; dc++ {
			r, c := cr+dr, cc+dc
			if r < 0 || r >= q.size || c < 0 || c >= q.size {
				continue
			}
			dist := chebyshev(dr, dc)
			q.setFunc(r, c, dist != 2 && dist != 4)
		}
	}
}

// drawAlignment renders a 5x5 alignment pattern centred on (cr, cc).
func (q *qrCode) drawAlignment(cr, cc int) {
	for dr := -2; dr <= 2; dr++ {
		for dc := -2; dc <= 2; dc++ {
			q.setFunc(cr+dr, cc+dc, chebyshev(dr, dc) != 1)
		}
	}
}

// formatPositionsCopy1 returns the 15 (row, col) module positions of the first
// format-information copy, wrapping the top-left finder, indexed by format bit.
func formatPositionsCopy1(size int) [15][2]int {
	var p [15][2]int
	for i := 0; i <= 5; i++ {
		p[i] = [2]int{i, 8}
	}
	p[6] = [2]int{7, 8}
	p[7] = [2]int{8, 8}
	p[8] = [2]int{8, 7}
	for i := 9; i < 15; i++ {
		p[i] = [2]int{8, 14 - i}
	}
	return p
}

// formatPositionsCopy2 returns the 15 (row, col) positions of the second
// format-information copy, split between the bottom-left and top-right finders.
func formatPositionsCopy2(size int) [15][2]int {
	var p [15][2]int
	for i := 0; i < 8; i++ {
		p[i] = [2]int{size - 1 - i, 8}
	}
	for i := 8; i < 15; i++ {
		p[i] = [2]int{8, size - 15 + i}
	}
	return p
}

// formatBits computes the 15-bit BCH-protected, masked format-information value
// for the given ECC-level field and mask number.
func formatBits(eclField, mask int) int {
	data := eclField<<3 | mask
	rem := data
	for i := 0; i < 10; i++ {
		rem = (rem << 1) ^ ((rem >> 9) * 0x537)
	}
	return (data<<10 | rem) ^ 0x5412
}

// drawFormatBits writes both format-information copies for the chosen mask.
func (q *qrCode) drawFormatBits(mask int) {
	bits := formatBits(eclFormat, mask)
	c1 := formatPositionsCopy1(q.size)
	c2 := formatPositionsCopy2(q.size)
	for i := 0; i < 15; i++ {
		v := (bits>>i)&1 != 0
		q.modules[c1[i][0]][c1[i][1]] = v
		q.modules[c2[i][0]][c2[i][1]] = v
	}
}

// placeCodewords lays out the interleaved data+EC codeword bytes along the
// zigzag module path of the specification, skipping function modules.
func (q *qrCode) placeCodewords(data []byte) {
	size := q.size
	i := 0
	for right := size - 1; right >= 1; right -= 2 {
		if right == 6 {
			right = 5
		}
		for vert := 0; vert < size; vert++ {
			for j := 0; j < 2; j++ {
				col := right - j
				upward := ((right + 1) & 2) == 0
				row := vert
				if upward {
					row = size - 1 - vert
				}
				if !q.isFunc[row][col] && i < len(data)*8 {
					q.modules[row][col] = (data[i>>3]>>(7-(i&7)))&1 != 0
					i++
				}
			}
		}
	}
}

// readCodewords reads modules back along the same zigzag path into n codewords.
func (q *qrCode) readCodewords(n int) []byte {
	size := q.size
	out := make([]byte, n)
	i := 0
	for right := size - 1; right >= 1; right -= 2 {
		if right == 6 {
			right = 5
		}
		for vert := 0; vert < size; vert++ {
			for j := 0; j < 2; j++ {
				col := right - j
				upward := ((right + 1) & 2) == 0
				row := vert
				if upward {
					row = size - 1 - vert
				}
				if !q.isFunc[row][col] && i < n*8 {
					if q.modules[row][col] {
						out[i>>3] |= 1 << (7 - (i & 7))
					}
					i++
				}
			}
		}
	}
	return out
}

// maskBit reports whether module (row, col) is inverted by the given data mask.
func maskBit(mask, row, col int) bool {
	x, y := col, row
	switch mask {
	case 0:
		return (x+y)%2 == 0
	case 1:
		return y%2 == 0
	case 2:
		return x%3 == 0
	case 3:
		return (x+y)%3 == 0
	case 4:
		return (y/2+x/3)%2 == 0
	case 5:
		return (x*y)%2+(x*y)%3 == 0
	case 6:
		return ((x*y)%2+(x*y)%3)%2 == 0
	case 7:
		return ((x+y)%2+(x*y)%3)%2 == 0
	}
	return false
}

// applyMask XORs the data modules with the chosen mask (its own inverse).
func (q *qrCode) applyMask(mask int) {
	for r := 0; r < q.size; r++ {
		for c := 0; c < q.size; c++ {
			if !q.isFunc[r][c] && maskBit(mask, r, c) {
				q.modules[r][c] = !q.modules[r][c]
			}
		}
	}
}

// chooseMask evaluates all eight masks and returns the one with the lowest
// penalty score, matching the selection philosophy of the specification (a
// simplified but valid penalty is used: adjacency runs, 2x2 blocks and
// dark/light balance).
func (q *qrCode) chooseMask() int {
	best, bestScore := 0, -1
	for m := 0; m < 8; m++ {
		q.applyMask(m)
		q.drawFormatBits(m)
		score := q.penalty()
		q.applyMask(m) // undo
		if bestScore < 0 || score < bestScore {
			bestScore = score
			best = m
		}
	}
	return best
}

// penalty computes a QR mask penalty score using rules 1, 2 and 4.
func (q *qrCode) penalty() int {
	size := q.size
	score := 0
	// Rule 1: runs of 5+ same-colour modules in each row and column.
	for r := 0; r < size; r++ {
		runColor, run := q.modules[r][0], 1
		for c := 1; c < size; c++ {
			if q.modules[r][c] == runColor {
				run++
			} else {
				score += runScore(run)
				runColor, run = q.modules[r][c], 1
			}
		}
		score += runScore(run)
	}
	for c := 0; c < size; c++ {
		runColor, run := q.modules[0][c], 1
		for r := 1; r < size; r++ {
			if q.modules[r][c] == runColor {
				run++
			} else {
				score += runScore(run)
				runColor, run = q.modules[r][c], 1
			}
		}
		score += runScore(run)
	}
	// Rule 2: 2x2 blocks of one colour.
	for r := 0; r < size-1; r++ {
		for c := 0; c < size-1; c++ {
			v := q.modules[r][c]
			if v == q.modules[r][c+1] && v == q.modules[r+1][c] && v == q.modules[r+1][c+1] {
				score += 3
			}
		}
	}
	// Rule 4: deviation of the dark-module proportion from 50%.
	dark := 0
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			if q.modules[r][c] {
				dark++
			}
		}
	}
	total := size * size
	k := (abs(dark*20-total*10)+total-1)/total - 1
	score += k * 10
	return score
}

func runScore(run int) int {
	if run >= 5 {
		return run - 2
	}
	return 0
}

// buildDataCodewords encodes text as a byte-mode segment padded to the version's
// data capacity, then appends the Reed-Solomon EC codewords.
func buildDataCodewords(text string, version int) []byte {
	dc := dataCodewords(version)
	bb := &bitBuffer{}
	bb.appendBits(byteMode, 4)
	bb.appendBits(len(text), 8)
	for _, b := range []byte(text) {
		bb.appendBits(int(b), 8)
	}
	capacityBits := dc * 8
	// Terminator (up to 4 zero bits) then pad to a byte boundary.
	term := 4
	if capacityBits-bb.len() < 4 {
		term = capacityBits - bb.len()
	}
	bb.appendBits(0, term)
	for bb.len()%8 != 0 {
		bb.appendBits(0, 1)
	}
	// Pad codewords alternate 0xEC and 0x11.
	pad := []int{0xEC, 0x11}
	for pi := 0; bb.len() < capacityBits; pi ^= 1 {
		bb.appendBits(pad[pi], 8)
	}
	data := bb.bytes()
	ecc := ReedSolomonEncode(data, eccCodewordsL[version])
	return append(data, ecc...)
}

// selectVersion returns the smallest supported version whose byte capacity
// holds text, or an error if none does.
func selectVersion(text string, version int) (int, error) {
	if len(text) > 255 {
		return 0, errors.New("barcode: QR byte-mode payload exceeds 255 bytes")
	}
	if version != 0 {
		if version < 1 || version > 4 {
			return 0, fmt.Errorf("barcode: QR version %d unsupported (only 1-4)", version)
		}
		if len(text) > maxByteCapacity(version) {
			return 0, fmt.Errorf("barcode: %d bytes do not fit QR version %d (max %d)", len(text), version, maxByteCapacity(version))
		}
		return version, nil
	}
	for v := 1; v <= 4; v++ {
		if len(text) <= maxByteCapacity(v) {
			return v, nil
		}
	}
	return 0, fmt.Errorf("barcode: %d bytes exceed the capacity of QR versions 1-4 (max %d)", len(text), maxByteCapacity(4))
}

// buildQR constructs a masked QR symbol for text at the given version (0 selects
// the smallest that fits).
func buildQR(text string, version int) (*qrCode, error) {
	v, err := selectVersion(text, version)
	if err != nil {
		return nil, err
	}
	q := newQRTemplate(v)
	q.placeCodewords(buildDataCodewords(text, v))
	mask := q.chooseMask()
	q.applyMask(mask)
	q.drawFormatBits(mask)
	return q, nil
}

// QREncode renders text as a QR Code symbol and returns it as a single-channel
// grayscale [cv.Mat] (dark modules are 0, light modules are 255) with a
// 4-module quiet zone and each module scaled to an 8x8 pixel block.
//
// The encoding is byte mode at error-correction level L. If version is 0 the
// smallest of versions 1-4 that fits the payload is chosen; otherwise version
// must be in 1..4 and large enough. QREncode panics if the text cannot be
// encoded (empty result, too long, or an unsupported/undersized version); use
// [QRCapacity] to check limits beforehand.
func QREncode(text string, version int) *cv.Mat {
	q, err := buildQR(text, version)
	if err != nil {
		panic(err.Error())
	}
	return q.render()
}

// QRCapacity reports the maximum byte-mode payload length (in bytes) for a
// supported QR version (1-4). It returns 0 for unsupported versions.
func QRCapacity(version int) int {
	if version < 1 || version > 4 {
		return 0
	}
	return maxByteCapacity(version)
}

// render draws the module grid into a grayscale Mat with quiet zone and scale.
func (q *qrCode) render() *cv.Mat {
	dim := (q.size + 2*quietZone) * moduleScale
	m := cv.NewMat(dim, dim, 1)
	m.SetTo(255)
	for r := 0; r < q.size; r++ {
		for c := 0; c < q.size; c++ {
			if !q.modules[r][c] {
				continue
			}
			y0 := (r + quietZone) * moduleScale
			x0 := (c + quietZone) * moduleScale
			for dy := 0; dy < moduleScale; dy++ {
				for dx := 0; dx < moduleScale; dx++ {
					m.Set(y0+dy, x0+dx, 0, 0)
				}
			}
		}
	}
	return m
}

// decodeMatrix reads a sampled module grid (dark = true) of the given version
// back into text. It reconstructs the function-pattern map, recovers the mask
// from the format information, removes the mask, reads and Reed-Solomon-corrects
// the codewords, and parses the byte-mode segment.
func decodeMatrix(modules [][]bool, version int) (string, bool) {
	size := version*4 + 17
	if len(modules) != size {
		return "", false
	}
	q := newQRTemplate(version)
	// Overlay the sampled data modules onto a fresh template (function modules
	// keep their known values; data modules are taken from the sample).
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			if !q.isFunc[r][c] {
				q.modules[r][c] = modules[r][c]
			}
		}
	}
	mask, ok := q.readFormatMask(modules)
	if !ok {
		return "", false
	}
	q.applyMask(mask)
	raw := q.readCodewords(totalCodewords(version))
	corrected, ok := ReedSolomonDecode(raw, eccCodewordsL[version])
	if !ok {
		return "", false
	}
	return parseByteSegment(corrected[:dataCodewords(version)])
}

// readFormatMask recovers the data mask from the format information, trying both
// copies and BCH-correcting against the 32 valid format strings.
func (q *qrCode) readFormatMask(modules [][]bool) (int, bool) {
	size := q.size
	read := func(pos [15][2]int) int {
		v := 0
		for i := 0; i < 15; i++ {
			if modules[pos[i][0]][pos[i][1]] {
				v |= 1 << i
			}
		}
		return v
	}
	candidates := []int{read(formatPositionsCopy1(size)), read(formatPositionsCopy2(size))}
	bestMask, bestDist := -1, 16
	for _, raw := range candidates {
		for ecl := 0; ecl < 4; ecl++ {
			for m := 0; m < 8; m++ {
				d := bitCount(raw ^ formatBits(ecl, m))
				if d < bestDist {
					bestDist = d
					bestMask = m
				}
			}
		}
	}
	if bestMask < 0 || bestDist > 3 {
		return 0, false
	}
	return bestMask, true
}

// parseByteSegment decodes a byte-mode QR data segment into its string payload.
func parseByteSegment(data []byte) (string, bool) {
	br := &bitReader{data: data}
	mode, ok := br.read(4)
	if !ok || mode != byteMode {
		return "", false
	}
	n, ok := br.read(8)
	if !ok {
		return "", false
	}
	out := make([]byte, 0, n)
	for i := 0; i < n; i++ {
		b, ok := br.read(8)
		if !ok {
			return "", false
		}
		out = append(out, byte(b))
	}
	return string(out), true
}

func chebyshev(a, b int) int {
	if a < 0 {
		a = -a
	}
	if b < 0 {
		b = -b
	}
	if a > b {
		return a
	}
	return b
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func bitCount(v int) int {
	c := 0
	for v != 0 {
		v &= v - 1
		c++
	}
	return c
}
