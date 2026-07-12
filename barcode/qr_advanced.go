package barcode

import (
	"fmt"
	"math"
	"strings"

	cv "github.com/malcolmston/opencv"
)

// This file extends the QR Code support of this package well beyond the
// versions 1-4 / byte-mode / level-L subset handled by qr.go. It adds:
//
//   - Versions 1 through 10 (21x21 up to 57x57 modules), including the multiple
//     alignment patterns and the 18-bit BCH version information of versions 7+.
//   - All four error-correction levels L, M, Q and H, with the standard
//     multi-block Reed-Solomon structure and codeword interleaving of
//     ISO/IEC 18004 Table 9.
//   - Numeric and alphanumeric encodation in addition to byte mode; the encoder
//     automatically selects the most compact mode that can represent the input,
//     and the decoder reads whichever mode a symbol carries.
//
// The public entry points are [QREncodeAdvanced] and [QRDetectAndDecodeAdvanced],
// a matched encode/decode pair, plus the capacity helper [QRDataCapacity]. They
// reuse the low-level machinery of qr.go and qr_detect.go (the qrCode grid, the
// zigzag codeword placement, the GF(256) Reed-Solomon codec, the finder
// localisation and the affine module sampler) so that the two paths share one
// tested core.

// QRECCLevel names a QR error-correction level. Higher levels devote more
// codewords to error correction and thus tolerate more damage at the cost of
// data capacity. The zero value is [QRECCLow] (level L).
type QRECCLevel int

const (
	// QRECCLow is level L, recovering roughly 7% of codewords.
	QRECCLow QRECCLevel = iota
	// QRECCMedium is level M, recovering roughly 15% of codewords.
	QRECCMedium
	// QRECCQuartile is level Q, recovering roughly 25% of codewords.
	QRECCQuartile
	// QRECCHigh is level H, recovering roughly 30% of codewords.
	QRECCHigh
)

// qrMaxVersion is the largest QR version supported by the advanced encoder and
// decoder.
const qrMaxVersion = 10

// QR encodation mode indicators (the 4-bit mode field).
const (
	modeNumeric = 1
	modeAlpha   = 2
	modeByteAdv = 4
)

// qrAlnumCharset is the 45-character alphanumeric mode alphabet, index = value.
const qrAlnumCharset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

// ecBlockSpec describes the Reed-Solomon block structure of one version/level:
// ec error-correction codewords per block, g1n blocks each holding g1d data
// codewords, and g2n blocks each holding g2d data codewords.
type ecBlockSpec struct {
	ec, g1n, g1d, g2n, g2d int
}

// dataWords returns the total number of data codewords across all blocks.
func (s ecBlockSpec) dataWords() int { return s.g1n*s.g1d + s.g2n*s.g2d }

// blocks returns the total number of Reed-Solomon blocks.
func (s ecBlockSpec) blocks() int { return s.g1n + s.g2n }

// ecTable holds the block structure for versions 1-10 indexed by version, then
// by [QRECCLevel] (L, M, Q, H). The figures are ISO/IEC 18004 Table 9.
var ecTable = map[int][4]ecBlockSpec{
	1:  {{7, 1, 19, 0, 0}, {10, 1, 16, 0, 0}, {13, 1, 13, 0, 0}, {17, 1, 9, 0, 0}},
	2:  {{10, 1, 34, 0, 0}, {16, 1, 28, 0, 0}, {22, 1, 22, 0, 0}, {28, 1, 16, 0, 0}},
	3:  {{15, 1, 55, 0, 0}, {26, 1, 44, 0, 0}, {18, 2, 17, 0, 0}, {22, 2, 13, 0, 0}},
	4:  {{20, 1, 80, 0, 0}, {18, 2, 32, 0, 0}, {26, 2, 24, 0, 0}, {16, 4, 9, 0, 0}},
	5:  {{26, 1, 108, 0, 0}, {24, 2, 43, 0, 0}, {18, 2, 15, 2, 16}, {22, 2, 11, 2, 12}},
	6:  {{18, 2, 68, 0, 0}, {16, 4, 27, 0, 0}, {24, 4, 19, 0, 0}, {28, 4, 15, 0, 0}},
	7:  {{20, 2, 78, 0, 0}, {18, 4, 31, 0, 0}, {18, 2, 14, 4, 15}, {26, 4, 13, 1, 14}},
	8:  {{24, 2, 97, 0, 0}, {22, 2, 38, 2, 39}, {22, 4, 18, 2, 19}, {26, 4, 14, 2, 15}},
	9:  {{30, 2, 116, 0, 0}, {22, 3, 36, 2, 37}, {20, 4, 16, 4, 17}, {24, 4, 12, 4, 13}},
	10: {{18, 2, 68, 2, 69}, {26, 4, 43, 1, 44}, {24, 6, 19, 2, 20}, {28, 6, 15, 2, 16}},
}

// alignPositions gives the alignment-pattern centre coordinates for each
// version; an alignment pattern is placed at every pairwise combination except
// the three that coincide with the finder patterns.
var alignPositions = map[int][]int{
	1:  nil,
	2:  {6, 18},
	3:  {6, 22},
	4:  {6, 26},
	5:  {6, 30},
	6:  {6, 34},
	7:  {6, 22, 38},
	8:  {6, 24, 42},
	9:  {6, 26, 46},
	10: {6, 28, 50},
}

// versionInfoBits gives the 18-bit BCH-protected version information for the
// versions (7+) that carry it.
var versionInfoBits = map[int]int{
	7:  0x07C94,
	8:  0x085BC,
	9:  0x09A99,
	10: 0x0A4D3,
}

// levelField maps a [QRECCLevel] to its 2-bit format-information field value.
func levelField(l QRECCLevel) int {
	switch l {
	case QRECCLow:
		return 1
	case QRECCMedium:
		return 0
	case QRECCQuartile:
		return 3
	default: // QRECCHigh
		return 2
	}
}

// fieldLevel is the inverse of levelField.
func fieldLevel(field int) QRECCLevel {
	switch field {
	case 1:
		return QRECCLow
	case 0:
		return QRECCMedium
	case 3:
		return QRECCQuartile
	default: // 2
		return QRECCHigh
	}
}

// numRawModulesEx returns the number of data-carrying modules of a version,
// correcting numRawDataModules for the version-information modules of versions
// 7+ (which that helper, written for versions 1-4, does not subtract).
func numRawModulesEx(version int) int {
	n := numRawDataModules(version)
	if version >= 7 {
		n -= 36
	}
	return n
}

// totalCodewordsEx returns the total number of 8-bit codewords in a version.
func totalCodewordsEx(version int) int {
	return numRawModulesEx(version) / 8
}

// QRDataCapacity returns the number of data (non-error-correction) codewords
// available for the given version (1-10) and error-correction level. It returns
// 0 for an unsupported version. The usable payload is a few codewords smaller
// because of the mode indicator, character count and padding.
func QRDataCapacity(version int, level QRECCLevel) int {
	specs, ok := ecTable[version]
	if !ok {
		return 0
	}
	return specs[level].dataWords()
}

// newTemplateEx builds a blank symbol of the given version (1-10) with all
// function patterns drawn and reserved: timing, three finders, every alignment
// pattern, the reserved format modules, the dark module, and — for versions 7+
// — the version-information modules.
func newTemplateEx(version int) *qrCode {
	size := version*4 + 17
	q := &qrCode{version: version, size: size}
	q.modules = make([][]bool, size)
	q.isFunc = make([][]bool, size)
	for i := range q.modules {
		q.modules[i] = make([]bool, size)
		q.isFunc[i] = make([]bool, size)
	}
	for i := 0; i < size; i++ {
		q.setFunc(6, i, i%2 == 0)
		q.setFunc(i, 6, i%2 == 0)
	}
	q.drawFinder(3, 3)
	q.drawFinder(3, size-4)
	q.drawFinder(size-4, 3)
	pos := alignPositions[version]
	for _, r := range pos {
		for _, c := range pos {
			if (r == 6 && c == 6) || (r == 6 && c == size-7) || (r == size-7 && c == 6) {
				continue
			}
			q.drawAlignment(r, c)
		}
	}
	for _, p := range formatPositionsCopy1(size) {
		q.isFunc[p[0]][p[1]] = true
	}
	for _, p := range formatPositionsCopy2(size) {
		q.isFunc[p[0]][p[1]] = true
	}
	q.setFunc(size-8, 8, true) // dark module
	if version >= 7 {
		drawVersionInfoEx(q, version)
	}
	return q
}

// drawVersionInfoEx writes both copies of the 18-bit version information and
// marks the modules as function modules.
func drawVersionInfoEx(q *qrCode, version int) {
	bits := versionInfoBits[version]
	size := q.size
	for i := 0; i < 6; i++ {
		for j := 0; j < 3; j++ {
			b := (bits>>(i*3+j))&1 != 0
			q.setFunc(size-11+j, i, b)
			q.setFunc(i, size-11+j, b)
		}
	}
}

// drawFormatBitsEx writes both format-information copies for the given level and
// mask (unlike drawFormatBits, which is fixed to level L).
func drawFormatBitsEx(q *qrCode, level QRECCLevel, mask int) {
	bits := formatBits(levelField(level), mask)
	c1 := formatPositionsCopy1(q.size)
	c2 := formatPositionsCopy2(q.size)
	for i := 0; i < 15; i++ {
		v := (bits>>i)&1 != 0
		q.modules[c1[i][0]][c1[i][1]] = v
		q.modules[c2[i][0]][c2[i][1]] = v
	}
}

// chooseMaskEx evaluates the eight data masks with the correct level format bits
// in place and returns the lowest-penalty mask.
func chooseMaskEx(q *qrCode, level QRECCLevel) int {
	best, bestScore := 0, -1
	for m := 0; m < 8; m++ {
		q.applyMask(m)
		drawFormatBitsEx(q, level, m)
		score := q.penalty()
		q.applyMask(m) // undo
		if bestScore < 0 || score < bestScore {
			bestScore = score
			best = m
		}
	}
	return best
}

// charCountBits returns the character-count indicator width for a mode at a
// version. Versions 1-9 and 10-26 use different widths; this package stops at 10.
func charCountBits(mode, version int) int {
	small := version <= 9
	switch mode {
	case modeNumeric:
		if small {
			return 10
		}
		return 12
	case modeAlpha:
		if small {
			return 9
		}
		return 11
	default: // byte
		if small {
			return 8
		}
		return 16
	}
}

// isNumericStr reports whether text is all decimal digits.
func isNumericStr(text string) bool {
	for i := 0; i < len(text); i++ {
		if text[i] < '0' || text[i] > '9' {
			return false
		}
	}
	return true
}

// isAlnumStr reports whether every byte of text lies in the alphanumeric set.
func isAlnumStr(text string) bool {
	for i := 0; i < len(text); i++ {
		if strings.IndexByte(qrAlnumCharset, text[i]) < 0 {
			return false
		}
	}
	return true
}

// selectMode picks the most compact mode able to represent text.
func selectMode(text string) int {
	switch {
	case isNumericStr(text):
		return modeNumeric
	case isAlnumStr(text):
		return modeAlpha
	default:
		return modeByteAdv
	}
}

// encodeSegmentEx appends the mode indicator, character count and encoded data
// of text (in the auto-selected mode) to bb for the given version.
func encodeSegmentEx(bb *bitBuffer, text string, version int) {
	mode := selectMode(text)
	bb.appendBits(mode, 4)
	bb.appendBits(len(text), charCountBits(mode, version))
	switch mode {
	case modeNumeric:
		i := 0
		for ; i+3 <= len(text); i += 3 {
			v := int(text[i]-'0')*100 + int(text[i+1]-'0')*10 + int(text[i+2]-'0')
			bb.appendBits(v, 10)
		}
		switch len(text) - i {
		case 2:
			bb.appendBits(int(text[i]-'0')*10+int(text[i+1]-'0'), 7)
		case 1:
			bb.appendBits(int(text[i]-'0'), 4)
		}
	case modeAlpha:
		i := 0
		for ; i+2 <= len(text); i += 2 {
			v := strings.IndexByte(qrAlnumCharset, text[i])*45 + strings.IndexByte(qrAlnumCharset, text[i+1])
			bb.appendBits(v, 11)
		}
		if i < len(text) {
			bb.appendBits(strings.IndexByte(qrAlnumCharset, text[i]), 6)
		}
	default:
		for i := 0; i < len(text); i++ {
			bb.appendBits(int(text[i]), 8)
		}
	}
}

// segmentBitLen returns the number of bits encodeSegmentEx would produce.
func segmentBitLen(text string, version int) int {
	mode := selectMode(text)
	bits := 4 + charCountBits(mode, version)
	switch mode {
	case modeNumeric:
		bits += (len(text) / 3) * 10
		switch len(text) % 3 {
		case 2:
			bits += 7
		case 1:
			bits += 4
		}
	case modeAlpha:
		bits += (len(text) / 2) * 11
		if len(text)%2 == 1 {
			bits += 6
		}
	default:
		bits += len(text) * 8
	}
	return bits
}

// selectVersionEx returns the smallest version (1-10) whose data capacity at the
// given level holds text, or the validated explicit version.
func selectVersionEx(text string, version int, level QRECCLevel) (int, error) {
	if version != 0 {
		if version < 1 || version > qrMaxVersion {
			return 0, fmt.Errorf("barcode: QR version %d unsupported (advanced encoder handles 1-%d)", version, qrMaxVersion)
		}
		if segmentBitLen(text, version) > QRDataCapacity(version, level)*8 {
			return 0, fmt.Errorf("barcode: %q does not fit QR version %d at the chosen level", text, version)
		}
		return version, nil
	}
	for v := 1; v <= qrMaxVersion; v++ {
		if segmentBitLen(text, v) <= QRDataCapacity(v, level)*8 {
			return v, nil
		}
	}
	return 0, fmt.Errorf("barcode: %q exceeds the capacity of QR versions 1-%d", text, qrMaxVersion)
}

// buildCodewordsEx encodes text into the interleaved data+EC codeword stream for
// the given version and level.
func buildCodewordsEx(text string, version int, level QRECCLevel) []byte {
	spec := ecTable[version][level]
	dataCap := spec.dataWords()
	bb := &bitBuffer{}
	encodeSegmentEx(bb, text, version)
	capBits := dataCap * 8
	term := 4
	if capBits-bb.len() < 4 {
		term = capBits - bb.len()
	}
	bb.appendBits(0, term)
	for bb.len()%8 != 0 {
		bb.appendBits(0, 1)
	}
	pad := []int{0xEC, 0x11}
	for pi := 0; bb.len() < capBits; pi ^= 1 {
		bb.appendBits(pad[pi], 8)
	}
	data := bb.bytes()

	// Split the data codewords into their blocks and compute EC per block.
	numBlocks := spec.blocks()
	blockData := make([][]byte, numBlocks)
	blockEc := make([][]byte, numBlocks)
	off := 0
	for b := 0; b < numBlocks; b++ {
		n := spec.g1d
		if b >= spec.g1n {
			n = spec.g2d
		}
		blockData[b] = data[off : off+n]
		off += n
		blockEc[b] = ReedSolomonEncode(blockData[b], spec.ec)
	}
	// Interleave data codewords, then EC codewords.
	maxData := spec.g1d
	if spec.g2d > maxData {
		maxData = spec.g2d
	}
	out := make([]byte, 0, totalCodewordsEx(version))
	for i := 0; i < maxData; i++ {
		for b := 0; b < numBlocks; b++ {
			if i < len(blockData[b]) {
				out = append(out, blockData[b][i])
			}
		}
	}
	for i := 0; i < spec.ec; i++ {
		for b := 0; b < numBlocks; b++ {
			out = append(out, blockEc[b][i])
		}
	}
	return out
}

// QREncodeAdvanced renders text as a QR Code symbol and returns it as a
// single-channel grayscale [cv.Mat] (dark modules 0, light 255) with a 4-module
// quiet zone. It supports versions 1-10, error-correction levels L/M/Q/H, and
// automatically selects numeric, alphanumeric or byte encodation for the input.
// If version is 0 the smallest version that fits is chosen; otherwise version
// must be in 1..10 and large enough. It returns an error if the text cannot be
// encoded. Symbols round-trip through [QRDetectAndDecodeAdvanced].
func QREncodeAdvanced(text string, version int, level QRECCLevel) (*cv.Mat, error) {
	if level < QRECCLow || level > QRECCHigh {
		return nil, fmt.Errorf("barcode: invalid QR error-correction level %d", int(level))
	}
	v, err := selectVersionEx(text, version, level)
	if err != nil {
		return nil, err
	}
	q := newTemplateEx(v)
	q.placeCodewords(buildCodewordsEx(text, v, level))
	mask := chooseMaskEx(q, level)
	q.applyMask(mask)
	drawFormatBitsEx(q, level, mask)
	return q.render(), nil
}

// readFormatFull recovers the error-correction level and mask from a sampled
// grid, trying both format copies and BCH-correcting against the 32 valid
// format strings. It reports false if no candidate is within distance 3.
func readFormatFull(q *qrCode, modules [][]bool) (QRECCLevel, int, bool) {
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
	bestField, bestMask, bestDist := -1, -1, 16
	for _, raw := range candidates {
		for field := 0; field < 4; field++ {
			for m := 0; m < 8; m++ {
				d := bitCount(raw ^ formatBits(field, m))
				if d < bestDist {
					bestDist = d
					bestField = field
					bestMask = m
				}
			}
		}
	}
	if bestMask < 0 || bestDist > 3 {
		return 0, 0, false
	}
	return fieldLevel(bestField), bestMask, true
}

// deinterleaveDecode de-interleaves the codeword stream into its blocks,
// Reed-Solomon-corrects each block, and concatenates the corrected data
// codewords into the message stream.
func deinterleaveDecode(raw []byte, version int, level QRECCLevel) ([]byte, bool) {
	spec := ecTable[version][level]
	numBlocks := spec.blocks()
	dataSizes := make([]int, numBlocks)
	for b := 0; b < numBlocks; b++ {
		if b < spec.g1n {
			dataSizes[b] = spec.g1d
		} else {
			dataSizes[b] = spec.g2d
		}
	}
	totalData := spec.dataWords()
	if len(raw) < totalData+numBlocks*spec.ec {
		return nil, false
	}
	rawData := raw[:totalData]
	rawEc := raw[totalData : totalData+numBlocks*spec.ec]

	blockData := make([][]byte, numBlocks)
	maxData := spec.g1d
	if spec.g2d > maxData {
		maxData = spec.g2d
	}
	idx := 0
	for i := 0; i < maxData; i++ {
		for b := 0; b < numBlocks; b++ {
			if i < dataSizes[b] {
				blockData[b] = append(blockData[b], rawData[idx])
				idx++
			}
		}
	}
	blockEc := make([][]byte, numBlocks)
	idx = 0
	for i := 0; i < spec.ec; i++ {
		for b := 0; b < numBlocks; b++ {
			blockEc[b] = append(blockEc[b], rawEc[idx])
			idx++
		}
	}
	var msg []byte
	for b := 0; b < numBlocks; b++ {
		full := append(append([]byte{}, blockData[b]...), blockEc[b]...)
		corr, ok := ReedSolomonDecode(full, spec.ec)
		if !ok {
			return nil, false
		}
		msg = append(msg, corr[:dataSizes[b]]...)
	}
	return msg, true
}

// parseSegmentsEx decodes the concatenated message codewords, reading a sequence
// of numeric, alphanumeric and byte segments until a terminator or the end.
func parseSegmentsEx(data []byte, version int) (string, bool) {
	br := &bitReader{data: data}
	var out []byte
	for {
		mode, ok := br.read(4)
		if !ok || mode == 0 {
			break
		}
		switch mode {
		case modeNumeric:
			n, ok := br.read(charCountBits(modeNumeric, version))
			if !ok {
				return "", false
			}
			full := n / 3
			for i := 0; i < full; i++ {
				v, ok := br.read(10)
				if !ok || v > 999 {
					return "", false
				}
				out = append(out, byte('0'+v/100), byte('0'+(v/10)%10), byte('0'+v%10))
			}
			switch n % 3 {
			case 2:
				v, ok := br.read(7)
				if !ok || v > 99 {
					return "", false
				}
				out = append(out, byte('0'+v/10), byte('0'+v%10))
			case 1:
				v, ok := br.read(4)
				if !ok || v > 9 {
					return "", false
				}
				out = append(out, byte('0'+v))
			}
		case modeAlpha:
			n, ok := br.read(charCountBits(modeAlpha, version))
			if !ok {
				return "", false
			}
			pairs := n / 2
			for i := 0; i < pairs; i++ {
				v, ok := br.read(11)
				if !ok || v >= 45*45 {
					return "", false
				}
				out = append(out, qrAlnumCharset[v/45], qrAlnumCharset[v%45])
			}
			if n%2 == 1 {
				v, ok := br.read(6)
				if !ok || v >= 45 {
					return "", false
				}
				out = append(out, qrAlnumCharset[v])
			}
		case modeByteAdv:
			n, ok := br.read(charCountBits(modeByteAdv, version))
			if !ok {
				return "", false
			}
			for i := 0; i < n; i++ {
				v, ok := br.read(8)
				if !ok {
					return "", false
				}
				out = append(out, byte(v))
			}
		default:
			// Unsupported mode (ECI, Kanji, structured append): stop cleanly.
			return string(out), true
		}
	}
	return string(out), true
}

// decodeMatrixEx reads a sampled module grid of the given version back into text
// across all supported modes and error-correction levels.
func decodeMatrixEx(modules [][]bool, version int) (string, bool) {
	size := version*4 + 17
	if len(modules) != size {
		return "", false
	}
	q := newTemplateEx(version)
	for r := 0; r < size; r++ {
		for c := 0; c < size; c++ {
			if !q.isFunc[r][c] {
				q.modules[r][c] = modules[r][c]
			}
		}
	}
	level, mask, ok := readFormatFull(q, modules)
	if !ok {
		return "", false
	}
	q.applyMask(mask)
	raw := q.readCodewords(totalCodewordsEx(version))
	msg, ok := deinterleaveDecode(raw, version, level)
	if !ok {
		return "", false
	}
	return parseSegmentsEx(msg, version)
}

// QRDetectAndDecodeAdvanced locates a QR Code symbol in img, samples its modules
// and decodes it, handling versions 1-10, all error-correction levels and the
// numeric, alphanumeric and byte modes produced by [QREncodeAdvanced] (including
// axis-aligned rotations). It returns the text and true on success, or
// ("", false) if no decodable symbol is found. Reed-Solomon correction means a
// few misread modules are recovered rather than fatal.
func QRDetectAndDecodeAdvanced(img *cv.Mat) (string, bool) {
	if img == nil || img.Empty() {
		return "", false
	}
	dark := toDarkGrid(img)
	centers := strongCenters(findFinderCenters(dark))
	if len(centers) < 3 {
		return "", false
	}
	var three [3]finderCenter
	copy(three[:], centers[:3])
	tl, tr, bl := orientFinders(three)
	module := (tl.module + tr.module + bl.module) / 3
	if module <= 0 {
		return "", false
	}
	distTR := math.Hypot(tr.x-tl.x, tr.y-tl.y)
	distBL := math.Hypot(bl.x-tl.x, bl.y-tl.y)
	dimEst := (distTR+distBL)/(2*module) + 7
	base := int(math.Round((dimEst - 17) / 4))

	// Try the estimated version and its immediate neighbours, since the module
	// estimate can be off by one for the larger symbols.
	for _, delta := range []int{0, 1, -1, 2, -2} {
		v := base + delta
		if v < 1 || v > qrMaxVersion {
			continue
		}
		dim := v*4 + 17
		grid := sampleGrid(dark, tl, tr, bl, dim)
		if text, ok := decodeMatrixEx(grid, v); ok {
			return text, true
		}
	}
	return "", false
}
