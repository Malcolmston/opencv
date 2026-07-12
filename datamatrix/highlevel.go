package datamatrix

import "math"

// This file is the high-level ECC200 encoder. It turns a byte string plus
// encoding options into the sequence of data codewords, choosing per-run among
// the ASCII, C40, Text, X12, EDIFACT and Base 256 schemes so as to keep the
// codeword count small, and emitting the ECI, GS1/FNC1 and structured-append
// codewords when requested. Scheme selection uses the ISO/IEC 16022 Annex P
// look-ahead cost model (the reference algorithm used to decide when switching
// schemes actually pays for its latch/unlatch overhead).

// internal encodation-mode identifiers used by the look-ahead cost model.
const (
	modeASCII = iota
	modeC40
	modeText
	modeX12
	modeEDIFACT
	modeBase256
)

// Scheme selects the encodation scheme used for the message content.
type Scheme int

const (
	// SchemeAuto minimises the codeword count by switching schemes as needed.
	SchemeAuto Scheme = iota
	// SchemeASCII forces ASCII encodation (digit pairs are still packed).
	SchemeASCII
	// SchemeC40 prefers C40 (efficient for upper-case and digits).
	SchemeC40
	// SchemeText prefers Text (efficient for lower-case and digits).
	SchemeText
	// SchemeX12 prefers ANSI X12 (upper-case, digits, space, CR, '*', '>').
	SchemeX12
	// SchemeEDIFACT prefers EDIFACT (ASCII 0x20..0x5E).
	SchemeEDIFACT
	// SchemeBase256 forces Base 256 (arbitrary bytes).
	SchemeBase256
)

func isDigitByte(b byte) bool     { return b >= '0' && b <= '9' }
func isExtendedByte(b byte) bool  { return b >= 0x80 }
func isNativeC40Byte(b byte) bool { return b == ' ' || isDigitByte(b) || (b >= 'A' && b <= 'Z') }
func isNativeTextByte(b byte) bool {
	return b == ' ' || isDigitByte(b) || (b >= 'a' && b <= 'z')
}
func isNativeX12Byte(b byte) bool {
	return b == '\r' || b == '*' || b == '>' || b == ' ' || isDigitByte(b) || (b >= 'A' && b <= 'Z')
}

// findMinimums fills intCharCounts with the ceilings of charCounts, records
// which modes attain the minimum in mins, and returns that minimum.
func findMinimums(charCounts []float64, intCharCounts []int, mins []int) int {
	min := math.MaxInt
	for i := range mins {
		mins[i] = 0
	}
	for i := 0; i < 6; i++ {
		intCharCounts[i] = int(math.Ceil(charCounts[i]))
		if intCharCounts[i] < min {
			min = intCharCounts[i]
			for j := range mins {
				mins[j] = 0
			}
		}
		if intCharCounts[i] == min {
			mins[i]++
		}
	}
	return min
}

func minimumCount(mins []int) int {
	c := 0
	for _, m := range mins {
		c += m
	}
	return c
}

// lookAheadTest implements the ISO/IEC 16022 Annex P look-ahead: starting at
// startpos in currentMode, it returns the encodation mode that is estimated to
// produce the fewest codewords for the run beginning here.
func lookAheadTest(msg []byte, startpos, currentMode int) int {
	newMode := lookAheadTestIntern(msg, startpos, currentMode)
	if currentMode == modeX12 && newMode == modeX12 {
		end := startpos + 3
		if end > len(msg) {
			end = len(msg)
		}
		for i := startpos; i < end; i++ {
			if !isNativeX12Byte(msg[i]) {
				return modeASCII
			}
		}
	} else if currentMode == modeEDIFACT && newMode == modeEDIFACT {
		end := startpos + 4
		if end > len(msg) {
			end = len(msg)
		}
		for i := startpos; i < end; i++ {
			if !isEDIFACTNative(msg[i]) {
				return modeASCII
			}
		}
	}
	return newMode
}

func lookAheadTestIntern(msg []byte, startpos, currentMode int) int {
	if startpos >= len(msg) {
		return currentMode
	}
	var charCounts []float64
	if currentMode == modeASCII {
		charCounts = []float64{0, 1, 1, 1, 1, 1.25}
	} else {
		charCounts = []float64{1, 2, 2, 2, 2, 2.25}
		charCounts[currentMode] = 0
	}

	intCharCounts := make([]int, 6)
	mins := make([]int, 6)
	charsProcessed := 0
	for {
		if startpos+charsProcessed == len(msg) {
			findMinimums(charCounts, intCharCounts, mins)
			mc := minimumCount(mins)
			min := intCharCounts[modeASCII]
			for _, v := range intCharCounts {
				if v < min {
					min = v
				}
			}
			if intCharCounts[modeASCII] == min {
				return modeASCII
			}
			if mc == 1 {
				if mins[modeBase256] > 0 {
					return modeBase256
				}
				if mins[modeEDIFACT] > 0 {
					return modeEDIFACT
				}
				if mins[modeText] > 0 {
					return modeText
				}
				if mins[modeX12] > 0 {
					return modeX12
				}
			}
			return modeC40
		}

		c := msg[startpos+charsProcessed]
		charsProcessed++

		if isDigitByte(c) {
			charCounts[modeASCII] += 0.5
		} else if isExtendedByte(c) {
			charCounts[modeASCII] = math.Ceil(charCounts[modeASCII]) + 2
		} else {
			charCounts[modeASCII] = math.Ceil(charCounts[modeASCII]) + 1
		}

		if isNativeC40Byte(c) {
			charCounts[modeC40] += 2.0 / 3.0
		} else if isExtendedByte(c) {
			charCounts[modeC40] += 8.0 / 3.0
		} else {
			charCounts[modeC40] += 4.0 / 3.0
		}

		if isNativeTextByte(c) {
			charCounts[modeText] += 2.0 / 3.0
		} else if isExtendedByte(c) {
			charCounts[modeText] += 8.0 / 3.0
		} else {
			charCounts[modeText] += 4.0 / 3.0
		}

		if isNativeX12Byte(c) {
			charCounts[modeX12] += 2.0 / 3.0
		} else if isExtendedByte(c) {
			charCounts[modeX12] += 13.0 / 3.0
		} else {
			charCounts[modeX12] += 10.0 / 3.0
		}

		if isEDIFACTNative(c) {
			charCounts[modeEDIFACT] += 3.0 / 4.0
		} else if isExtendedByte(c) {
			charCounts[modeEDIFACT] += 17.0 / 4.0
		} else {
			charCounts[modeEDIFACT] += 13.0 / 4.0
		}

		charCounts[modeBase256]++

		if charsProcessed >= 4 {
			findMinimums(charCounts, intCharCounts, mins)
			mc := minimumCount(mins)

			if intCharCounts[modeASCII] < intCharCounts[modeBase256] &&
				intCharCounts[modeASCII] < intCharCounts[modeC40] &&
				intCharCounts[modeASCII] < intCharCounts[modeText] &&
				intCharCounts[modeASCII] < intCharCounts[modeX12] &&
				intCharCounts[modeASCII] < intCharCounts[modeEDIFACT] {
				return modeASCII
			}
			if intCharCounts[modeBase256] < intCharCounts[modeASCII] ||
				mins[modeC40]+mins[modeText]+mins[modeX12]+mins[modeEDIFACT] == 0 {
				return modeBase256
			}
			if mc == 1 && mins[modeEDIFACT] > 0 {
				return modeEDIFACT
			}
			if mc == 1 && mins[modeText] > 0 {
				return modeText
			}
			if mc == 1 && mins[modeX12] > 0 {
				return modeX12
			}
			if intCharCounts[modeC40]+1 < intCharCounts[modeASCII] &&
				intCharCounts[modeC40]+1 < intCharCounts[modeBase256] &&
				intCharCounts[modeC40]+1 < intCharCounts[modeEDIFACT] &&
				intCharCounts[modeC40]+1 < intCharCounts[modeText] {
				if intCharCounts[modeC40] < intCharCounts[modeX12] {
					return modeC40
				}
				if intCharCounts[modeC40] == intCharCounts[modeX12] {
					p := startpos + charsProcessed + 1
					for p < len(msg) {
						tc := msg[p]
						if isNativeX12Byte(tc) {
							return modeX12
						}
						if !isNativeC40Byte(tc) {
							break
						}
						p++
					}
					return modeC40
				}
			}
		}
	}
}

// encodeASCIIChar encodes a single character (or a digit pair) in ASCII mode and
// returns the codewords and the number of source bytes consumed.
func encodeASCIIChar(msg []byte, pos int, gs1 bool) ([]int, int) {
	b := msg[pos]
	if gs1 && b == 0x1D {
		return []int{232}, 1 // FNC1 separator
	}
	if isDigitByte(b) && pos+1 < len(msg) && isDigitByte(msg[pos+1]) {
		return []int{130 + int(b-'0')*10 + int(msg[pos+1]-'0')}, 2
	}
	if b < 0x80 {
		return []int{int(b) + 1}, 1
	}
	return []int{235, int(b) - 127}, 1 // Upper Shift encodes one source byte
}

// c40NativeRun returns the number of consecutive characters from pos that are
// native to the given variant (each contributing exactly one C40 value).
func c40NativeRun(msg []byte, pos int, variant c40Variant) int {
	n := 0
	for pos+n < len(msg) {
		var native bool
		switch variant {
		case variantC40:
			native = isNativeC40Byte(msg[pos+n])
		case variantText:
			native = isNativeTextByte(msg[pos+n])
		default:
			native = isNativeX12Byte(msg[pos+n])
		}
		if !native {
			break
		}
		n++
	}
	return n
}

// encodeC40Run encodes a whole-triple run of native characters in the given
// C40-like variant. It reports the number of characters consumed and whether a
// full triple could be formed (a run shorter than three characters cannot).
func encodeC40Run(msg []byte, pos int, variant c40Variant) (adv int, cws []int, ok bool) {
	run := c40NativeRun(msg, pos, variant)
	k := run - run%3
	if k < 3 {
		return 0, nil, false
	}
	values := make([]int, 0, k)
	for i := 0; i < k; i++ {
		v, _ := c40Values(msg[pos+i], variant)
		values = append(values, v...)
	}
	cws = append([]int{variant.latch()}, packC40Triples(values)...)
	cws = append(cws, 254) // unlatch to ASCII
	return k, cws, true
}

// encodeEDIFACTRun encodes a whole-group run of EDIFACT-native characters. It
// reports the number of characters consumed and whether a full four-character
// group could be formed.
func encodeEDIFACTRun(msg []byte, pos int) (adv int, cws []int, ok bool) {
	n := 0
	for pos+n < len(msg) && isEDIFACTNative(msg[pos+n]) {
		n++
	}
	k := n - n%4
	if k < 4 {
		return 0, nil, false
	}
	cws = append([]int{240}, packEDIFACT(msg[pos:pos+k])...)
	return k, cws, true
}

// base256RunLength returns how many bytes from pos should be grouped into a
// single Base 256 segment.
func base256RunLength(msg []byte, pos int, scheme Scheme) int {
	if scheme == SchemeBase256 {
		return len(msg) - pos
	}
	l := 1
	for pos+l < len(msg) && lookAheadTest(msg, pos+l, modeASCII) == modeBase256 {
		l++
	}
	return l
}

// chooseMode decides the encodation mode for the run beginning at pos, honouring
// a forced scheme or, for SchemeAuto, the Annex P look-ahead.
func chooseMode(msg []byte, pos int, scheme Scheme) int {
	switch scheme {
	case SchemeASCII:
		return modeASCII
	case SchemeC40:
		if isNativeC40Byte(msg[pos]) {
			return modeC40
		}
		return modeASCII
	case SchemeText:
		if isNativeTextByte(msg[pos]) {
			return modeText
		}
		return modeASCII
	case SchemeX12:
		if isNativeX12Byte(msg[pos]) {
			return modeX12
		}
		return modeASCII
	case SchemeEDIFACT:
		if isEDIFACTNative(msg[pos]) {
			return modeEDIFACT
		}
		return modeASCII
	case SchemeBase256:
		return modeBase256
	default:
		return lookAheadTest(msg, pos, modeASCII)
	}
}

// encodeContent encodes the message body (after any ECI/GS1/structured-append
// prefix, whose length is offset codewords) into data codewords.
func encodeContent(msg []byte, scheme Scheme, gs1 bool, offset int) []int {
	out := []int{}
	pos := 0
	for pos < len(msg) {
		switch chooseMode(msg, pos, scheme) {
		case modeC40, modeText, modeX12:
			variant := variantC40
			switch chooseMode(msg, pos, scheme) {
			case modeText:
				variant = variantText
			case modeX12:
				variant = variantX12
			}
			if adv, cws, ok := encodeC40Run(msg, pos, variant); ok {
				out = append(out, cws...)
				pos += adv
				continue
			}
			cws, adv := encodeASCIIChar(msg, pos, gs1)
			out = append(out, cws...)
			pos += adv
		case modeEDIFACT:
			if adv, cws, ok := encodeEDIFACTRun(msg, pos); ok {
				out = append(out, cws...)
				pos += adv
				continue
			}
			cws, adv := encodeASCIIChar(msg, pos, gs1)
			out = append(out, cws...)
			pos += adv
		case modeBase256:
			run := base256RunLength(msg, pos, scheme)
			seg := packBase256(msg[pos:pos+run], offset+len(out)+1)
			out = append(out, seg...)
			pos += run
		default:
			cws, adv := encodeASCIIChar(msg, pos, gs1)
			out = append(out, cws...)
			pos += adv
		}
	}
	return out
}

// structuredAppendCodewords returns the four data codewords that introduce a
// structured-append symbol: the marker 233, the symbol-sequence indicator and
// the two file-identifier codewords.
func structuredAppendCodewords(sa StructuredAppend) ([]int, error) {
	if sa.Position < 1 || sa.Position > 16 || sa.Total < 1 || sa.Total > 16 || sa.Position > sa.Total {
		return nil, errInvalidAppend
	}
	if sa.FileID[0] < 1 || sa.FileID[0] > 254 || sa.FileID[1] < 1 || sa.FileID[1] > 254 {
		return nil, errInvalidAppend
	}
	seq := (sa.Position-1)<<4 | (sa.Total - 1)
	return []int{233, seq, int(sa.FileID[0]), int(sa.FileID[1])}, nil
}

// buildDataCodewords runs the full high-level encoder: it assembles any
// prefixes, encodes the content, selects the smallest fitting symbol and pads
// the data codewords.
func buildDataCodewords(input []byte, opts EncodeOptions) ([]int, symbolInfo, error) {
	var pre []int
	if opts.Append != nil {
		sa, err := structuredAppendCodewords(*opts.Append)
		if err != nil {
			return nil, symbolInfo{}, err
		}
		pre = append(pre, sa...)
	}
	if opts.GS1 {
		pre = append(pre, 232) // FNC1 in the first position
	}
	if opts.UseECI {
		if opts.ECI < 0 || opts.ECI > eciMaxValue {
			return nil, symbolInfo{}, errInvalidECI
		}
		pre = append(pre, encodeECI(opts.ECI)...)
	}

	scheme := opts.Scheme
	if opts.GS1 && scheme == SchemeAuto {
		// FNC1 group separators are only handled in ASCII mode by this codec.
		scheme = SchemeASCII
	}
	content := encodeContent(input, scheme, opts.GS1, len(pre))

	data := append(pre, content...)
	info, ok := chooseSymbol(len(data), opts.Size)
	if !ok {
		return nil, symbolInfo{}, errTooLong
	}
	data = padCodewords(data, info.dataCW)
	return data, info, nil
}
