package datamatrix

// This file is the high-level ECC200 decoder: it walks a corrected data-codeword
// stream and reconstructs the original bytes plus any metadata (ECI, GS1/FNC1,
// structured append). It is the exact inverse of highlevel.go and understands
// every encodation scheme this package emits: ASCII, C40, Text, X12, EDIFACT and
// Base 256.

// decodedStream holds the reconstructed content and metadata of a symbol.
type decodedStream struct {
	bytes  []byte
	eci    int
	gs1    bool
	append *StructuredAppend
}

// decodeDataCodewords decodes info.dataCW data codewords into their content and
// metadata.
func decodeDataCodewords(cw []int) (*decodedStream, error) {
	res := &decodedStream{eci: -1}
	i := 0
	// Structured append occupies the first four codewords when present.
	if len(cw) >= 4 && cw[0] == 233 {
		seq := cw[1]
		res.append = &StructuredAppend{
			Position: (seq>>4)&0x0F + 1,
			Total:    seq&0x0F + 1,
			FileID:   [2]byte{byte(cw[2]), byte(cw[3])},
		}
		i = 4
	}
	for i < len(cw) {
		c := cw[i]
		switch {
		case c == 129: // end of message / padding
			return res, nil
		case c == 0:
			i++
		case c >= 1 && c <= 128:
			res.bytes = append(res.bytes, byte(c-1))
			i++
		case c >= 130 && c <= 229:
			v := c - 130
			res.bytes = append(res.bytes, byte('0'+v/10), byte('0'+v%10))
			i++
		case c == 230: // C40 latch
			ni, err := decodeC40Segment(cw, i+1, variantC40, res)
			if err != nil {
				return nil, err
			}
			i = ni
		case c == 239: // Text latch
			ni, err := decodeC40Segment(cw, i+1, variantText, res)
			if err != nil {
				return nil, err
			}
			i = ni
		case c == 238: // X12 latch
			ni, err := decodeC40Segment(cw, i+1, variantX12, res)
			if err != nil {
				return nil, err
			}
			i = ni
		case c == 240: // EDIFACT latch
			i = decodeEDIFACTSegment(cw, i+1, res)
		case c == 231: // Base 256 latch
			ni, err := decodeBase256Segment(cw, i+1, res)
			if err != nil {
				return nil, err
			}
			i = ni
		case c == 232: // FNC1
			if len(res.bytes) == 0 && !res.gs1 {
				res.gs1 = true
			} else {
				res.bytes = append(res.bytes, 0x1D)
			}
			i++
		case c == 235: // Upper Shift
			if i+1 >= len(cw) {
				return nil, errBadCodewords
			}
			res.bytes = append(res.bytes, byte(cw[i+1]-1+128))
			i += 2
		case c == 241: // ECI
			eci, used, err := decodeECI(cw[i+1:])
			if err != nil {
				return nil, err
			}
			res.eci = eci
			i += 1 + used
		case c == 254: // unlatch in ASCII context: no-op
			i++
		default:
			return nil, errUnsupportedMode
		}
	}
	return res, nil
}

// decodeC40Segment decodes a C40/Text/X12 segment starting at index i (just past
// the latch) until the unlatch codeword 254, appending bytes to res. It returns
// the index of the first codeword after the segment.
func decodeC40Segment(cw []int, i int, variant c40Variant, res *decodedStream) (int, error) {
	if variant == variantX12 {
		return decodeX12Segment(cw, i, res)
	}
	pendingSet := 0
	upperShift := false
	emit := func(b byte) {
		if upperShift {
			b += 128
			upperShift = false
		}
		res.bytes = append(res.bytes, b)
	}
	for i < len(cw) {
		if cw[i] == 254 {
			return i + 1, nil
		}
		if i+1 >= len(cw) {
			return i, nil // truncated triple: stop gracefully
		}
		x := cw[i]*256 + cw[i+1] - 1
		i += 2
		vals := [3]int{x / 1600, (x / 40) % 40, x % 40}
		for _, v := range vals {
			switch pendingSet {
			case 0:
				switch v {
				case 0:
					pendingSet = 1
				case 1:
					pendingSet = 2
				case 2:
					pendingSet = 3
				default:
					b, _, _, ok := c40ValueToByte(0, v, variant)
					if ok {
						emit(b)
					}
				}
			case 1:
				emit(byte(v))
				pendingSet = 0
			case 2:
				b, up, fnc1, ok := c40ValueToByte(2, v, variant)
				switch {
				case up:
					upperShift = true
				case fnc1:
					res.bytes = append(res.bytes, 0x1D)
				case ok:
					emit(b)
				}
				pendingSet = 0
			default:
				b, _, _, ok := c40ValueToByte(3, v, variant)
				if ok {
					emit(b)
				}
				pendingSet = 0
			}
		}
	}
	return i, nil
}

// decodeX12Segment decodes an ANSI X12 segment starting at index i (just past
// the latch) until the unlatch codeword 254. X12 has no shift sets: the three
// values of every triple map directly to characters.
func decodeX12Segment(cw []int, i int, res *decodedStream) (int, error) {
	for i < len(cw) {
		if cw[i] == 254 {
			return i + 1, nil
		}
		if i+1 >= len(cw) {
			return i, nil
		}
		x := cw[i]*256 + cw[i+1] - 1
		i += 2
		for _, v := range [3]int{x / 1600, (x / 40) % 40, x % 40} {
			switch {
			case v == 0:
				res.bytes = append(res.bytes, '\r')
			case v == 1:
				res.bytes = append(res.bytes, '*')
			case v == 2:
				res.bytes = append(res.bytes, '>')
			case v == 3:
				res.bytes = append(res.bytes, ' ')
			case v >= 4 && v <= 13:
				res.bytes = append(res.bytes, byte('0'+v-4))
			case v >= 14 && v <= 39:
				res.bytes = append(res.bytes, byte('A'+v-14))
			}
		}
	}
	return i, nil
}

// decodeEDIFACTSegment decodes an EDIFACT segment starting at index i (just past
// the latch). It returns the index of the first codeword after the segment.
func decodeEDIFACTSegment(cw []int, i int, res *decodedStream) int {
	bitBuf, bitCnt := 0, 0
	idx := i
	for {
		for bitCnt < 6 {
			if idx >= len(cw) {
				return idx
			}
			bitBuf = bitBuf<<8 | cw[idx]
			idx++
			bitCnt += 8
		}
		v := (bitBuf >> (bitCnt - 6)) & 0x3F
		bitCnt -= 6
		if v == edifactUnlatch {
			return idx // remaining bits of this codeword are padding
		}
		res.bytes = append(res.bytes, edifactByteToASCII(v))
	}
}

// decodeBase256Segment decodes a Base 256 segment starting at index i (just past
// the latch). It returns the index of the first codeword after the segment.
func decodeBase256Segment(cw []int, i int, res *decodedStream) (int, error) {
	if i >= len(cw) {
		return 0, errBadCodewords
	}
	d1 := unrandomize255(cw[i], i+1)
	length := d1
	i++
	if d1 >= 250 {
		if i >= len(cw) {
			return 0, errBadCodewords
		}
		d2 := unrandomize255(cw[i], i+1)
		length = (d1-249)*250 + d2
		i++
	}
	if length < 0 || i+length > len(cw) {
		return 0, errBadCodewords
	}
	for j := 0; j < length; j++ {
		res.bytes = append(res.bytes, byte(unrandomize255(cw[i], i+1)))
		i++
	}
	return i, nil
}
