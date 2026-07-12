package datamatrix

// This file implements the shared machinery of the three "C40-like" encodation
// schemes of ECC200 (C40, Text and ANSI X12). All three pack three intermediate
// "C40 values" (each 0..39) into two codewords as 1600*v1 + 40*v2 + v3 + 1, and
// differ only in how a character maps to values. Characters outside a scheme's
// native repertoire are reached through shift sets; bytes above 0x7F use the
// Shift-2 "Upper Shift" (value 30) prefix, exactly as in ISO/IEC 16022.

// c40Variant selects which C40-like value mapping to use.
type c40Variant int

const (
	variantC40 c40Variant = iota
	variantText
	variantX12
)

// latch codeword for each C40-like variant.
func (v c40Variant) latch() int {
	switch v {
	case variantC40:
		return 230
	case variantText:
		return 239
	default:
		return 238
	}
}

// c40Values returns the C40 intermediate values encoding a single byte in the
// given variant, and reports whether the byte is representable. X12 rejects any
// character outside its fixed repertoire; C40 and Text represent every byte.
func c40Values(b byte, variant c40Variant) ([]int, bool) {
	if variant == variantX12 {
		switch {
		case b == '\r':
			return []int{0}, true
		case b == '*':
			return []int{1}, true
		case b == '>':
			return []int{2}, true
		case b == ' ':
			return []int{3}, true
		case b >= '0' && b <= '9':
			return []int{int(b-'0') + 4}, true
		case b >= 'A' && b <= 'Z':
			return []int{int(b-'A') + 14}, true
		default:
			return nil, false
		}
	}
	// C40 and Text share structure; only the basic letters and Shift-3 set differ.
	if b >= 0x80 {
		inner, _ := c40Values(b-0x80, variant)
		return append([]int{1, 30}, inner...), true // Shift 2, Upper Shift
	}
	switch {
	case b == ' ':
		return []int{3}, true
	case b >= '0' && b <= '9':
		return []int{int(b-'0') + 4}, true
	}
	if variant == variantC40 {
		if b >= 'A' && b <= 'Z' {
			return []int{int(b-'A') + 14}, true
		}
	} else { // Text: lowercase in the basic set
		if b >= 'a' && b <= 'z' {
			return []int{int(b-'a') + 14}, true
		}
	}
	switch {
	case b <= 0x1F:
		return []int{0, int(b)}, true // Shift 1
	case b >= 0x21 && b <= 0x2F:
		return []int{1, int(b) - 33}, true // Shift 2
	case b >= 0x3A && b <= 0x40:
		return []int{1, int(b) - 43}, true // Shift 2
	case b >= 0x5B && b <= 0x5F:
		return []int{1, int(b) - 69}, true // Shift 2
	}
	// Remaining range 0x60..0x7F uses Shift 3, which differs between variants.
	if variant == variantC40 {
		return []int{2, int(b) - 96}, true
	}
	// Text Shift 3: `=0, A-Z=1..26, {|}~DEL=27..31.
	switch {
	case b == 0x60:
		return []int{2, 0}, true
	case b >= 'A' && b <= 'Z':
		return []int{2, int(b) - 64}, true
	default: // 0x7B..0x7F
		return []int{2, int(b) - 96}, true
	}
}

// packC40Triples packs a value list whose length is a multiple of three into
// codewords, two per triple.
func packC40Triples(values []int) []int {
	out := make([]int, 0, len(values)/3*2)
	for i := 0; i < len(values); i += 3 {
		n := 1600*values[i] + 40*values[i+1] + values[i+2] + 1
		out = append(out, n>>8, n&0xFF)
	}
	return out
}

// c40ValueToByte converts a decoded C40 value in the given set back to the
// source byte for the given variant. set is 0 (basic), 1, 2 or 3 (shift sets).
// It returns the byte, whether the value is an Upper-Shift toggle (Shift 2
// value 30), whether it is FNC1 (Shift 2 value 27), and whether the value is a
// valid decodable character.
func c40ValueToByte(set, value int, variant c40Variant) (b byte, upperShift, fnc1, ok bool) {
	switch set {
	case 0:
		switch {
		case value == 3:
			return ' ', false, false, true
		case value >= 4 && value <= 13:
			return byte('0' + value - 4), false, false, true
		case value >= 14 && value <= 39:
			if variant == variantText {
				return byte('a' + value - 14), false, false, true
			}
			return byte('A' + value - 14), false, false, true
		}
		return 0, false, false, false
	case 1:
		return byte(value), false, false, true
	case 2:
		switch {
		case value >= 0 && value <= 14:
			return byte(33 + value), false, false, true
		case value >= 15 && value <= 21:
			return byte(0x3A + value - 15), false, false, true
		case value >= 22 && value <= 26:
			return byte(0x5B + value - 22), false, false, true
		case value == 27:
			return 0, false, true, true // FNC1
		case value == 30:
			return 0, true, false, true // Upper Shift
		}
		return 0, false, false, false
	default: // set 3
		if variant == variantText {
			switch {
			case value == 0:
				return 0x60, false, false, true
			case value >= 1 && value <= 26:
				return byte('A' + value - 1), false, false, true
			default:
				return byte(value + 96), false, false, true
			}
		}
		return byte(value + 96), false, false, true
	}
}
