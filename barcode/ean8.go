package barcode

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// This file implements EAN-8, the eight-digit retail symbology for small
// packages. Unlike EAN-13, EAN-8 uses no parity encoding of a hidden digit: its
// four left-hand digits are all L-code (odd parity) and its four right-hand
// digits are all R-code. The 67-module layout is start guard (101), four left
// digits, centre guard (01010), four right digits, end guard (101). The last of
// the eight digits is a modulo-10 check digit. [EncodeEAN8] and [DecodeEAN8]
// reuse the shared eanL/eanR tables and form a matched pair.

// ean8Checksum returns the EAN-8 check digit for the first seven digits.
func ean8Checksum(d []int) int {
	sum := 0
	for i := 0; i < 7; i++ {
		if i%2 == 0 {
			sum += 3 * d[i]
		} else {
			sum += d[i]
		}
	}
	return (10 - sum%10) % 10
}

// ean8Modules builds the 67-module bar pattern for eight digits.
func ean8Modules(d []int) []bool {
	m := make([]bool, 0, 67)
	put := func(pattern, bits int) {
		for i := bits - 1; i >= 0; i-- {
			m = append(m, (pattern>>i)&1 != 0)
		}
	}
	put(0x5, 3) // start guard 101
	for i := 0; i < 4; i++ {
		put(eanL[d[i]], 7)
	}
	put(0x0A, 5) // centre guard 01010
	for i := 0; i < 4; i++ {
		put(eanR[d[i+4]], 7)
	}
	put(0x5, 3) // end guard 101
	return m
}

// EncodeEAN8 renders digits as an EAN-8 barcode and returns it as a
// single-channel grayscale [cv.Mat] (bars 0, spaces 255) with quiet zones.
// digits must be 7 digits (the check digit is computed) or 8 digits (the check
// digit is validated). It returns an error for malformed input.
func EncodeEAN8(digits string) (*cv.Mat, error) {
	d, err := parseDigits(digits)
	if err != nil {
		return nil, err
	}
	switch len(d) {
	case 7:
		d = append(d, ean8Checksum(d))
	case 8:
		if ean8Checksum(d[:7]) != d[7] {
			return nil, fmt.Errorf("barcode: EAN-8 check digit is %d, want %d", d[7], ean8Checksum(d[:7]))
		}
	default:
		return nil, fmt.Errorf("barcode: EAN-8 needs 7 or 8 digits, got %d", len(d))
	}
	return renderModules1D(ean8Modules(d)), nil
}

// DecodeEAN8 scans a rendered EAN-8 barcode and returns the eight-digit string
// and true on success, or ("", false) if no valid symbol is found. It samples
// the 67 modules of a middle scanline, checks the guard patterns, decodes the
// four L-code and four R-code digits, and verifies the check digit.
func DecodeEAN8(img *cv.Mat) (string, bool) {
	mod, ok := sampleFixed1D(img, 67)
	if !ok {
		return "", false
	}
	if !(mod[0] && !mod[1] && mod[2]) ||
		!(!mod[31] && mod[32] && !mod[33] && mod[34] && !mod[35]) ||
		!(mod[64] && !mod[65] && mod[66]) {
		return "", false
	}
	group := func(start int) int {
		v := 0
		for i := 0; i < 7; i++ {
			v <<= 1
			if mod[start+i] {
				v |= 1
			}
		}
		return v
	}
	digits := make([]int, 8)
	for i := 0; i < 4; i++ {
		d := matchDigit(group(3+i*7), &eanL)
		if d < 0 {
			return "", false
		}
		digits[i] = d
	}
	for i := 0; i < 4; i++ {
		d := matchDigit(group(36+i*7), &eanR)
		if d < 0 {
			return "", false
		}
		digits[i+4] = d
	}
	if ean8Checksum(digits[:7]) != digits[7] {
		return "", false
	}
	out := make([]byte, 8)
	for i, d := range digits {
		out[i] = byte('0' + d)
	}
	return string(out), true
}
