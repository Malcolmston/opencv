package barcode

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// This file implements UPC-A, the twelve-digit North American retail
// symbology. Structurally UPC-A is EAN-13 with an implied leading zero: the
// 95-module layout is start guard (101), six L-code (odd parity) left digits,
// centre guard (01010), six R-code right digits, end guard (101). The twelfth
// digit is a modulo-10 check digit. [EncodeUPCA] and [DecodeUPCA] reuse the
// shared eanL/eanR tables and form a matched pair.

// upcaChecksum returns the UPC-A check digit for the first eleven digits.
func upcaChecksum(d []int) int {
	sum := 0
	for i := 0; i < 11; i++ {
		if i%2 == 0 {
			sum += 3 * d[i]
		} else {
			sum += d[i]
		}
	}
	return (10 - sum%10) % 10
}

// upcaModules builds the 95-module bar pattern for twelve digits.
func upcaModules(d []int) []bool {
	m := make([]bool, 0, 95)
	put := func(pattern, bits int) {
		for i := bits - 1; i >= 0; i-- {
			m = append(m, (pattern>>i)&1 != 0)
		}
	}
	put(0x5, 3) // start guard 101
	for i := 0; i < 6; i++ {
		put(eanL[d[i]], 7)
	}
	put(0x0A, 5) // centre guard 01010
	for i := 0; i < 6; i++ {
		put(eanR[d[i+6]], 7)
	}
	put(0x5, 3) // end guard 101
	return m
}

// EncodeUPCA renders digits as a UPC-A barcode and returns it as a
// single-channel grayscale [cv.Mat] (bars 0, spaces 255) with quiet zones.
// digits must be 11 digits (the check digit is computed) or 12 digits (the
// check digit is validated). It returns an error for malformed input.
func EncodeUPCA(digits string) (*cv.Mat, error) {
	d, err := parseDigits(digits)
	if err != nil {
		return nil, err
	}
	switch len(d) {
	case 11:
		d = append(d, upcaChecksum(d))
	case 12:
		if upcaChecksum(d[:11]) != d[11] {
			return nil, fmt.Errorf("barcode: UPC-A check digit is %d, want %d", d[11], upcaChecksum(d[:11]))
		}
	default:
		return nil, fmt.Errorf("barcode: UPC-A needs 11 or 12 digits, got %d", len(d))
	}
	return renderModules1D(upcaModules(d)), nil
}

// DecodeUPCA scans a rendered UPC-A barcode and returns the twelve-digit string
// and true on success, or ("", false) if no valid symbol is found. It samples
// the 95 modules of a middle scanline, checks the guard patterns, decodes the
// six L-code and six R-code digits, and verifies the check digit.
func DecodeUPCA(img *cv.Mat) (string, bool) {
	mod, ok := sampleFixed1D(img, 95)
	if !ok {
		return "", false
	}
	if !(mod[0] && !mod[1] && mod[2]) ||
		!(!mod[45] && mod[46] && !mod[47] && mod[48] && !mod[49]) ||
		!(mod[92] && !mod[93] && mod[94]) {
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
	digits := make([]int, 12)
	for i := 0; i < 6; i++ {
		d := matchDigit(group(3+i*7), &eanL)
		if d < 0 {
			return "", false
		}
		digits[i] = d
	}
	for i := 0; i < 6; i++ {
		d := matchDigit(group(50+i*7), &eanR)
		if d < 0 {
			return "", false
		}
		digits[i+6] = d
	}
	if upcaChecksum(digits[:11]) != digits[11] {
		return "", false
	}
	out := make([]byte, 12)
	for i, d := range digits {
		out[i] = byte('0' + d)
	}
	return string(out), true
}
