package barcode

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// This file implements MSI (Modified Plessey), a numeric-only symbology used on
// warehouse shelf labels. Each decimal digit is encoded as four bits, most
// significant first, and every bit becomes a three-module bar/space group: a
// '1' bit is a wide bar then a narrow space ("110") and a '0' bit is a narrow
// bar then a wide space ("100"). A symbol opens with a start pattern ("110") and
// closes with a stop pattern ("1001"). A trailing Luhn (modulo-10) check digit
// is appended. [EncodeMSI] and [DecodeMSI] form a matched pair.

const (
	msiStart = "110"
	msiStop  = "1001"
	msiOne   = "110"
	msiZero  = "100"
)

// msiChecksum returns the Luhn (modulo-10) check digit of the given digits.
func msiChecksum(d []int) int {
	sum := 0
	dbl := true
	for i := len(d) - 1; i >= 0; i-- {
		v := d[i]
		if dbl {
			v *= 2
			if v > 9 {
				v -= 9
			}
		}
		sum += v
		dbl = !dbl
	}
	return (10 - sum%10) % 10
}

// msiAppendBits appends a "1"/"0" module string to mods (bar = '1').
func msiAppendBits(mods []bool, s string) []bool {
	for i := 0; i < len(s); i++ {
		mods = append(mods, s[i] == '1')
	}
	return mods
}

// msiAppendDigit appends the four-bit encoding of a digit (MSB first).
func msiAppendDigit(mods []bool, d int) []bool {
	for b := 3; b >= 0; b-- {
		if (d>>b)&1 != 0 {
			mods = msiAppendBits(mods, msiOne)
		} else {
			mods = msiAppendBits(mods, msiZero)
		}
	}
	return mods
}

// EncodeMSI renders digits as an MSI (Modified Plessey) barcode and returns it
// as a single-channel grayscale [cv.Mat] (bars 0, spaces 255) with quiet zones.
// digits must be all decimal digits; a Luhn modulo-10 check digit is appended
// automatically. It returns an error for non-digit or empty input.
func EncodeMSI(digits string) (*cv.Mat, error) {
	d, err := parseDigits(digits)
	if err != nil {
		return nil, err
	}
	if len(d) == 0 {
		return nil, fmt.Errorf("barcode: MSI input is empty")
	}
	d = append(d, msiChecksum(d))
	mods := msiAppendBits(nil, msiStart)
	for _, digit := range d {
		mods = msiAppendDigit(mods, digit)
	}
	mods = msiAppendBits(mods, msiStop)
	return renderModules1D(mods), nil
}

// DecodeMSI scans a rendered MSI barcode and returns the digit string
// (including the trailing check digit) and true on success, or ("", false) if no
// valid symbol is found. It recovers the module array from a middle scanline,
// strips the start and stop patterns, decodes each four-bit digit and verifies
// the Luhn check digit.
func DecodeMSI(img *cv.Mat) (string, bool) {
	mods, ok := recoverModules1D(img)
	if !ok {
		return "", false
	}
	// start (3) + n*12 + stop (4)
	if len(mods) < len(msiStart)+12+len(msiStop) {
		return "", false
	}
	body := len(mods) - len(msiStart) - len(msiStop)
	if body%12 != 0 {
		return "", false
	}
	// Verify start and stop.
	if !modsEqual(mods[:3], msiStart) || !modsEqual(mods[len(mods)-4:], msiStop) {
		return "", false
	}
	n := body / 12
	digits := make([]int, n)
	pos := len(msiStart)
	for i := 0; i < n; i++ {
		v := 0
		for b := 0; b < 4; b++ {
			grp := mods[pos : pos+3]
			pos += 3
			switch {
			case modsEqual(grp, msiOne):
				v = v<<1 | 1
			case modsEqual(grp, msiZero):
				v = v << 1
			default:
				return "", false
			}
		}
		digits[i] = v
	}
	if n < 2 {
		return "", false
	}
	if msiChecksum(digits[:n-1]) != digits[n-1] {
		return "", false
	}
	out := make([]byte, n)
	for i, d := range digits {
		if d > 9 {
			return "", false
		}
		out[i] = byte('0' + d)
	}
	return string(out), true
}

// modsEqual reports whether a boolean module slice equals a "1"/"0" pattern.
func modsEqual(mods []bool, pattern string) bool {
	if len(mods) != len(pattern) {
		return false
	}
	for i := 0; i < len(pattern); i++ {
		if mods[i] != (pattern[i] == '1') {
			return false
		}
	}
	return true
}
