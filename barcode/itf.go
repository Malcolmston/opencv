package barcode

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// This file implements Interleaved 2 of 5 (ITF), a dense numeric-only symbology
// that encodes digits in pairs: the five elements of the first digit are drawn
// as bars while the five elements of the second are interleaved as the spaces
// between them. Each digit's five-element pattern has exactly two wide elements.
// A symbol begins with a narrow start guard (bar space bar space) and ends with
// a stop guard (wide bar, narrow space, narrow bar). ITF requires an even number
// of digits; [EncodeITF] pads a leading zero when needed. [EncodeITF] and
// [DecodeITF] form a matched pair rendered with a 3:1 wide-to-narrow ratio.

// itfDigits gives the five-element narrow/wide pattern of each digit, '0' =
// narrow and '1' = wide.
var itfDigits = [10]string{
	"00110", "10001", "01001", "11000", "00101",
	"10100", "01100", "00011", "10010", "01010",
}

// itfDigitRev maps a five-element pattern back to its digit, or -1.
func itfDigitRev(pattern string) int {
	for d, p := range itfDigits {
		if p == pattern {
			return d
		}
	}
	return -1
}

// EncodeITF renders a numeric string as an Interleaved 2 of 5 barcode and
// returns it as a single-channel grayscale [cv.Mat] (bars 0, spaces 255) with
// quiet zones. digits must be all decimal digits; an odd count is left-padded
// with a single zero. It returns an error for non-digit input.
func EncodeITF(digits string) (*cv.Mat, error) {
	for _, c := range digits {
		if c < '0' || c > '9' {
			return nil, fmt.Errorf("barcode: ITF input %q has a non-digit", digits)
		}
	}
	if len(digits) == 0 {
		return nil, fmt.Errorf("barcode: ITF input is empty")
	}
	if len(digits)%2 != 0 {
		digits = "0" + digits
	}
	var mods []bool
	// Start guard: narrow bar, space, bar, space.
	mods = appendElement(mods, true, false)
	mods = appendElement(mods, false, false)
	mods = appendElement(mods, true, false)
	mods = appendElement(mods, false, false)
	for i := 0; i < len(digits); i += 2 {
		barPat := itfDigits[digits[i]-'0']
		spacePat := itfDigits[digits[i+1]-'0']
		for k := 0; k < 5; k++ {
			mods = appendElement(mods, true, barPat[k] == '1')
			mods = appendElement(mods, false, spacePat[k] == '1')
		}
	}
	// Stop guard: wide bar, narrow space, narrow bar.
	mods = appendElement(mods, true, true)
	mods = appendElement(mods, false, false)
	mods = appendElement(mods, true, false)
	return renderModules1D(mods), nil
}

// DecodeITF scans a rendered Interleaved 2 of 5 barcode and returns the decoded
// digit string and true on success, or ("", false) if no valid symbol is found.
// It recovers the module array from a middle scanline, classifies elements as
// narrow or wide, verifies the start and stop guards, and de-interleaves each
// digit pair.
func DecodeITF(img *cv.Mat) (string, bool) {
	mods, ok := recoverModules1D(img)
	if !ok {
		return "", false
	}
	e := elementsNW(mods)
	if len(e) < 4+10+3 {
		return "", false
	}
	// Start guard: four narrow elements.
	for k := 0; k < 4; k++ {
		if e[k] {
			return "", false
		}
	}
	// Stop guard: wide bar, narrow space, narrow bar.
	n := len(e)
	if !e[n-3] || e[n-2] || e[n-1] {
		return "", false
	}
	mid := e[4 : n-3]
	if len(mid)%10 != 0 {
		return "", false
	}
	var out []byte
	for p := 0; p < len(mid); p += 10 {
		var barPat, spacePat [5]byte
		for k := 0; k < 5; k++ {
			if mid[p+2*k] {
				barPat[k] = '1'
			} else {
				barPat[k] = '0'
			}
			if mid[p+2*k+1] {
				spacePat[k] = '1'
			} else {
				spacePat[k] = '0'
			}
		}
		d1 := itfDigitRev(string(barPat[:]))
		d2 := itfDigitRev(string(spacePat[:]))
		if d1 < 0 || d2 < 0 {
			return "", false
		}
		out = append(out, byte('0'+d1), byte('0'+d2))
	}
	return string(out), true
}
