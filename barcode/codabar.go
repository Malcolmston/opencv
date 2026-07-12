package barcode

import (
	"fmt"
	"strings"

	cv "github.com/malcolmston/opencv"
)

// This file implements Codabar (also known as NW-7 and USD-4), a self-checking
// numeric-plus-symbol symbology used on blood bags, film canisters and library
// cards. Each character is seven elements — four bars and three spaces,
// alternating and beginning and ending with a bar — separated from its
// neighbours by a single narrow space. A symbol is framed by one of the four
// start/stop characters A, B, C or D. [EncodeCodabar] and [DecodeCodabar] form a
// matched pair rendered with a 3:1 wide-to-narrow ratio.

// codabarPatterns maps each Codabar character to its seven-element narrow/wide
// pattern, '0' = narrow and '1' = wide, ordered bar, space, bar, ... (bar last).
var codabarPatterns = map[rune]string{
	'0': "0000011", '1': "0000110", '2': "0001001", '3': "1100000",
	'4': "0010010", '5': "1000010", '6': "0100001", '7': "0100100",
	'8': "0110000", '9': "1001000", '-': "0001100", '$': "0011000",
	':': "1000101", '/': "1010001", '.': "1010100", '+': "0010101",
	'A': "0011010", 'B': "0101001", 'C': "0001011", 'D': "0001110",
}

// codabarRev is the inverse of codabarPatterns.
var codabarRev = map[string]rune{}

func init() {
	for r, p := range codabarPatterns {
		codabarRev[p] = r
	}
}

// isCodabarStartStop reports whether r is one of the framing characters A-D.
func isCodabarStartStop(r rune) bool {
	return r == 'A' || r == 'B' || r == 'C' || r == 'D'
}

// EncodeCodabar renders text as a Codabar barcode and returns it as a
// single-channel grayscale [cv.Mat] (bars 0, spaces 255) with quiet zones. text
// must begin and end with a start/stop letter (A, B, C or D) and otherwise
// contain only digits and the symbols "-$:/.+"; the framing letters may not
// appear in the interior. It returns an error for malformed input.
func EncodeCodabar(text string) (*cv.Mat, error) {
	if len(text) < 3 {
		return nil, fmt.Errorf("barcode: Codabar needs a start letter, data and a stop letter")
	}
	runes := []rune(text)
	if !isCodabarStartStop(runes[0]) || !isCodabarStartStop(runes[len(runes)-1]) {
		return nil, fmt.Errorf("barcode: Codabar must be framed by start/stop letters A-D")
	}
	for i, r := range runes {
		if _, ok := codabarPatterns[r]; !ok {
			return nil, fmt.Errorf("barcode: Codabar cannot encode %q", r)
		}
		if i > 0 && i < len(runes)-1 && isCodabarStartStop(r) {
			return nil, fmt.Errorf("barcode: Codabar start/stop letter %q inside the data", r)
		}
	}
	var mods []bool
	for i, r := range runes {
		bar := true
		for _, ch := range codabarPatterns[r] {
			mods = appendElement(mods, bar, ch == '1')
			bar = !bar
		}
		if i != len(runes)-1 {
			mods = append(mods, false) // narrow inter-character gap
		}
	}
	return renderModules1D(mods), nil
}

// DecodeCodabar scans a rendered Codabar barcode and returns the decoded text
// (including the framing start/stop letters) and true on success, or ("", false)
// if no valid symbol is found. It recovers the module array from a middle
// scanline, classifies elements as narrow or wide, splits them into
// seven-element characters separated by narrow gaps, and requires A-D framing.
func DecodeCodabar(img *cv.Mat) (string, bool) {
	mods, ok := recoverModules1D(img)
	if !ok {
		return "", false
	}
	e := elementsNW(mods)
	var chars []rune
	i := 0
	for i+7 <= len(e) {
		var b strings.Builder
		for k := 0; k < 7; k++ {
			if e[i+k] {
				b.WriteByte('1')
			} else {
				b.WriteByte('0')
			}
		}
		r, ok := codabarRev[b.String()]
		if !ok {
			return "", false
		}
		chars = append(chars, r)
		i += 7
		if i < len(e) {
			if e[i] { // inter-character gap must be narrow
				return "", false
			}
			i++
		}
	}
	if i != len(e) {
		return "", false
	}
	if len(chars) < 3 || !isCodabarStartStop(chars[0]) || !isCodabarStartStop(chars[len(chars)-1]) {
		return "", false
	}
	return string(chars), true
}
