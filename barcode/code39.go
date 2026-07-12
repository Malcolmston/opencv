package barcode

import (
	"fmt"
	"strings"

	cv "github.com/malcolmston/opencv"
)

// This file implements Code 39 (also called Code 3 of 9), a self-checking,
// variable-length alphanumeric symbology. Each character is nine elements — five
// bars and four spaces, alternating and beginning with a bar — of which exactly
// three are wide and six narrow; characters are separated by a single narrow
// space. A symbol is framed by the start/stop character '*'. Because the pattern
// is self-checking, Code 39 carries no mandatory check digit. [EncodeCode39] and
// [DecodeCode39] form a matched pair rendered with a 3:1 wide-to-narrow ratio.

// code39Patterns maps each encodable character to its nine-element narrow/wide
// pattern, '0' = narrow and '1' = wide, ordered bar, space, bar, ... (bar last).
var code39Patterns = map[rune]string{
	'0': "000110100", '1': "100100001", '2': "001100001", '3': "101100000",
	'4': "000110001", '5': "100110000", '6': "001110000", '7': "000100101",
	'8': "100100100", '9': "001100100", 'A': "100001001", 'B': "001001001",
	'C': "101001000", 'D': "000011001", 'E': "100011000", 'F': "001011000",
	'G': "000001101", 'H': "100001100", 'I': "001001100", 'J': "000011100",
	'K': "100000011", 'L': "001000011", 'M': "101000010", 'N': "000010011",
	'O': "100010010", 'P': "001010010", 'Q': "000000111", 'R': "100000110",
	'S': "001000110", 'T': "000010110", 'U': "110000001", 'V': "011000001",
	'W': "111000000", 'X': "010010001", 'Y': "110010000", 'Z': "011010000",
	'-': "010000101", '.': "110000100", ' ': "011000100", '$': "010101000",
	'/': "010100010", '+': "010001010", '%': "000101010", '*': "010010100",
}

// code39Rev is the inverse of code39Patterns, resolving a nine-element pattern
// back to its character.
var code39Rev = map[string]rune{}

func init() {
	for r, p := range code39Patterns {
		code39Rev[p] = r
	}
}

// code39AppendChar appends the nine elements of one character pattern to mods,
// alternating bar/space and starting with a bar.
func code39AppendChar(mods []bool, pattern string) []bool {
	bar := true
	for _, ch := range pattern {
		mods = appendElement(mods, bar, ch == '1')
		bar = !bar
	}
	return mods
}

// EncodeCode39 renders text as a Code 39 barcode and returns it as a
// single-channel grayscale [cv.Mat] (bars 0, spaces 255) with quiet zones. The
// start/stop '*' characters are added automatically; text itself must contain
// only the 43 Code 39 characters (0-9, A-Z, space and "-.$/+%") and no '*'. It
// returns an error for any unencodable character.
func EncodeCode39(text string) (*cv.Mat, error) {
	if text == "" {
		return nil, fmt.Errorf("barcode: Code 39 input is empty")
	}
	for _, r := range text {
		if r == '*' {
			return nil, fmt.Errorf("barcode: Code 39 cannot encode the start/stop character '*'")
		}
		if _, ok := code39Patterns[r]; !ok {
			return nil, fmt.Errorf("barcode: Code 39 cannot encode %q", r)
		}
	}
	var mods []bool
	seq := "*" + text + "*"
	for i, r := range seq {
		mods = code39AppendChar(mods, code39Patterns[r])
		if i != len(seq)-1 {
			mods = append(mods, false) // narrow inter-character gap
		}
	}
	return renderModules1D(mods), nil
}

// DecodeCode39 scans a rendered Code 39 barcode and returns the decoded text
// (without the framing '*' characters) and true on success, or ("", false) if no
// valid symbol is found. It recovers the module array from a middle scanline,
// classifies elements as narrow or wide, splits them into nine-element
// characters separated by narrow gaps, and requires the '*' start/stop frame.
func DecodeCode39(img *cv.Mat) (string, bool) {
	mods, ok := recoverModules1D(img)
	if !ok {
		return "", false
	}
	e := elementsNW(mods)
	var chars []rune
	i := 0
	for i+9 <= len(e) {
		var b strings.Builder
		for k := 0; k < 9; k++ {
			if e[i+k] {
				b.WriteByte('1')
			} else {
				b.WriteByte('0')
			}
		}
		r, ok := code39Rev[b.String()]
		if !ok {
			return "", false
		}
		chars = append(chars, r)
		i += 9
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
	if len(chars) < 3 || chars[0] != '*' || chars[len(chars)-1] != '*' {
		return "", false
	}
	inner := chars[1 : len(chars)-1]
	for _, r := range inner {
		if r == '*' {
			return "", false
		}
	}
	return string(inner), true
}
