package barcode

import (
	"fmt"
	"strings"

	cv "github.com/malcolmston/opencv"
)

// This file implements Code 11 (USD-8), a numeric-plus-dash symbology used in
// telecommunications labelling. Each character is five elements — three bars and
// two spaces, alternating and beginning and ending with a bar — separated by a
// single narrow space, and framed by a dedicated start/stop character. A single
// weighted modulo-11 "C" check character is appended. [EncodeCode11] and
// [DecodeCode11] form a matched pair rendered with a 3:1 wide-to-narrow ratio.

// code11Chars lists the encodable characters in value order; value 10 is '-'.
const code11Chars = "0123456789-"

// code11Patterns holds the five-element narrow/wide pattern ('1' = wide, ordered
// bar, space, bar, space, bar) of each value 0-10, and the start/stop pattern.
var code11Patterns = []string{
	"00001", "10001", "01001", "11000", "00101", // 0-4
	"10100", "01100", "00011", "10010", "10000", // 5-9
	"00100", // '-'
}

const code11StartStop = "00110"

// code11Rev maps a five-element pattern to its value (0-10), or -1.
func code11Rev(pattern string) int {
	for v, p := range code11Patterns {
		if p == pattern {
			return v
		}
	}
	return -1
}

// code11Value returns the value of a Code 11 character, or -1.
func code11Value(r byte) int {
	return strings.IndexByte(code11Chars, r)
}

// code11Check returns the weighted modulo-11 "C" check value for the data.
func code11Check(values []int) int {
	sum := 0
	weight := 1
	for i := len(values) - 1; i >= 0; i-- {
		sum += values[i] * weight
		weight++
		if weight > 10 {
			weight = 1
		}
	}
	return sum % 11
}

// code11AppendChar appends the five elements of a character pattern to mods.
func code11AppendChar(mods []bool, pattern string) []bool {
	bar := true
	for _, ch := range pattern {
		mods = appendElement(mods, bar, ch == '1')
		bar = !bar
	}
	return mods
}

// EncodeCode11 renders text as a Code 11 barcode and returns it as a
// single-channel grayscale [cv.Mat] (bars 0, spaces 255) with quiet zones. text
// must contain only digits and the dash '-'. The start/stop frame and the
// modulo-11 check character are added automatically. It returns an error for any
// unencodable or empty input.
func EncodeCode11(text string) (*cv.Mat, error) {
	if text == "" {
		return nil, fmt.Errorf("barcode: Code 11 input is empty")
	}
	values := make([]int, 0, len(text))
	for i := 0; i < len(text); i++ {
		v := code11Value(text[i])
		if v < 0 {
			return nil, fmt.Errorf("barcode: Code 11 cannot encode %q", text[i])
		}
		values = append(values, v)
	}
	check := code11Check(values)
	seq := append(append([]int{}, values...), check)

	mods := code11AppendChar(nil, code11StartStop)
	mods = append(mods, false) // gap
	for _, v := range seq {
		mods = code11AppendChar(mods, code11Patterns[v])
		mods = append(mods, false) // gap
	}
	mods = code11AppendChar(mods, code11StartStop)
	return renderModules1D(mods), nil
}

// DecodeCode11 scans a rendered Code 11 barcode and returns the decoded text
// (without the framing or check character) and true on success, or ("", false)
// if no valid symbol is found. It recovers the module array from a middle
// scanline, classifies elements as narrow or wide, splits them into
// five-element characters separated by narrow gaps, verifies the start/stop
// frame and the modulo-11 check character.
func DecodeCode11(img *cv.Mat) (string, bool) {
	mods, ok := recoverModules1D(img)
	if !ok {
		return "", false
	}
	e := elementsNW(mods)
	var pats []string
	i := 0
	for i+5 <= len(e) {
		var b strings.Builder
		for k := 0; k < 5; k++ {
			if e[i+k] {
				b.WriteByte('1')
			} else {
				b.WriteByte('0')
			}
		}
		pats = append(pats, b.String())
		i += 5
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
	if len(pats) < 4 { // start, >=1 data, check, stop
		return "", false
	}
	if pats[0] != code11StartStop || pats[len(pats)-1] != code11StartStop {
		return "", false
	}
	inner := pats[1 : len(pats)-1]
	vals := make([]int, len(inner))
	for j, p := range inner {
		v := code11Rev(p)
		if v < 0 {
			return "", false
		}
		vals[j] = v
	}
	data := vals[:len(vals)-1]
	if code11Check(data) != vals[len(vals)-1] {
		return "", false
	}
	out := make([]byte, len(data))
	for j, v := range data {
		out[j] = code11Chars[v]
	}
	return string(out), true
}
