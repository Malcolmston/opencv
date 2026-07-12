package barcode

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// This file implements Code 93, a compact alphanumeric symbology. Every
// character is nine modules wide — three bars and three spaces — and carries no
// inter-character gap, so patterns are represented directly as nine-bit module
// strings (bar = 1). A symbol is framed by the start/stop character '*',
// followed by two weighted modulo-47 check characters (C then K) and a final
// termination bar. This package encodes the 43 standard characters (the same set
// as Code 39). [EncodeCode93] and [DecodeCode93] form a matched pair; the two
// check characters are computed on encoding and verified on decoding.

// code93Alphabet lists the 43 encodable characters in value order (value = index
// into this string), which is also the order used by the check-character sums.
const code93Alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%"

// code93Patterns holds the nine-module bit pattern (bar = '1') of each of the 47
// Code 93 values. Values 0-42 are the encodable characters in code93Alphabet
// order; values 43-46 are the four control (shift) characters, which never
// appear in data but can arise as the modulo-47 check characters. The '*'
// start/stop character is kept separately in code93StartStop.
var code93Patterns = []string{
	"100010100", "101001000", "101000100", "101000010", "100101000", // 0-4
	"100100100", "100100010", "101010000", "100010010", "100001010", // 5-9
	"110101000", "110100100", "110100010", "110010100", "110010010", // A-E
	"110001010", "101101000", "101100100", "101100010", "100110100", // F-J
	"100011010", "101011000", "101001100", "101000110", "100101100", // K-O
	"100010110", "110110100", "110110010", "110101100", "110100110", // P-T
	"110010110", "110011010", "101101100", "101100110", "100110110", // U-Y
	"100111010", "100101110", "111010100", "111010010", "111001010", // Z - . space $
	"101101110", "101110110", "110101110", // / + %
	"100100110", "111011010", "111010110", "100110010", // 43-46 control chars
}

// code93StartStop is the pattern of the '*' start/stop character.
const code93StartStop = "101011110"

// code93Value returns the Code 93 value of a character, or -1 if unencodable.
func code93Value(r byte) int {
	for i := 0; i < len(code93Alphabet); i++ {
		if code93Alphabet[i] == r {
			return i
		}
	}
	return -1
}

// code93Checks computes the two check characters (C then K) for the given data
// values, using the weighted modulo-47 sums of the specification.
func code93Checks(values []int) (c, k int) {
	weight := 1
	sum := 0
	for i := len(values) - 1; i >= 0; i-- {
		sum += values[i] * weight
		weight++
		if weight > 20 {
			weight = 1
		}
	}
	c = sum % 47
	with := append(append([]int{}, values...), c)
	weight = 1
	sum = 0
	for i := len(with) - 1; i >= 0; i-- {
		sum += with[i] * weight
		weight++
		if weight > 15 {
			weight = 1
		}
	}
	k = sum % 47
	return c, k
}

// appendCode93 appends a nine-module pattern (bar = '1') to mods.
func appendCode93(mods []bool, pattern string) []bool {
	for i := 0; i < len(pattern); i++ {
		mods = append(mods, pattern[i] == '1')
	}
	return mods
}

// EncodeCode93 renders text as a Code 93 barcode and returns it as a
// single-channel grayscale [cv.Mat] (bars 0, spaces 255) with quiet zones. text
// must contain only the 43 standard Code 93 characters (0-9, A-Z, space and
// "-.$/+%") and no '*'. The start/stop frame, the two modulo-47 check characters
// and the termination bar are added automatically. It returns an error for any
// unencodable character.
func EncodeCode93(text string) (*cv.Mat, error) {
	if text == "" {
		return nil, fmt.Errorf("barcode: Code 93 input is empty")
	}
	values := make([]int, 0, len(text))
	for i := 0; i < len(text); i++ {
		v := code93Value(text[i])
		if v < 0 {
			return nil, fmt.Errorf("barcode: Code 93 cannot encode %q", text[i])
		}
		values = append(values, v)
	}
	c, k := code93Checks(values)

	var mods []bool
	mods = appendCode93(mods, code93StartStop)
	for _, v := range values {
		mods = appendCode93(mods, code93Patterns[v])
	}
	mods = appendCode93(mods, code93Patterns[c])
	mods = appendCode93(mods, code93Patterns[k])
	mods = appendCode93(mods, code93StartStop)
	mods = append(mods, true) // termination bar
	return renderModules1D(mods), nil
}

// code93PatternRev resolves a nine-module bit pattern to its value, or -1.
func code93PatternRev(pattern string) int {
	for v, p := range code93Patterns {
		if p == pattern {
			return v
		}
	}
	return -1
}

// DecodeCode93 scans a rendered Code 93 barcode and returns the decoded text
// (without the framing, check characters or termination bar) and true on
// success, or ("", false) if no valid symbol is found. It recovers the module
// array from a middle scanline, splits it into nine-module characters, verifies
// the '*' frame and both modulo-47 check characters.
func DecodeCode93(img *cv.Mat) (string, bool) {
	mods, ok := recoverModules1D(img)
	if !ok {
		return "", false
	}
	if len(mods) < 1 {
		return "", false
	}
	// Drop the final termination bar, then require a whole number of 9-module
	// characters.
	body := mods[:len(mods)-1]
	if len(body)%9 != 0 {
		return "", false
	}
	nchar := len(body) / 9
	if nchar < 5 { // start, >=1 data, C, K, stop
		return "", false
	}
	pats := make([]string, nchar)
	for i := 0; i < nchar; i++ {
		pat := make([]byte, 9)
		for j := 0; j < 9; j++ {
			if body[i*9+j] {
				pat[j] = '1'
			} else {
				pat[j] = '0'
			}
		}
		pats[i] = string(pat)
	}
	if pats[0] != code93StartStop || pats[nchar-1] != code93StartStop {
		return "", false
	}
	// Interior characters (data plus the two check characters) map to values.
	vals := make([]int, nchar-2)
	for i := 1; i < nchar-1; i++ {
		v := code93PatternRev(pats[i])
		if v < 0 {
			return "", false
		}
		vals[i-1] = v
	}
	data := vals[:len(vals)-2]
	gotC, gotK := vals[len(vals)-2], vals[len(vals)-1]
	wantC, wantK := code93Checks(data)
	if gotC != wantC || gotK != wantK {
		return "", false
	}
	out := make([]byte, len(data))
	for i, v := range data {
		if v < 0 || v >= len(code93Alphabet) {
			return "", false
		}
		out[i] = code93Alphabet[v]
	}
	return string(out), true
}
