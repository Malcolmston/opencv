package barcode

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// This file implements EAN-13, the 13-digit retail barcode. An EAN-13 symbol is
// 95 modules wide: a start guard (101), six left-hand digits, a centre guard
// (01010), six right-hand digits and an end guard (101). The first of the 13
// digits is not drawn directly; it is encoded in the odd/even parity choice of
// the six left-hand digits. Both an encoder ([EncodeEAN13], for rendering test
// symbols and general use) and a scanline decoder ([DecodeEAN13]) are provided
// so the pair round-trips.

// eanL holds the 7-module L-code (odd parity) patterns for digits 0-9, as 7-bit
// integers with the most significant bit the leftmost module (1 = bar).
var eanL = [10]int{
	0x0D, 0x19, 0x13, 0x3D, 0x23, 0x31, 0x2F, 0x3B, 0x37, 0x0B,
}

// eanR (right / even-parity-complement) and eanG (even parity, left) are derived
// from eanL: R is the bitwise complement, G is R with its 7 bits reversed.
var (
	eanR [10]int
	eanG [10]int
)

// eanParity[d] is the parity pattern selected by first digit d for the six
// left-hand digits: bit 5 is the first left digit, bit 0 the sixth; a set bit
// means even parity (G-code), a clear bit means odd parity (L-code).
var eanParity = [10]int{
	0x00, 0x0B, 0x0D, 0x0E, 0x13, 0x19, 0x1C, 0x15, 0x16, 0x1A,
}

func init() {
	for d := 0; d < 10; d++ {
		eanR[d] = (^eanL[d]) & 0x7F
		eanG[d] = reverse7(eanR[d])
	}
}

// reverse7 reverses the low 7 bits of v.
func reverse7(v int) int {
	r := 0
	for i := 0; i < 7; i++ {
		r |= ((v >> i) & 1) << (6 - i)
	}
	return r
}

// ean13Checksum returns the EAN-13 check digit for the first 12 digits.
func ean13Checksum(d []int) int {
	sum := 0
	for i := 0; i < 12; i++ {
		if i%2 == 0 {
			sum += d[i]
		} else {
			sum += 3 * d[i]
		}
	}
	return (10 - sum%10) % 10
}

// parseDigits converts a numeric string into a slice of digit values, reporting
// an error on any non-digit character.
func parseDigits(s string) ([]int, error) {
	out := make([]int, len(s))
	for i, c := range s {
		if c < '0' || c > '9' {
			return nil, fmt.Errorf("barcode: EAN-13 input %q has a non-digit", s)
		}
		out[i] = int(c - '0')
	}
	return out, nil
}

// ean13Modules builds the 95-module bar pattern (true = bar) for 13 digits.
func ean13Modules(d []int) []bool {
	m := make([]bool, 0, 95)
	put := func(pattern, bits int) {
		for i := bits - 1; i >= 0; i-- {
			m = append(m, (pattern>>i)&1 != 0)
		}
	}
	put(0x5, 3) // start guard 101
	parity := eanParity[d[0]]
	for i := 0; i < 6; i++ {
		digit := d[i+1]
		if (parity>>(5-i))&1 != 0 {
			put(eanG[digit], 7)
		} else {
			put(eanL[digit], 7)
		}
	}
	put(0x0A, 5) // centre guard 01010
	for i := 0; i < 6; i++ {
		put(eanR[d[i+7]], 7)
	}
	put(0x5, 3) // end guard 101
	return m
}

// EAN13 rendering parameters.
const (
	eanModuleWidth = 3
	eanHeight      = 60
	eanQuiet       = 10 // quiet-zone modules on each side
)

// EncodeEAN13 renders digits as an EAN-13 barcode and returns it as a
// single-channel grayscale [cv.Mat] (bars are 0, spaces 255) with quiet zones.
// digits must be 12 digits (the check digit is computed) or 13 digits (the
// check digit is validated). It returns an error for malformed input.
func EncodeEAN13(digits string) (*cv.Mat, error) {
	d, err := parseDigits(digits)
	if err != nil {
		return nil, err
	}
	switch len(d) {
	case 12:
		d = append(d, ean13Checksum(d))
	case 13:
		if ean13Checksum(d) != d[12] {
			return nil, fmt.Errorf("barcode: EAN-13 check digit is %d, want %d", d[12], ean13Checksum(d))
		}
	default:
		return nil, fmt.Errorf("barcode: EAN-13 needs 12 or 13 digits, got %d", len(d))
	}
	modules := ean13Modules(d)
	totalModules := len(modules) + 2*eanQuiet
	w := totalModules * eanModuleWidth
	m := cv.NewMat(eanHeight, w, 1)
	m.SetTo(255)
	for i, bar := range modules {
		if !bar {
			continue
		}
		x0 := (i + eanQuiet) * eanModuleWidth
		for y := 0; y < eanHeight; y++ {
			for dx := 0; dx < eanModuleWidth; dx++ {
				m.Set(y, x0+dx, 0, 0)
			}
		}
	}
	return m, nil
}

// matchDigit returns the digit whose pattern in table equals code, or -1.
func matchDigit(code int, table *[10]int) int {
	for d := 0; d < 10; d++ {
		if table[d] == code {
			return d
		}
	}
	return -1
}

// DecodeEAN13 scans a rendered EAN-13 barcode and returns the 13-digit string
// and true on success, or ("", false) if no valid symbol is found. It reads a
// single horizontal scanline through the middle of the image, locates the
// barcode by its dark extent, samples the 95 modules, checks the guard patterns,
// recovers the first digit from the left-hand parity, and verifies the check
// digit.
func DecodeEAN13(img *cv.Mat) (string, bool) {
	if img == nil || img.Empty() {
		return "", false
	}
	gray := img
	if img.Channels != 1 {
		gray = cv.CvtColor(img, cv.ColorRGB2Gray)
	}
	bin, _ := cv.Threshold(gray, 0, 255, cv.ThreshBinaryInv|cv.ThreshOtsu)
	w := bin.Cols
	y := bin.Rows / 2
	first, last := -1, -1
	for x := 0; x < w; x++ {
		if bin.Data[y*w+x] != 0 {
			if first < 0 {
				first = x
			}
			last = x
		}
	}
	if first < 0 || last-first < 95 {
		return "", false
	}
	span := float64(last - first + 1)
	sample := func(k int) bool {
		x := first + int((float64(k)+0.5)*span/95)
		return bin.Data[y*w+x] != 0
	}
	mod := make([]bool, 95)
	for k := 0; k < 95; k++ {
		mod[k] = sample(k)
	}
	// Guard checks: start 101, centre 01010, end 101.
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
	digits := make([]int, 13)
	parity := 0
	for i := 0; i < 6; i++ {
		code := group(3 + i*7)
		if d := matchDigit(code, &eanL); d >= 0 {
			digits[i+1] = d
		} else if d := matchDigit(code, &eanG); d >= 0 {
			digits[i+1] = d
			parity |= 1 << (5 - i)
		} else {
			return "", false
		}
	}
	first13 := matchDigit(parity, &eanParity)
	if first13 < 0 {
		return "", false
	}
	digits[0] = first13
	for i := 0; i < 6; i++ {
		d := matchDigit(group(50+i*7), &eanR)
		if d < 0 {
			return "", false
		}
		digits[i+7] = d
	}
	if ean13Checksum(digits) != digits[12] {
		return "", false
	}
	out := make([]byte, 13)
	for i, d := range digits {
		out[i] = byte('0' + d)
	}
	return string(out), true
}
