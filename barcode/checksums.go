package barcode

import (
	"errors"
	"strings"
)

// This file provides exported check-digit computation and validation helpers
// for the linear symbologies handled elsewhere in the package. They operate on
// digit/character strings rather than images so that callers can validate a
// payload before encoding or after decoding, independently of any [cv.Mat].
// Every routine is pure, deterministic and uses the standard-library only.

// errNonDigit is returned when a payload expected to be numeric contains a
// non-decimal character.
var errNonDigit = errors.New("barcode: payload must contain only decimal digits")

// errLength is returned when a payload does not have the length a symbology
// requires.
var errLength = errors.New("barcode: payload has the wrong length")

// digitsOf converts a decimal string to a slice of its digit values, reporting
// false if any rune is not a decimal digit.
func digitsOf(s string) ([]int, bool) {
	out := make([]int, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c < '0' || c > '9' {
			return nil, false
		}
		out[i] = int(c - '0')
	}
	return out, true
}

// GS1CheckDigit computes the standard GS1 modulo-10 check digit for a numeric
// payload (the code without its trailing check digit). Digits are weighted 3
// and 1 alternately starting from the rightmost payload digit, and the check
// digit is the amount needed to raise the weighted sum to a multiple of ten.
// This is the check digit used by EAN-13, EAN-8, UPC-A, GTIN and ITF-14. It
// returns an error if the payload is empty or contains a non-digit.
func GS1CheckDigit(payload string) (int, error) {
	ds, ok := digitsOf(payload)
	if !ok {
		return 0, errNonDigit
	}
	if len(ds) == 0 {
		return 0, errLength
	}
	sum, weight := 0, 3
	for i := len(ds) - 1; i >= 0; i-- {
		sum += ds[i] * weight
		weight = 4 - weight // toggles 3 <-> 1
	}
	return (10 - sum%10) % 10, nil
}

// EAN13CheckDigit returns the modulo-10 check digit of a 12-digit EAN-13
// payload. It returns an error if the payload is not exactly 12 decimal digits.
func EAN13CheckDigit(payload string) (int, error) {
	if len(payload) != 12 {
		return 0, errLength
	}
	return GS1CheckDigit(payload)
}

// ValidateEAN13 reports whether a full 13-digit EAN-13 code has a correct check
// digit. Any non-13-digit or non-numeric input yields false.
func ValidateEAN13(code string) bool {
	if len(code) != 13 {
		return false
	}
	want, err := GS1CheckDigit(code[:12])
	return err == nil && int(code[12]-'0') == want
}

// EAN8CheckDigit returns the modulo-10 check digit of a 7-digit EAN-8 payload.
// It returns an error if the payload is not exactly 7 decimal digits.
func EAN8CheckDigit(payload string) (int, error) {
	if len(payload) != 7 {
		return 0, errLength
	}
	return GS1CheckDigit(payload)
}

// ValidateEAN8 reports whether a full 8-digit EAN-8 code has a correct check
// digit. Any non-8-digit or non-numeric input yields false.
func ValidateEAN8(code string) bool {
	if len(code) != 8 {
		return false
	}
	want, err := GS1CheckDigit(code[:7])
	return err == nil && int(code[7]-'0') == want
}

// UPCACheckDigit returns the modulo-10 check digit of an 11-digit UPC-A
// payload. It returns an error if the payload is not exactly 11 decimal digits.
func UPCACheckDigit(payload string) (int, error) {
	if len(payload) != 11 {
		return 0, errLength
	}
	return GS1CheckDigit(payload)
}

// ValidateUPCA reports whether a full 12-digit UPC-A code has a correct check
// digit. Any non-12-digit or non-numeric input yields false.
func ValidateUPCA(code string) bool {
	if len(code) != 12 {
		return false
	}
	want, err := GS1CheckDigit(code[:11])
	return err == nil && int(code[11]-'0') == want
}

// ITF14CheckDigit returns the modulo-10 check digit of a 13-digit ITF-14
// payload. It returns an error if the payload is not exactly 13 decimal digits.
func ITF14CheckDigit(payload string) (int, error) {
	if len(payload) != 13 {
		return 0, errLength
	}
	return GS1CheckDigit(payload)
}

// ValidateITF14 reports whether a full 14-digit ITF-14 code has a correct check
// digit. Any non-14-digit or non-numeric input yields false.
func ValidateITF14(code string) bool {
	if len(code) != 14 {
		return false
	}
	want, err := GS1CheckDigit(code[:13])
	return err == nil && int(code[13]-'0') == want
}

// LuhnChecksum returns the Luhn (modulo-10, "double every second digit") check
// digit for a numeric payload that does not yet include the check digit. It
// returns an error if the payload is empty or contains a non-digit. This is the
// algorithm used by MSI Modulo-10 and by most payment-card numbers.
func LuhnChecksum(payload string) (int, error) {
	ds, ok := digitsOf(payload)
	if !ok {
		return 0, errNonDigit
	}
	if len(ds) == 0 {
		return 0, errLength
	}
	sum := 0
	double := true
	for i := len(ds) - 1; i >= 0; i-- {
		d := ds[i]
		if double {
			d *= 2
			if d > 9 {
				d -= 9
			}
		}
		sum += d
		double = !double
	}
	return (10 - sum%10) % 10, nil
}

// ValidateLuhn reports whether a full numeric string (payload followed by its
// Luhn check digit) is self-consistent. Empty or non-numeric input yields
// false.
func ValidateLuhn(code string) bool {
	if len(code) < 2 {
		return false
	}
	want, err := LuhnChecksum(code[:len(code)-1])
	return err == nil && int(code[len(code)-1]-'0') == want
}

// MSICheckDigitMod10 returns the MSI Modulo-10 check digit of a numeric
// payload. MSI Modulo-10 is defined to be the Luhn checksum, so this is a
// thin, self-documenting alias of [LuhnChecksum].
func MSICheckDigitMod10(payload string) (int, error) {
	return LuhnChecksum(payload)
}

// code39Charset lists the 43 Code 39 data characters in value order; the index
// of a character is its modulo-43 value.
const code39Charset = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%"

// Code39CheckChar computes the Code 39 modulo-43 check character for a message
// made of the 43 valid Code 39 data characters (digits, uppercase letters and
// the symbols "-. $/+%"). The check character is the one whose value equals the
// sum of the message character values modulo 43. It returns an error if the
// message is empty or contains a character outside the Code 39 set.
func Code39CheckChar(message string) (byte, error) {
	if len(message) == 0 {
		return 0, errLength
	}
	sum := 0
	for i := 0; i < len(message); i++ {
		v := strings.IndexByte(code39Charset, message[i])
		if v < 0 {
			return 0, errors.New("barcode: character not in the Code 39 set")
		}
		sum += v
	}
	return code39Charset[sum%43], nil
}
