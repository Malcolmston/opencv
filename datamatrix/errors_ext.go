package datamatrix

import "errors"

var (
	// errBadCodewords is returned when a codeword stream ends unexpectedly or
	// contains a structurally invalid sequence during decoding.
	errBadCodewords = errors.New("datamatrix: malformed or truncated codeword stream")
	// errInvalidAppend is returned when structured-append parameters are out of
	// range (position/total must be 1..16, file identifiers 1..254).
	errInvalidAppend = errors.New("datamatrix: invalid structured-append parameters")
	// errInvalidECI is returned when an ECI value is outside the encodable range.
	errInvalidECI = errors.New("datamatrix: ECI value out of range (0..999999)")
	// errNoSymbols is returned by DecodeAll when no symbols are located.
	errNoSymbols = errors.New("datamatrix: no Data Matrix symbols found")
)
