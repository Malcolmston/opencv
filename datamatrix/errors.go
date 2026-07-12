package datamatrix

import "errors"

var (
	// errNonASCII is returned when the input contains a byte outside the
	// 0-127 ASCII range, which the ASCII encodation scheme cannot represent.
	errNonASCII = errors.New("datamatrix: input contains a non-ASCII byte (only ASCII encodation is supported)")
	// errTooLong is returned when the encoded data does not fit in the
	// largest supported square symbol.
	errTooLong = errors.New("datamatrix: data too long for the supported square symbol sizes")
	// errTooManyErrors is returned when Reed-Solomon decoding cannot recover
	// the codewords because the number of errors exceeds the correction
	// capacity.
	errTooManyErrors = errors.New("datamatrix: too many errors to correct")
	// errNotFound is returned when no valid Data Matrix symbol can be located
	// or sampled in the supplied image.
	errNotFound = errors.New("datamatrix: no Data Matrix symbol found")
	// errBadMatrix is returned when a supplied module grid is not a valid
	// square symbol of a supported size.
	errBadMatrix = errors.New("datamatrix: invalid or unsupported module matrix")
	// errUnsupportedMode is returned when a decoded codeword selects an
	// encodation mode (C40, Text, Base256, ...) that this package does not
	// implement; see the DEFERRED list in the package documentation.
	errUnsupportedMode = errors.New("datamatrix: unsupported encodation mode in symbol")
)
