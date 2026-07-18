package barcode

import "errors"

// This file provides exported conversions between the closely related GS1
// retail symbologies: the compressed UPC-E form and its expanded UPC-A
// equivalent, and the trivial embedding of UPC-A into EAN-13. They are pure
// string transforms with no image dependency.

// UPCEToUPCA expands an 8-digit UPC-E code (a number-system digit of 0 or 1,
// six data digits and a check digit) into its equivalent 12-digit UPC-A code.
// The expansion follows the standard rule keyed on the sixth data digit, and
// the returned UPC-A carries a freshly computed, correct check digit. It
// returns an error if the input is not 8 digits, uses a number system other
// than 0 or 1, or fails its own check digit.
func UPCEToUPCA(upce string) (string, error) {
	if len(upce) != 8 {
		return "", errLength
	}
	ds, ok := digitsOf(upce)
	if !ok {
		return "", errNonDigit
	}
	ns := ds[0]
	if ns != 0 && ns != 1 {
		return "", errors.New("barcode: UPC-E number system must be 0 or 1")
	}
	x := ds[1:7] // the six data digits X1..X6
	var body [10]int
	switch x[5] {
	case 0, 1, 2:
		body = [10]int{x[0], x[1], x[5], 0, 0, 0, 0, x[2], x[3], x[4]}
	case 3:
		body = [10]int{x[0], x[1], x[2], 0, 0, 0, 0, 0, x[3], x[4]}
	case 4:
		body = [10]int{x[0], x[1], x[2], x[3], 0, 0, 0, 0, 0, x[4]}
	default: // 5..9
		body = [10]int{x[0], x[1], x[2], x[3], x[4], 0, 0, 0, 0, x[5]}
	}
	payload := make([]byte, 0, 11)
	payload = append(payload, byte('0'+ns))
	for _, d := range body {
		payload = append(payload, byte('0'+d))
	}
	check, err := GS1CheckDigit(string(payload))
	if err != nil {
		return "", err
	}
	if ds[7] != check {
		return "", errors.New("barcode: UPC-E check digit is invalid")
	}
	return string(payload) + string(byte('0'+check)), nil
}

// UPCAToUPCE compresses a 12-digit UPC-A code into its 8-digit UPC-E form when
// the code is compressible (its number system is 0 or 1 and its
// manufacturer/product digits match one of the standard zero-suppression
// patterns). It returns an error if the input is not a valid 12-digit UPC-A or
// cannot be represented as UPC-E.
func UPCAToUPCE(upca string) (string, error) {
	if len(upca) != 12 {
		return "", errLength
	}
	ds, ok := digitsOf(upca)
	if !ok {
		return "", errNonDigit
	}
	if !ValidateUPCA(upca) {
		return "", errors.New("barcode: UPC-A check digit is invalid")
	}
	ns := ds[0]
	if ns != 0 && ns != 1 {
		return "", errors.New("barcode: UPC-A is not compressible (number system must be 0 or 1)")
	}
	// b is the 10-digit manufacturer/product body (positions 1..10).
	b := ds[1:11]
	var x [6]int
	switch {
	case b[2] <= 2 && b[3] == 0 && b[4] == 0 && b[5] == 0 && b[6] == 0:
		// X1 X2 X6 0000 X3 X4 X5, with X6 = b[2] in {0,1,2}.
		x = [6]int{b[0], b[1], b[7], b[8], b[9], b[2]}
	case b[3] == 0 && b[4] == 0 && b[5] == 0 && b[6] == 0 && b[7] == 0:
		// X1 X2 X3 00000 X4 X5, X6 = 3.
		x = [6]int{b[0], b[1], b[2], b[8], b[9], 3}
	case b[4] == 0 && b[5] == 0 && b[6] == 0 && b[7] == 0 && b[8] == 0:
		// X1 X2 X3 X4 00000 X5, X6 = 4.
		x = [6]int{b[0], b[1], b[2], b[3], b[9], 4}
	case b[5] == 0 && b[6] == 0 && b[7] == 0 && b[8] == 0 && b[9] >= 5:
		// X1 X2 X3 X4 X5 0000 X6, X6 = b[9] in {5..9}.
		x = [6]int{b[0], b[1], b[2], b[3], b[4], b[9]}
	default:
		return "", errors.New("barcode: UPC-A cannot be compressed to UPC-E")
	}
	out := make([]byte, 0, 8)
	out = append(out, byte('0'+ns))
	for _, d := range x {
		out = append(out, byte('0'+d))
	}
	out = append(out, byte('0'+ds[11]))
	return string(out), nil
}

// UPCAToEAN13 embeds a 12-digit UPC-A code into the 13-digit EAN-13 space by
// prefixing a leading zero. The check digit is unchanged because the GS1
// modulo-10 algorithm is invariant to a leading zero. It returns an error if
// the input is not a valid 12-digit UPC-A code.
func UPCAToEAN13(upca string) (string, error) {
	if len(upca) != 12 {
		return "", errLength
	}
	if !ValidateUPCA(upca) {
		return "", errors.New("barcode: UPC-A check digit is invalid")
	}
	return "0" + upca, nil
}
