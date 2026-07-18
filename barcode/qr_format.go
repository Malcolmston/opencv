package barcode

import (
	"errors"
	"fmt"
	"math/bits"
)

// This file exposes the QR "metadata" codecs that were previously only used
// internally by the encoder: the BCH(15,5) format information (error-correction
// level plus data mask), the BCH(18,6) version information carried by versions
// 7 and above, the version/size relationship, and the alignment-pattern centre
// coordinates of ISO/IEC 18004 Annex E. Making them public lets callers decode
// the metadata of a QR symbol they have sampled without running the full
// decoder, and recover it even when a few of the bits are damaged.

// String returns the single-letter name of the error-correction level ("L",
// "M", "Q" or "H").
func (l QRECCLevel) String() string {
	switch l {
	case QRECCLow:
		return "L"
	case QRECCMedium:
		return "M"
	case QRECCQuartile:
		return "Q"
	case QRECCHigh:
		return "H"
	default:
		return fmt.Sprintf("QRECCLevel(%d)", int(l))
	}
}

// QRFormatInfo is the decoded 5-bit QR format information: the error-correction
// level and the data-mask pattern number (0-7).
type QRFormatInfo struct {
	// Level is the error-correction level encoded in the format bits.
	Level QRECCLevel
	// Mask is the data-mask pattern number, in the range 0-7.
	Mask int
}

// String renders the format information as e.g. "M/mask3".
func (f QRFormatInfo) String() string {
	return fmt.Sprintf("%s/mask%d", f.Level, f.Mask)
}

// EncodeQRFormatInfo returns the 15-bit BCH-protected, XOR-masked format
// information value for the given error-correction level and data-mask number.
// It returns an error if the mask is not in the range 0-7. The result matches
// the bit pattern written into a QR symbol's two format-information regions.
func EncodeQRFormatInfo(level QRECCLevel, mask int) (int, error) {
	if mask < 0 || mask > 7 {
		return 0, errors.New("barcode: QR data-mask must be in the range 0-7")
	}
	return formatBits(levelField(level), mask), nil
}

// DecodeQRFormatInfo recovers the error-correction level and mask from a 15-bit
// format-information value, tolerating up to three bit errors by choosing the
// valid codeword with the smallest Hamming distance. It returns the decoded
// information and true, or the zero value and false when no codeword lies
// within the correction radius.
func DecodeQRFormatInfo(raw int) (QRFormatInfo, bool) {
	raw &= 0x7FFF
	bestDist := 16
	var best QRFormatInfo
	for _, field := range []int{0, 1, 2, 3} {
		for mask := 0; mask < 8; mask++ {
			cand := formatBits(field, mask)
			d := bits.OnesCount(uint(raw ^ cand))
			if d < bestDist {
				bestDist = d
				best = QRFormatInfo{Level: fieldLevel(field), Mask: mask}
			}
		}
	}
	if bestDist > 3 {
		return QRFormatInfo{}, false
	}
	return best, true
}

// qrVersionInfoBits computes the 18-bit BCH-protected version information for a
// version (valid only for 7-40; the 6 data bits are the version number and the
// 12 check bits use the generator polynomial 0x1F25).
func qrVersionInfoBits(version int) int {
	rem := version
	for i := 0; i < 12; i++ {
		rem = (rem << 1) ^ ((rem >> 11) * 0x1F25)
	}
	return version<<12 | (rem & 0xFFF)
}

// EncodeQRVersionInfo returns the 18-bit BCH-protected version-information value
// carried by QR versions 7 through 40. It returns an error for versions below 7
// (which carry no version information) or above 40.
func EncodeQRVersionInfo(version int) (int, error) {
	if version < 7 || version > 40 {
		return 0, errors.New("barcode: QR version information exists only for versions 7-40")
	}
	return qrVersionInfoBits(version), nil
}

// DecodeQRVersionInfo recovers a QR version number (7-40) from an 18-bit
// version-information value, tolerating up to three bit errors by choosing the
// version whose codeword has the smallest Hamming distance. It returns the
// version and true, or 0 and false when no version lies within the correction
// radius.
func DecodeQRVersionInfo(raw int) (int, bool) {
	raw &= 0x3FFFF
	bestDist := 19
	bestVer := 0
	for v := 7; v <= 40; v++ {
		d := bits.OnesCount(uint(raw ^ qrVersionInfoBits(v)))
		if d < bestDist {
			bestDist = d
			bestVer = v
		}
	}
	if bestDist > 3 {
		return 0, false
	}
	return bestVer, true
}

// QRSizeForVersion returns the side length in modules of a QR symbol of the
// given version, which is 17+4*version. It returns 0 for versions outside the
// valid range 1-40.
func QRSizeForVersion(version int) int {
	if version < 1 || version > 40 {
		return 0
	}
	return 17 + 4*version
}

// QRVersionForSize returns the version of a square QR symbol with the given
// side length in modules. It returns the version and true when size is a valid
// QR dimension (21, 25, ... 177), or 0 and false otherwise.
func QRVersionForSize(size int) (int, bool) {
	if size < 21 || size > 177 || (size-17)%4 != 0 {
		return 0, false
	}
	return (size - 17) / 4, true
}

// qrAlignmentTable holds the alignment-pattern centre coordinates for every QR
// version (index = version), per ISO/IEC 18004 Annex E. Version 1 has none.
var qrAlignmentTable = [41][]int{
	1:  nil,
	2:  {6, 18},
	3:  {6, 22},
	4:  {6, 26},
	5:  {6, 30},
	6:  {6, 34},
	7:  {6, 22, 38},
	8:  {6, 24, 42},
	9:  {6, 26, 46},
	10: {6, 28, 50},
	11: {6, 30, 54},
	12: {6, 32, 58},
	13: {6, 34, 62},
	14: {6, 26, 46, 66},
	15: {6, 26, 48, 70},
	16: {6, 26, 50, 74},
	17: {6, 30, 54, 78},
	18: {6, 30, 56, 82},
	19: {6, 30, 58, 86},
	20: {6, 34, 62, 90},
	21: {6, 28, 50, 72, 94},
	22: {6, 26, 50, 74, 98},
	23: {6, 30, 54, 78, 102},
	24: {6, 28, 54, 80, 106},
	25: {6, 32, 58, 84, 110},
	26: {6, 30, 58, 86, 114},
	27: {6, 34, 62, 90, 118},
	28: {6, 26, 50, 74, 98, 122},
	29: {6, 30, 54, 78, 102, 126},
	30: {6, 26, 52, 78, 104, 130},
	31: {6, 30, 56, 82, 108, 134},
	32: {6, 34, 60, 86, 112, 138},
	33: {6, 30, 58, 86, 114, 142},
	34: {6, 34, 62, 90, 118, 146},
	35: {6, 30, 54, 78, 102, 126, 150},
	36: {6, 24, 50, 76, 102, 128, 154},
	37: {6, 28, 54, 80, 106, 132, 158},
	38: {6, 32, 58, 84, 110, 136, 162},
	39: {6, 26, 54, 82, 110, 138, 166},
	40: {6, 30, 58, 86, 114, 142, 170},
}

// QRAlignmentCenters returns the row/column centre coordinates at which a QR
// symbol of the given version places alignment patterns (an alignment pattern
// sits at every pairwise combination of these coordinates except where it would
// overlap a finder pattern). It returns a fresh slice, nil for version 1, and
// nil for versions outside 1-40.
func QRAlignmentCenters(version int) []int {
	if version < 1 || version > 40 {
		return nil
	}
	src := qrAlignmentTable[version]
	if len(src) == 0 {
		return nil
	}
	out := make([]int, len(src))
	copy(out, src)
	return out
}
