// Package datamatrix is a self-contained, dependency-free ECC200 Data Matrix
// codec for the stdlib-only OpenCV port. It encodes ASCII strings into square
// Data Matrix symbol bitmaps and detects and decodes those bitmaps back into
// the original text, correcting a bounded number of errors along the way.
//
// # Overview
//
// The encoder ([Encode], [EncodeWithOptions]) turns a string into a *cv.Mat
// whose modules are drawn black on a white background. The pipeline is the
// genuine ECC200 pipeline:
//
//   - ASCII encodation: printable and control ASCII bytes map to codeword
//     value+1, and adjacent digit pairs are packed into a single codeword
//     (value+130), exactly as in ISO/IEC 16022.
//   - Symbol selection: the smallest supported square symbol whose data
//     capacity fits is chosen automatically.
//   - Padding: a single end-of-message codeword (129) followed by the
//     253-state randomised pad fills any remaining data capacity.
//   - Reed-Solomon: error-correction codewords are generated over GF(256)
//     using the ECC200 primitive polynomial 0x12D, generator element 2 and
//     first root exponent 1, reproducing the standard generator polynomials.
//   - Placement: codeword bits are laid into the mapping matrix with the
//     ECC200 "utah" placement algorithm (ISO/IEC 16022 Annex F), including
//     the corner special cases and the bottom-right fixed pattern.
//   - Finder/timing: a solid "L" (left column and bottom row) plus an
//     alternating clock track (top row and right column) frame the symbol.
//
// The decoder ([DetectAndDecode], [DecodeMatrix]) reverses this: it locates the
// symbol via its solid finder pattern, samples the module grid, reads the
// codewords with the same placement, runs a full Reed-Solomon syndrome decoder
// (Berlekamp-Massey error locator, Chien search, Forney magnitudes) to repair
// errors, and finally decodes the ASCII codewords back to text.
//
// # Supported symbol sizes
//
// The original [Encode]/[DecodeMatrix] path auto-selects the smallest of the six
// small square sizes (10x10 through 20x20). The extended [EncodeText]/[DecodeGrid]
// path additionally supports every larger square (22x22 through 132x132, with
// multi-region data mapping and interleaved Reed-Solomon blocks) and all six
// rectangular sizes (8x18, 8x32, 12x26, 12x36, 16x36 and 16x48).
//
// # Extended encoder ([EncodeText])
//
// [EncodeText] and [EncodeTextSymbol] accept [EncodeOptions] that select the
// encodation [Scheme], the [SizePreference] (square, rectangular or automatic),
// an Extended Channel Interpretation (ECI) number, a GS1 FNC1 indicator and
// structured-append parameters. With SchemeAuto the encoder switches among the
// ASCII, C40, Text, X12, EDIFACT and Base 256 schemes using the ISO/IEC 16022
// Annex P look-ahead cost model to keep the codeword count small, emitting the
// correct latch and unlatch codewords at every scheme boundary.
//
// # Extended decoder ([DecodeGrid], [DecodeText], [DecodeAll])
//
// [DecodeGrid] decodes a module grid of any supported size; [DecodeText] locates
// and decodes a single symbol in an image; and [DecodeAll] segments an image by
// a recursive X-Y cut and decodes every symbol it contains. All three return a
// [DecodedResult] carrying the content bytes plus any ECI, GS1 and
// structured-append metadata, and repair errors per interleaved Reed-Solomon
// block before decoding.
//
// # Determinism
//
// Encoding is fully deterministic: the same input always yields byte-identical
// codewords, module placement and bitmap. Padding uses the fixed ECC200
// randomisation, and no source of nondeterminism (maps, time, randomness) is
// involved. Decoding is likewise deterministic.
//
// # DEFERRED
//
// This package implements a broad, genuinely-working subset of ECC200. The
// following remain NOT implemented:
//
//   - The single 144x144 symbol, whose Reed-Solomon block structure is a special
//     case, is not in the size table (every other standard size, 10x10 through
//     132x132 and the six rectangular sizes, is supported).
//   - Detection ([DetectAndDecode], [DecodeText], [DecodeAll]) is tolerant of an
//     integer scale factor and a white quiet zone and expects axis-aligned,
//     dark-on-light symbols separated by white space. Full perspective or
//     rotation correction and illumination normalisation are not implemented.
//   - Reader programming and the Macro 05/06 codewords are not implemented; GS1
//     FNC1 is handled only in ASCII-mode content (group separators map to FNC1).
//   - The legacy [Encode]/[DecodeMatrix] path still handles ASCII encodation and
//     the six small square sizes only; use [EncodeText]/[DecodeGrid] for the
//     other schemes and sizes.
package datamatrix
