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
// The following square sizes are supported, and [Encode] auto-selects the
// smallest that fits:
//
//	Size   Data CW   ECC CW   ASCII chars   Digit chars
//	10x10        3        5          ~2            ~6
//	12x12        5        7          ~4           ~10
//	14x14        8       10          ~7           ~16
//	16x16       12       12         ~11           ~24
//	18x18       18       14         ~17           ~36
//	20x20       22       18         ~21           ~44
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
// This package intentionally implements a focused, genuinely-working subset.
// The following are NOT implemented:
//
//   - Only ASCII encodation. The C40, Text, X12, EDIFACT and Base256 modes are
//     not implemented; inputs must be 7-bit ASCII (bytes 0-127). A symbol that
//     selects one of these modes is reported via an unsupported-mode error.
//   - Only the six square sizes listed above. Larger square symbols (which use
//     multiple data regions and interleaved Reed-Solomon blocks) and all
//     rectangular symbols are not supported.
//   - Detection is tolerant of an integer scale factor and a white quiet zone,
//     and expects an axis-aligned, dark-on-light symbol. Full perspective or
//     rotation correction, illumination normalisation and multi-symbol
//     detection are not implemented.
//   - Extended Channel Interpretation (ECI), structured append, reader
//     programming and macro codewords are not implemented.
package datamatrix
