// Package barcode provides 1D and 2D barcode generation and detection on top of
// the standard-library-only OpenCV port github.com/malcolmston/opencv (imported
// here as cv). It mirrors the spirit of OpenCV's barcode and objdetect QR
// modules while depending only on that root package and the Go standard library.
//
// # QR Codes
//
// [QREncode] renders text as a QR Code symbol (a grayscale [cv.Mat]) and
// [QRDetectAndDecode] locates and decodes such a symbol back into text. The two
// form a matched, standards-faithful pair for a bounded but genuinely
// specification-compliant subset of ISO/IEC 18004:
//
//   - Versions 1-4 (21x21 to 33x33 modules).
//   - Byte (8-bit) encoding mode.
//   - Error-correction level L (a single Reed-Solomon block per symbol).
//
// Within that scope the symbols are real QR codes: three finder patterns with
// separators, the horizontal and vertical timing patterns, the version-2+
// alignment pattern, BCH(15,5) error-protected format information, a
// Reed-Solomon error-correction block over GF(256), the zigzag data layout and
// automatic data-mask selection all follow the standard. As a result the output
// is readable by conformant third-party QR scanners, and — the property this
// package tests — it round-trips through [QRDetectAndDecode].
//
// Detection is a self-contained pipeline: the image is reduced to bi-level
// modules with cv.CvtColor and cv.Threshold (Otsu), rows are scanned for the
// finder pattern's 1:1:3:1:1 run-length signature, candidates are confirmed with
// a vertical cross-check and clustered into the three finder centres, the
// symbol's orientation and version are recovered from their geometry, and every
// module is sampled on a finder-derived affine grid. Because sampling uses an
// affine basis, detection tolerates translation, scaling, rotation and shear;
// strong perspective distortion is not corrected. Reed-Solomon decoding means a
// handful of misread or damaged modules are corrected rather than fatal.
//
// # 1D barcodes
//
// [EncodeEAN13] / [DecodeEAN13] handle the 13-digit EAN-13 retail symbology,
// including the first-digit parity encoding and the modulo-10 check digit.
// [EncodeCode128] / [DecodeCode128] handle Code 128 code set B (printable ASCII
// 32-126) with its modulo-103 checksum. Each decoder reads a single horizontal
// scanline, locates the symbol by its dark extent, samples the modules and
// verifies the check digit, so a rendered symbol decodes back to its input.
//
// # Reed-Solomon
//
// [ReedSolomonEncode] and [ReedSolomonDecode] expose the package's GF(256)
// Reed-Solomon codec (primitive polynomial 0x11D, generator 2), which underlies
// the QR error correction and can be used directly. The decoder uses syndrome
// computation, Berlekamp-Massey, a Chien search and the Forney algorithm to
// correct up to nsym/2 byte errors.
//
// # Conventions and determinism
//
// Coordinates follow the cv convention: x is the column and y is the row with
// the origin at the top-left. Symbols are rendered as single-channel grayscale
// Mats with dark modules at 0 and light at 255. Every function in this package
// is deterministic — mask selection, sampling and decoding perform no
// randomised or concurrent work — so the same input always yields byte-identical
// output.
//
// # Deferred
//
// Larger QR versions (5+), the numeric, alphanumeric and Kanji QR modes, ECC
// levels M/Q/H with multi-block interleaving, QR perspective rectification, and
// other 2D symbologies (Data Matrix, Aztec, PDF417) are not implemented. Code
// 128 code sets A and C (and the auto-switch shift codes) are likewise deferred;
// only code set B is supported.
package barcode
