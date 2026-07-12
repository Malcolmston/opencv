package imghash

import (
	"encoding/hex"
	"math/bits"
)

// HexEncode returns the lower-case hexadecimal text form of a hash, the usual
// way to store or transmit a fingerprint. Each byte becomes two hex digits, so
// an 8-byte hash yields a 16-character string. It is the inverse of [HexDecode].
func HexEncode(h []byte) string {
	return hex.EncodeToString(h)
}

// HexDecode parses the hexadecimal text produced by [HexEncode] back into hash
// bytes. It returns an error if s is not valid hexadecimal or has an odd length,
// so that a corrupt or truncated fingerprint is rejected rather than silently
// misread.
func HexDecode(s string) ([]byte, error) {
	return hex.DecodeString(s)
}

// HammingNormalized returns the Hamming distance between two equal-length binary
// hashes divided by their bit count, a value in [0, 1] where 0 means identical
// and 1 means every bit differs. Normalising by length lets a single threshold
// apply across hashes of different sizes. It panics if the hashes differ in
// length or are empty.
func HammingNormalized(a, b []byte) float64 {
	requireSameLen(a, b, "HammingNormalized")
	if len(a) == 0 {
		panic("imghash: HammingNormalized requires non-empty hashes")
	}
	var d int
	for i := range a {
		d += bits.OnesCount8(a[i] ^ b[i])
	}
	return float64(d) / float64(len(a)*8)
}

// Similarity returns a normalised similarity score in [0, 1] between two binary
// hashes: 1 for identical fingerprints, falling to 0 as bits diverge. It is
// simply 1 − [HammingNormalized] and is a convenient monotone alternative to a
// raw distance when a "percentage alike" reading is wanted. It panics if the
// hashes differ in length or are empty.
func Similarity(a, b []byte) float64 {
	return 1 - HammingNormalized(a, b)
}

// IsDuplicate reports whether two binary hashes are near-duplicates under a
// normalised-distance threshold: it returns true when [HammingNormalized] of a
// and b is at most maxNormalizedDistance. A threshold around 0.1–0.15 (that is,
// up to roughly 10–15% of the bits differing) is a common near-duplicate cutoff
// for the 64-bit hashes, though the right value depends on the corpus and should
// be tuned. It panics if the hashes differ in length or are empty.
func IsDuplicate(a, b []byte, maxNormalizedDistance float64) bool {
	return HammingNormalized(a, b) <= maxNormalizedDistance
}
