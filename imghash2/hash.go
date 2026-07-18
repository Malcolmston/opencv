package imghash2

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"math/bits"

	cv "github.com/malcolmston/opencv"
)

// Hash is a binary perceptual fingerprint: a fixed-length bit string packed
// most-significant-bit first into bytes. Bit i lives in byte i/8 at position
// 7-(i%8). It is the value returned by every binary hasher in this package and
// is compared by Hamming distance, where a smaller distance means the two
// source images are more visually similar.
type Hash []byte

// Hasher is the common interface implemented by every binary perceptual hash in
// this package. It reduces a [cv.Mat] image to a fixed-length [Hash].
type Hasher interface {
	// Compute returns the perceptual [Hash] of img as a fresh value.
	Compute(img *cv.Mat) Hash
	// Bits returns the fixed number of bits the hasher produces.
	Bits() int
	// Name returns a short stable identifier for the hasher, such as "phash".
	Name() string
}

// FloatHasher is the interface implemented by the real-valued descriptor hashes
// ([RadialVarianceHash] and [ColorMomentHash]). Its fingerprints are compared
// by [L1Distance] or [L2Distance] rather than Hamming distance.
type FloatHasher interface {
	// ComputeFloat returns the descriptor of img as a fresh [FloatHash].
	ComputeFloat(img *cv.Mat) FloatHash
	// Name returns a short stable identifier for the hasher.
	Name() string
}

// Bytes returns the underlying packed bytes of the hash. The result aliases the
// hash's storage; use [Hash.Clone] first if an independent copy is required.
func (h Hash) Bytes() []byte { return []byte(h) }

// Clone returns a fresh copy of the hash that shares no storage with the
// original.
func (h Hash) Clone() Hash {
	out := make(Hash, len(h))
	copy(out, h)
	return out
}

// Bits returns the number of bits stored in the hash, which is eight times its
// byte length.
func (h Hash) Bits() int { return len(h) * 8 }

// Bit reports whether bit i of the hash is set, counting most-significant-bit
// first to match the packing used by the hashers. It panics if i is out of
// range.
func (h Hash) Bit(i int) bool {
	if i < 0 || i >= h.Bits() {
		panic(fmt.Sprintf("imghash2: Hash.Bit index %d out of range for %d bits", i, h.Bits()))
	}
	return h[i/8]&(1<<uint(7-(i%8))) != 0
}

// OnesCount returns the number of set bits in the hash (its Hamming weight).
func (h Hash) OnesCount() int {
	n := 0
	for _, b := range h {
		n += bits.OnesCount8(b)
	}
	return n
}

// Equal reports whether two hashes have the same length and identical bits.
func (h Hash) Equal(other Hash) bool {
	if len(h) != len(other) {
		return false
	}
	for i := range h {
		if h[i] != other[i] {
			return false
		}
	}
	return true
}

// Hamming returns the Hamming distance — the number of differing bits — between
// this hash and another of the same length. It panics on a length mismatch,
// which indicates the two fingerprints came from different hashers.
func (h Hash) Hamming(other Hash) int {
	requireSameLen(h, other, "Hash.Hamming")
	d := 0
	for i := range h {
		d += bits.OnesCount8(h[i] ^ other[i])
	}
	return d
}

// NormalizedHamming returns the Hamming distance divided by the bit count, a
// value in [0, 1] where 0 means identical and 1 means every bit differs.
// Normalising lets a single threshold apply across hashes of different sizes.
// It panics if the hashes differ in length or are empty.
func (h Hash) NormalizedHamming(other Hash) float64 {
	requireSameLen(h, other, "Hash.NormalizedHamming")
	if len(h) == 0 {
		panic("imghash2: NormalizedHamming requires non-empty hashes")
	}
	return float64(h.Hamming(other)) / float64(h.Bits())
}

// Similarity returns 1 − [Hash.NormalizedHamming], a score in [0, 1] that is 1
// for identical fingerprints and falls toward 0 as bits diverge. It panics if
// the hashes differ in length or are empty.
func (h Hash) Similarity(other Hash) float64 {
	return 1 - h.NormalizedHamming(other)
}

// String returns the lower-case hexadecimal text form of the hash, the usual
// way to store or transmit a fingerprint. It is the inverse of [ParseHash].
func (h Hash) String() string { return hex.EncodeToString(h) }

// ParseHash decodes the hexadecimal text produced by [Hash.String] back into a
// [Hash]. It returns an error if s is not valid hexadecimal or has an odd
// length, so a corrupt or truncated fingerprint is rejected rather than
// silently misread.
func ParseHash(s string) (Hash, error) {
	b, err := hex.DecodeString(s)
	if err != nil {
		return nil, err
	}
	return Hash(b), nil
}

// FloatHash is a real-valued perceptual descriptor: a fixed-length vector of
// float64 features returned by the descriptor hashes. It is compared by
// [FloatHash.L1] or [FloatHash.L2] distance.
type FloatHash []float64

// Dims returns the number of features in the descriptor.
func (f FloatHash) Dims() int { return len(f) }

// Clone returns a fresh copy of the descriptor that shares no storage with the
// original.
func (f FloatHash) Clone() FloatHash {
	out := make(FloatHash, len(f))
	copy(out, f)
	return out
}

// L1 returns the L1 (Manhattan) distance between this descriptor and another of
// the same length: the sum of absolute per-feature differences. It panics on a
// length mismatch.
func (f FloatHash) L1(other FloatHash) float64 {
	requireSameFloatLen(f, other, "FloatHash.L1")
	var sum float64
	for i := range f {
		sum += math.Abs(f[i] - other[i])
	}
	return sum
}

// L2 returns the L2 (Euclidean) distance between this descriptor and another of
// the same length. It panics on a length mismatch.
func (f FloatHash) L2(other FloatHash) float64 {
	requireSameFloatLen(f, other, "FloatHash.L2")
	var sum float64
	for i := range f {
		d := f[i] - other[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// Bytes serialises the descriptor as consecutive big-endian IEEE-754 float64
// words, a portable storage form recoverable with [ParseFloatHash].
func (f FloatHash) Bytes() []byte {
	out := make([]byte, len(f)*8)
	for i, v := range f {
		binary.BigEndian.PutUint64(out[i*8:], math.Float64bits(v))
	}
	return out
}

// ParseFloatHash is the inverse of [FloatHash.Bytes]: it decodes big-endian
// float64 words back into a [FloatHash]. It panics if the input length is not a
// multiple of eight.
func ParseFloatHash(b []byte) FloatHash {
	if len(b)%8 != 0 {
		panic(fmt.Sprintf("imghash2: ParseFloatHash needs a length multiple of 8, got %d", len(b)))
	}
	out := make(FloatHash, len(b)/8)
	for i := range out {
		out[i] = math.Float64frombits(binary.BigEndian.Uint64(b[i*8:]))
	}
	return out
}

// requireImage panics if img is nil or empty, matching the fail-fast behaviour
// of the parent package's pixel helpers.
func requireImage(img *cv.Mat, name string) {
	if img == nil || img.Empty() {
		panic(fmt.Sprintf("imghash2: %s requires a non-empty image", name))
	}
}

// requireSameLen panics if the two binary hashes differ in length.
func requireSameLen(a, b Hash, name string) {
	if len(a) != len(b) {
		panic(fmt.Sprintf("imghash2: %s requires equal-length hashes, got %d and %d", name, len(a), len(b)))
	}
}

// requireSameFloatLen panics if the two descriptors differ in length.
func requireSameFloatLen(a, b FloatHash, name string) {
	if len(a) != len(b) {
		panic(fmt.Sprintf("imghash2: %s requires equal-length descriptors, got %d and %d", name, len(a), len(b)))
	}
}

// packBits packs bits most-significant-bit first into a fresh Hash. The output
// length is ceil(len(in)/8).
func packBits(in []bool) Hash {
	out := make(Hash, (len(in)+7)/8)
	for i, b := range in {
		if b {
			out[i/8] |= 1 << uint(7-(i%8))
		}
	}
	return out
}
