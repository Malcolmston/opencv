package imghash2

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// defaultPHashSize is the side length of the working image fed to the DCT by a
// zero-value [PHash].
const defaultPHashSize = 32

// PHash implements the perceptual hash (pHash) of Zauner. The image is reduced
// to grayscale and scaled to Size×Size, a 2-D discrete cosine transform is
// applied, and the top-left 8×8 block of low-frequency coefficients is kept.
// Each of the 64 coefficients becomes one bit, set when it exceeds the median
// of the block. The result is a 64-bit (8-byte) [Hash] compared by Hamming
// distance.
//
// Working in the frequency domain and thresholding at the median makes pHash
// markedly more robust than [AverageHash] to gamma, brightness and mild blur:
// the median threshold is invariant to any monotonic change that preserves the
// coefficient ordering, and the low-frequency coefficients ignore fine detail
// and noise. The DC coefficient at index 0 is retained, matching the original
// pHash. The zero value uses a 32×32 working image; [NewPHash] returns one
// explicitly.
type PHash struct {
	// Size is the side length of the square working image the DCT runs on. It
	// must be at least 8 (the low-frequency block width); a zero value means
	// the 32×32 default.
	Size int
}

// NewPHash returns a [PHash] using the default 32×32 working image.
func NewPHash() PHash { return PHash{Size: defaultPHashSize} }

// NewPHashSize returns a [PHash] whose DCT runs on a size×size working image.
// It panics if size is less than 8.
func NewPHashSize(size int) PHash {
	if size < 8 {
		panic(fmt.Sprintf("imghash2: NewPHashSize requires size >= 8, got %d", size))
	}
	return PHash{Size: size}
}

// Name returns the identifier "phash".
func (PHash) Name() string { return "phash" }

// Bits returns 64, the fixed length of a perceptual hash.
func (PHash) Bits() int { return 64 }

// size returns the effective working-image side length, applying the default.
func (h PHash) size() int {
	if h.Size <= 0 {
		return defaultPHashSize
	}
	if h.Size < 8 {
		panic(fmt.Sprintf("imghash2: PHash.Size must be >= 8, got %d", h.Size))
	}
	return h.Size
}

// LowFrequencies returns the top-left 8×8 block of low-frequency DCT
// coefficients (row-major, 64 values) that [PHash.Compute] thresholds.
// Coefficient 0 is the DC term. It is exposed so callers can inspect or
// re-threshold the frequency content directly.
func (h PHash) LowFrequencies(img *cv.Mat) []float64 {
	requireImage(img, "PHash.LowFrequencies")
	n := h.size()
	coeffs := DCT2D(grayFloats(img, n, n), n)
	low := make([]float64, 64)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			low[y*8+x] = coeffs[y*n+x]
		}
	}
	return low
}

// Compute returns the 8-byte perceptual hash of img.
func (h PHash) Compute(img *cv.Mat) Hash {
	low := h.LowFrequencies(img)
	thr := Median(low)
	out := make([]bool, 64)
	for i, v := range low {
		out[i] = v > thr
	}
	return packBits(out)
}

// Perceptual is a convenience wrapper returning the default [PHash] of img.
func Perceptual(img *cv.Mat) Hash { return NewPHash().Compute(img) }
