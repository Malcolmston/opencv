package imghash

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// defaultPHashN is the low-frequency block side used by the zero value of
// [PHashN]: an 8×8 block, matching [PHash].
const defaultPHashN = 8

// PHashN generalises [PHash] to a configurable low-frequency block size. The
// image is reduced to grayscale and scaled to 32×32, a 2-D discrete cosine
// transform is applied, and the top-left k×k block of low-frequency coefficients
// is kept. Each coefficient becomes one bit, set when it exceeds the median of
// the block. The result is a k²-bit fingerprint packed into ceil(k²/8) bytes and
// compared by Hamming distance.
//
// A larger k keeps more mid-frequency detail, discriminating finer differences
// at the cost of a little robustness to blur and noise; k=8 reproduces the
// classic 64-bit [PHash]. Choose a k whose square is a multiple of 8 (for
// example 8, 12 or 16) so the bits fill whole bytes. k must not exceed the 32×32
// working size.
//
// The zero value behaves as the 8×8 [PHash]; [NewPHashN] sets k explicitly.
type PHashN struct {
	k int
}

// NewPHashN returns a [PHashN] keeping the top-left k×k DCT block. It panics if
// k is not positive or exceeds the 32-pixel working size.
func NewPHashN(k int) PHashN {
	if k <= 0 || k > pHashSize {
		panic(fmt.Sprintf("imghash: NewPHashN requires 0 < k <= %d, got %d", pHashSize, k))
	}
	return PHashN{k: k}
}

func (h PHashN) block() int {
	if h.k <= 0 {
		return defaultPHashN
	}
	return h.k
}

// Compute returns the k²-bit perceptual hash of img.
func (h PHashN) Compute(img *cv.Mat) []byte {
	requireImage(img, "PHashN.Compute")
	k := h.block()
	small := grayResize(img, pHashSize, pHashSize)

	in := make([]float64, pHashSize*pHashSize)
	for i := range in {
		in[i] = float64(small.Data[i])
	}
	coeffs := dct2D(in, pHashSize)

	low := make([]float64, k*k)
	for y := 0; y < k; y++ {
		for x := 0; x < k; x++ {
			low[y*k+x] = coeffs[y*pHashSize+x]
		}
	}
	thr := median(low)
	bitsOut := make([]bool, k*k)
	for i := range low {
		bitsOut[i] = low[i] > thr
	}
	return packBits(bitsOut)
}

// Compare returns the Hamming distance between two perceptual hashes.
func (PHashN) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "PHashN.Compare")
	return float64(hamming(a, b))
}

// PerceptualN is a convenience wrapper returning the [PHashN] of img keeping the
// top-left k×k DCT block.
func PerceptualN(img *cv.Mat, k int) []byte { return NewPHashN(k).Compute(img) }
