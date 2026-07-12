package imghash

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// defaultAverageHashN is the grid resolution used by the zero value of
// [AverageHashN]: an 8×8 grid, matching [AverageHash].
const defaultAverageHashN = 8

// AverageHashN generalises [AverageHash] to an arbitrary n×n grid. The image is
// reduced to grayscale, scaled to n×n and each of the n² cells becomes one bit,
// set when it exceeds the mean of the grid. The result is an n²-bit fingerprint
// packed into ceil(n²/8) bytes and compared by Hamming distance.
//
// A larger n keeps more spatial detail — and so discriminates better between
// superficially similar images — at the cost of a longer fingerprint and a
// little less robustness to blur. Choose an n whose square is a multiple of 8
// (for example 8, 16 or 24) so the bits fill whole bytes.
//
// The zero value behaves as an 8×8 [AverageHash]; [NewAverageHashN] sets the
// grid explicitly.
type AverageHashN struct {
	n int
}

// NewAverageHashN returns an [AverageHashN] using an n×n grid. It panics if n is
// not positive.
func NewAverageHashN(n int) AverageHashN {
	if n <= 0 {
		panic(fmt.Sprintf("imghash: NewAverageHashN requires positive n, got %d", n))
	}
	return AverageHashN{n: n}
}

// grid returns the configured resolution, defaulting the zero value to 8.
func (h AverageHashN) grid() int {
	if h.n <= 0 {
		return defaultAverageHashN
	}
	return h.n
}

// Compute returns the n²-bit average hash of img.
func (h AverageHashN) Compute(img *cv.Mat) []byte {
	requireImage(img, "AverageHashN.Compute")
	n := h.grid()
	small := grayResize(img, n, n)
	vals := make([]float64, n*n)
	for i := range vals {
		vals[i] = float64(small.Data[i])
	}
	thr := mean(vals)
	bitsOut := make([]bool, n*n)
	for i := range vals {
		bitsOut[i] = vals[i] > thr
	}
	return packBits(bitsOut)
}

// Compare returns the Hamming distance between two average hashes.
func (AverageHashN) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "AverageHashN.Compare")
	return float64(hamming(a, b))
}

// AverageN is a convenience wrapper returning the n×n [AverageHashN] of img.
func AverageN(img *cv.Mat, n int) []byte { return NewAverageHashN(n).Compute(img) }
