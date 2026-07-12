package imghash

import cv "github.com/malcolmston/opencv"

// pHashSize is the side length of the working image fed to the DCT.
const pHashSize = 32

// PHash implements the perceptual hash (pHash) of Zauner. The image is reduced
// to grayscale and scaled to 32×32, a 2-D discrete cosine transform is applied,
// and the top-left 8×8 block of low-frequency coefficients is kept. Each of the
// 64 coefficients becomes one bit: set when it exceeds the median of the block.
// The result is a 64-bit (8-byte) fingerprint compared by Hamming distance.
//
// Working in the frequency domain and thresholding at the median makes pHash
// markedly more robust than [AverageHash] to gamma, brightness and mild blur:
// the median threshold is invariant to any monotonic change that preserves the
// coefficient ordering, and the low-frequency coefficients ignore fine detail
// and noise. The DC coefficient (0,0) is retained, matching the original pHash.
//
// The zero value is ready to use; [NewPHash] is provided for symmetry.
type PHash struct{}

// NewPHash returns a ready-to-use [PHash].
func NewPHash() PHash { return PHash{} }

// Compute returns the 8-byte perceptual hash of img.
func (PHash) Compute(img *cv.Mat) []byte {
	low := pHashLowFreq(img)
	thr := median(low)
	bitsOut := make([]bool, 64)
	for i := 0; i < 64; i++ {
		bitsOut[i] = low[i] > thr
	}
	return packBits(bitsOut)
}

// pHashLowFreq returns the top-left 8×8 block of low-frequency DCT coefficients
// (row-major, 64 values) that the perceptual hash thresholds. Coefficient 0 is
// the DC term. It is the shared front end of [PHash.Compute] and is exercised
// directly by the tests.
func pHashLowFreq(img *cv.Mat) []float64 {
	requireImage(img, "PHash.Compute")
	small := grayResize(img, pHashSize, pHashSize)

	in := make([]float64, pHashSize*pHashSize)
	for i := range in {
		in[i] = float64(small.Data[i])
	}
	coeffs := dct2D(in, pHashSize)

	low := make([]float64, 64)
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			low[y*8+x] = coeffs[y*pHashSize+x]
		}
	}
	return low
}

// Compare returns the Hamming distance between two perceptual hashes.
func (PHash) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "PHash.Compare")
	return float64(hamming(a, b))
}

// Perceptual is a convenience wrapper returning the [PHash] of img.
func Perceptual(img *cv.Mat) []byte { return PHash{}.Compute(img) }
