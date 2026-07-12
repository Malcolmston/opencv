package imghash

import cv "github.com/malcolmston/opencv"

// AverageHash implements the average hash (aHash) of Krawetz. The image is
// reduced to grayscale, scaled to 8×8 and each of the 64 pixels is turned into
// one bit: set when the pixel is brighter than the mean of the block. The
// result is a 64-bit (8-byte) fingerprint compared by Hamming distance.
//
// aHash is the simplest perceptual hash. It is robust to scaling, aspect and —
// because the threshold is the block mean — to uniform brightness changes, but
// it is more sensitive to gamma and contrast changes than [PHash].
//
// The zero value is ready to use; [NewAverageHash] is provided for symmetry.
type AverageHash struct{}

// NewAverageHash returns a ready-to-use [AverageHash].
func NewAverageHash() AverageHash { return AverageHash{} }

// Compute returns the 8-byte average hash of img.
func (AverageHash) Compute(img *cv.Mat) []byte {
	requireImage(img, "AverageHash.Compute")
	small := grayResize(img, 8, 8)
	vals := make([]float64, 64)
	for i := 0; i < 64; i++ {
		vals[i] = float64(small.Data[i])
	}
	thr := mean(vals)
	bitsOut := make([]bool, 64)
	for i := 0; i < 64; i++ {
		bitsOut[i] = vals[i] > thr
	}
	return packBits(bitsOut)
}

// Compare returns the Hamming distance between two average hashes.
func (AverageHash) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "AverageHash.Compare")
	return float64(hamming(a, b))
}

// Average is a convenience wrapper returning the [AverageHash] of img.
func Average(img *cv.Mat) []byte { return AverageHash{}.Compute(img) }
