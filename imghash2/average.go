package imghash2

import cv "github.com/malcolmston/opencv"

// AverageHash implements the average hash (aHash) of Krawetz. The image is
// reduced to grayscale, scaled to 8×8 and each of the 64 pixels becomes one
// bit, set when the pixel is brighter than the mean of the block. The result is
// a 64-bit (8-byte) [Hash] compared by Hamming distance.
//
// aHash is the simplest perceptual hash. It is robust to scaling, aspect and —
// because the threshold is the block mean — to uniform brightness changes, but
// it is more sensitive to gamma and contrast changes than [PHash]. The zero
// value is ready to use; [NewAverageHash] is provided for symmetry.
type AverageHash struct{}

// NewAverageHash returns a ready-to-use [AverageHash].
func NewAverageHash() AverageHash { return AverageHash{} }

// Name returns the identifier "ahash".
func (AverageHash) Name() string { return "ahash" }

// Bits returns 64, the fixed length of an average hash.
func (AverageHash) Bits() int { return 64 }

// Compute returns the 8-byte average hash of img.
func (AverageHash) Compute(img *cv.Mat) Hash {
	requireImage(img, "AverageHash.Compute")
	vals := grayFloats(img, 8, 8)
	thr := Mean(vals)
	out := make([]bool, 64)
	for i, v := range vals {
		out[i] = v > thr
	}
	return packBits(out)
}

// Average is a convenience wrapper returning the [AverageHash] of img.
func Average(img *cv.Mat) Hash { return AverageHash{}.Compute(img) }
