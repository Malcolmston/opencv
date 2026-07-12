package imghash

import cv "github.com/malcolmston/opencv"

// DHash implements the difference (gradient) hash of Krawetz. The image is
// reduced to grayscale and scaled to 9×8, and each bit records the sign of the
// horizontal gradient between two adjacent pixels: set when the right pixel is
// brighter than its left neighbour. The eight comparisons per row over eight
// rows give a 64-bit (8-byte) fingerprint compared by Hamming distance.
//
// Because it encodes relative brightness between neighbours rather than
// absolute values, dHash is naturally invariant to uniform brightness and
// contrast changes and is a strong, cheap complement to [AverageHash] and
// [PHash].
//
// The zero value is ready to use; [NewDHash] is provided for symmetry.
type DHash struct{}

// NewDHash returns a ready-to-use [DHash].
func NewDHash() DHash { return DHash{} }

// Compute returns the 8-byte difference hash of img.
func (DHash) Compute(img *cv.Mat) []byte {
	requireImage(img, "DHash.Compute")
	// Width one greater than height so each row yields eight comparisons.
	small := grayResize(img, 9, 8)
	bitsOut := make([]bool, 64)
	i := 0
	for y := 0; y < 8; y++ {
		for x := 0; x < 8; x++ {
			left := small.Data[y*9+x]
			right := small.Data[y*9+x+1]
			bitsOut[i] = right > left
			i++
		}
	}
	return packBits(bitsOut)
}

// Compare returns the Hamming distance between two difference hashes.
func (DHash) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "DHash.Compare")
	return float64(hamming(a, b))
}

// Difference is a convenience wrapper returning the [DHash] of img.
func Difference(img *cv.Mat) []byte { return DHash{}.Compute(img) }
