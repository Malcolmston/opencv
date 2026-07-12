package imghash

import cv "github.com/malcolmston/opencv"

// MedianHash implements the median hash, a close relative of [AverageHash]. The
// image is reduced to grayscale, scaled to 8×8 and each of the 64 pixels becomes
// one bit, set when the pixel exceeds the median of the block rather than its
// mean. The result is a 64-bit (8-byte) fingerprint compared by Hamming distance.
//
// Thresholding at the median guarantees a balanced fingerprint — as close as
// possible to 32 bits set and 32 clear — which spreads distances more evenly
// than the mean when the intensity distribution is skewed (for example a mostly
// dark image with a bright highlight). It is otherwise as cheap and as robust to
// scale and uniform brightness as [AverageHash].
//
// The zero value is ready to use; [NewMedianHash] is provided for symmetry.
type MedianHash struct{}

// NewMedianHash returns a ready-to-use [MedianHash].
func NewMedianHash() MedianHash { return MedianHash{} }

// Compute returns the 8-byte median hash of img.
func (MedianHash) Compute(img *cv.Mat) []byte {
	requireImage(img, "MedianHash.Compute")
	small := grayResize(img, 8, 8)
	vals := make([]float64, 64)
	for i := 0; i < 64; i++ {
		vals[i] = float64(small.Data[i])
	}
	thr := median(vals)
	bitsOut := make([]bool, 64)
	for i := 0; i < 64; i++ {
		bitsOut[i] = vals[i] > thr
	}
	return packBits(bitsOut)
}

// Compare returns the Hamming distance between two median hashes.
func (MedianHash) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "MedianHash.Compare")
	return float64(hamming(a, b))
}

// Median is a convenience wrapper returning the [MedianHash] of img.
func Median(img *cv.Mat) []byte { return MedianHash{}.Compute(img) }
