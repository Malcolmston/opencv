package imghash2

import cv "github.com/malcolmston/opencv"

// DifferenceHash implements the difference (gradient) hash of Krawetz. The
// image is reduced to grayscale and scaled so each bit can record the sign of a
// gradient between two adjacent pixels: for the default horizontal hash the
// image is scaled to 9×8 and each bit is set when the right pixel is brighter
// than its left neighbour; when Vertical is true the image is scaled to 8×9 and
// each bit compares a pixel with the one below it. Either way eight comparisons
// over eight rows (or columns) give a 64-bit (8-byte) [Hash] compared by
// Hamming distance.
//
// Because it encodes relative brightness between neighbours rather than
// absolute values, dHash is naturally invariant to uniform brightness and
// contrast changes and is a strong, cheap complement to [AverageHash] and
// [PHash]. The zero value is the horizontal hash; [NewDifferenceHash] and
// [NewVerticalDifferenceHash] construct the two orientations explicitly.
type DifferenceHash struct {
	// Vertical selects the vertical gradient (each pixel compared with the one
	// below it) instead of the default horizontal gradient.
	Vertical bool
}

// NewDifferenceHash returns a horizontal [DifferenceHash].
func NewDifferenceHash() DifferenceHash { return DifferenceHash{} }

// NewVerticalDifferenceHash returns a vertical [DifferenceHash].
func NewVerticalDifferenceHash() DifferenceHash { return DifferenceHash{Vertical: true} }

// Name returns "dhash" for the horizontal hash and "dhash-v" for the vertical
// hash.
func (h DifferenceHash) Name() string {
	if h.Vertical {
		return "dhash-v"
	}
	return "dhash"
}

// Bits returns 64, the fixed length of a difference hash.
func (DifferenceHash) Bits() int { return 64 }

// Compute returns the 8-byte difference hash of img.
func (h DifferenceHash) Compute(img *cv.Mat) Hash {
	requireImage(img, "DifferenceHash.Compute")
	out := make([]bool, 64)
	if h.Vertical {
		// Height one greater than width so each column yields eight comparisons.
		small := GrayResize(img, 8, 9)
		i := 0
		for y := 0; y < 8; y++ {
			for x := 0; x < 8; x++ {
				top := small.Data[y*8+x]
				bottom := small.Data[(y+1)*8+x]
				out[i] = bottom > top
				i++
			}
		}
	} else {
		// Width one greater than height so each row yields eight comparisons.
		small := GrayResize(img, 9, 8)
		i := 0
		for y := 0; y < 8; y++ {
			for x := 0; x < 8; x++ {
				left := small.Data[y*9+x]
				right := small.Data[y*9+x+1]
				out[i] = right > left
				i++
			}
		}
	}
	return packBits(out)
}

// Difference is a convenience wrapper returning the horizontal [DifferenceHash]
// of img.
func Difference(img *cv.Mat) Hash { return DifferenceHash{}.Compute(img) }
