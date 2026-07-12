package imghash

import cv "github.com/malcolmston/opencv"

// DHashVertical implements the vertical difference (gradient) hash, the
// transpose of [DHash]. The image is reduced to grayscale and scaled to 8×9
// (one more row than columns), and each bit records the sign of the vertical
// gradient between two vertically adjacent pixels: set when the lower pixel is
// brighter than the one above it. The eight comparisons per column over eight
// columns give a 64-bit (8-byte) fingerprint compared by Hamming distance.
//
// Where [DHash] keys on horizontal structure, DHashVertical keys on horizontal
// edges and vertical gradients; the two are complementary and are combined by
// [DHashCombined]. Like [DHash] it encodes relative rather than absolute
// brightness, so it is naturally invariant to uniform brightness and contrast
// changes.
//
// The zero value is ready to use; [NewDHashVertical] is provided for symmetry.
type DHashVertical struct{}

// NewDHashVertical returns a ready-to-use [DHashVertical].
func NewDHashVertical() DHashVertical { return DHashVertical{} }

// Compute returns the 8-byte vertical difference hash of img.
func (DHashVertical) Compute(img *cv.Mat) []byte {
	requireImage(img, "DHashVertical.Compute")
	// Height one greater than width so each column yields eight comparisons.
	small := grayResize(img, 8, 9)
	return packBits(verticalBits(small))
}

// verticalBits returns the 64 vertical-gradient bits of an 8-wide, 9-tall gray
// image, column by column.
func verticalBits(small *cv.Mat) []bool {
	bitsOut := make([]bool, 64)
	i := 0
	for x := 0; x < 8; x++ {
		for y := 0; y < 8; y++ {
			top := small.Data[y*8+x]
			bottom := small.Data[(y+1)*8+x]
			bitsOut[i] = bottom > top
			i++
		}
	}
	return bitsOut
}

// horizontalBits returns the 64 horizontal-gradient bits of a 9-wide, 8-tall
// gray image, row by row (the same encoding as [DHash]).
func horizontalBits(small *cv.Mat) []bool {
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
	return bitsOut
}

// Compare returns the Hamming distance between two vertical difference hashes.
func (DHashVertical) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "DHashVertical.Compare")
	return float64(hamming(a, b))
}

// DifferenceVertical is a convenience wrapper returning the [DHashVertical] of
// img.
func DifferenceVertical(img *cv.Mat) []byte { return DHashVertical{}.Compute(img) }

// DHashCombined concatenates the horizontal difference hash of [DHash] with the
// vertical difference hash of [DHashVertical], producing a 128-bit (16-byte)
// fingerprint compared by Hamming distance. Encoding gradients along both axes
// makes the combined hash markedly more discriminative than either half alone —
// two images that happen to share a horizontal-gradient signature rarely share
// the vertical one too — while retaining the brightness and contrast invariance
// of the difference hash.
//
// The zero value is ready to use; [NewDHashCombined] is provided for symmetry.
type DHashCombined struct{}

// NewDHashCombined returns a ready-to-use [DHashCombined].
func NewDHashCombined() DHashCombined { return DHashCombined{} }

// Compute returns the 16-byte combined difference hash of img: the eight bytes
// of the horizontal hash followed by the eight bytes of the vertical hash.
func (DHashCombined) Compute(img *cv.Mat) []byte {
	requireImage(img, "DHashCombined.Compute")
	h := packBits(horizontalBits(grayResize(img, 9, 8)))
	v := packBits(verticalBits(grayResize(img, 8, 9)))
	out := make([]byte, 0, 16)
	out = append(out, h...)
	out = append(out, v...)
	return out
}

// Compare returns the Hamming distance between two combined difference hashes.
func (DHashCombined) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "DHashCombined.Compare")
	return float64(hamming(a, b))
}

// DifferenceCombined is a convenience wrapper returning the [DHashCombined] of
// img.
func DifferenceCombined(img *cv.Mat) []byte { return DHashCombined{}.Compute(img) }
