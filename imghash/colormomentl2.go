package imghash

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ColorMomentL2Hash computes the same full 42-float colour-moment descriptor as
// [ColorMomentHash] — the seven log-scaled Hu moment invariants of the R, G, B,
// H, S and V planes — but compares two descriptors by the Euclidean (L2)
// distance rather than L1. OpenCV's cv::img_hash::ColorMomentHash reports the L2
// norm of the difference of its feature vectors, so this variant reproduces
// OpenCV's distance scale exactly while [ColorMomentHash] keeps the package's
// L1 convention.
//
// The L2 metric weights a single large per-feature deviation more heavily than
// L1 does, which sharpens the separation between near-duplicates and genuinely
// different images for this descriptor. Identical images still compare as 0.
//
// The zero value is ready to use; [NewColorMomentL2Hash] is provided for
// symmetry.
type ColorMomentL2Hash struct{}

// NewColorMomentL2Hash returns a ready-to-use [ColorMomentL2Hash].
func NewColorMomentL2Hash() ColorMomentL2Hash { return ColorMomentL2Hash{} }

// Compute returns the 336-byte colour-moment hash of img (42 float64 values),
// identical to [ColorMomentHash.Compute].
func (ColorMomentL2Hash) Compute(img *cv.Mat) []byte {
	requireImage(img, "ColorMomentL2Hash.Compute")
	return ColorMomentHash{}.Compute(img)
}

// Compare returns the Euclidean (L2) distance between two colour-moment hashes.
func (ColorMomentL2Hash) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "ColorMomentL2Hash.Compare")
	fa := decodeFloats(a)
	fb := decodeFloats(b)
	var sum float64
	for i := range fa {
		d := fa[i] - fb[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// ColorMomentL2 is a convenience wrapper returning the [ColorMomentL2Hash] of
// img.
func ColorMomentL2(img *cv.Mat) []byte { return ColorMomentL2Hash{}.Compute(img) }
