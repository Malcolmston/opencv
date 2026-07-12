package imghash

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// PeakCrossCorrelation returns the peak normalised cross-correlation between two
// equal-length real vectors over all circular shifts, a value in [-1, 1]. For
// each shift the Pearson correlation coefficient of a against the circularly
// shifted b is computed, and the maximum over every shift is returned. Identical
// vectors give 1; a circularly shifted copy also gives 1, which is exactly the
// rotation invariance OpenCV exploits for radial-variance hashes (a rotation of
// the image circularly shifts its per-angle variance profile). It panics if the
// vectors differ in length or are empty.
//
// This is the comparison metric OpenCV's cv::img_hash::RadialVarianceHash uses;
// [RadialVarianceCorrHash] wraps it as an [ImgHash] distance.
func PeakCrossCorrelation(a, b []float64) float64 {
	n := len(a)
	if n == 0 || len(b) != n {
		panic("imghash: PeakCrossCorrelation requires equal, non-empty vectors")
	}
	ma := mean(a)
	mb := mean(b)
	da := make([]float64, n)
	db := make([]float64, n)
	var na, nb float64
	for i := 0; i < n; i++ {
		da[i] = a[i] - ma
		db[i] = b[i] - mb
		na += da[i] * da[i]
		nb += db[i] * db[i]
	}
	denom := math.Sqrt(na * nb)
	if denom == 0 {
		// One vector is constant; correlation is undefined. Treat two constant
		// vectors as identical, a constant against a varying one as uncorrelated.
		if na == 0 && nb == 0 {
			return 1
		}
		return 0
	}
	peak := math.Inf(-1)
	for shift := 0; shift < n; shift++ {
		var acc float64
		for i := 0; i < n; i++ {
			acc += da[i] * db[(i+shift)%n]
		}
		if c := acc / denom; c > peak {
			peak = c
		}
	}
	return peak
}

// RadialVarianceCorrHash computes the same 40-byte radial-variance fingerprint
// as [RadialVarianceHash] but compares two fingerprints by OpenCV's peak
// cross-correlation instead of the L1 distance. The [Compare] result is
// 1 − peak-correlation, so identical images give 0, a rotated copy (whose
// variance profile is a circular shift) also gives near 0, and unrelated images
// give a value approaching 1 or more. This is the deferred peak-cross-correlation
// comparison noted in the package documentation.
//
// Because the comparison is shift-invariant, this hash tolerates image rotation
// far better than the L1-based [RadialVarianceHash]; use it when rotational
// near-duplicates must be caught.
//
// The zero value is ready to use; [NewRadialVarianceCorrHash] is provided for
// symmetry.
type RadialVarianceCorrHash struct{}

// NewRadialVarianceCorrHash returns a ready-to-use [RadialVarianceCorrHash].
func NewRadialVarianceCorrHash() RadialVarianceCorrHash { return RadialVarianceCorrHash{} }

// Compute returns the 40-byte radial-variance hash of img, identical to
// [RadialVarianceHash.Compute].
func (RadialVarianceCorrHash) Compute(img *cv.Mat) []byte {
	requireImage(img, "RadialVarianceCorrHash.Compute")
	return RadialVarianceHash{}.Compute(img)
}

// Compare returns 1 − the peak cross-correlation of the two hashes, a distance
// in roughly [0, 2] that is 0 for identical (or purely rotated) images.
func (RadialVarianceCorrHash) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "RadialVarianceCorrHash.Compare")
	fa := make([]float64, len(a))
	fb := make([]float64, len(b))
	for i := range a {
		fa[i] = float64(a[i])
		fb[i] = float64(b[i])
	}
	return 1 - PeakCrossCorrelation(fa, fb)
}

// RadialVarianceCorr is a convenience wrapper returning the
// [RadialVarianceCorrHash] of img.
func RadialVarianceCorr(img *cv.Mat) []byte { return RadialVarianceCorrHash{}.Compute(img) }
