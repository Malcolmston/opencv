package quality

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Entropy returns the Shannon entropy, in bits, of the luminance histogram of
// img. It measures the average information content of the image: a flat image
// has entropy 0, and the maximum for 8-bit data is 8 bits (a perfectly uniform
// histogram). img is reduced to luminance first. It panics on an empty image.
func Entropy(img *cv.Mat) float64 {
	requireImage(img, "Entropy")
	return entropyOf(toGray(img))
}

// entropyOf computes the Shannon entropy (bits) of a luminance grid quantised
// into 256 bins.
func entropyOf(g grid) float64 {
	var hist [256]float64
	for _, v := range g.data {
		b := int(v + 0.5)
		if b < 0 {
			b = 0
		} else if b > 255 {
			b = 255
		}
		hist[b]++
	}
	n := float64(len(g.data))
	var h float64
	for _, c := range hist {
		if c > 0 {
			p := c / n
			h -= p * math.Log2(p)
		}
	}
	return h
}

// EntropyDiff returns the absolute difference between the luminance [Entropy] of
// a and that of b. Identical images (and, more generally, any two images with
// the same luminance histogram) score zero; distortions that reshape the
// histogram — additive noise broadening it, blur concentrating it — drive the
// score up. It is symmetric in its arguments. It panics unless the two images
// share a size and channel count.
func EntropyDiff(a, b *cv.Mat) float64 {
	requireComparable(a, b, "EntropyDiff")
	return math.Abs(entropyOf(toGray(a)) - entropyOf(toGray(b)))
}

// EdgePreservationRatio returns the edge-preservation index of candidate b
// relative to reference a: the Pearson correlation coefficient between their
// high-pass (Laplacian) responses. It is the standard β measure used to judge
// how well an edge-aware filter keeps structure. The score lies in [-1, 1]; it
// is 1 when the two edge maps are identical (as for identical images) and falls
// toward 0 as blur weakens edges or noise decorrelates them.
//
// Both images are reduced to luminance and high-pass filtered with the
// 4-neighbour Laplacian. It panics unless the two images share a size and
// channel count.
func EdgePreservationRatio(a, b *cv.Mat) float64 {
	requireComparable(a, b, "EdgePreservationRatio")
	la := conv3(toGray(a), laplacian4)
	lb := conv3(toGray(b), laplacian4)
	ma := meanOf(la.data)
	mb := meanOf(lb.data)
	var cov, va, vb float64
	for i := range la.data {
		da := la.data[i] - ma
		db := lb.data[i] - mb
		cov += da * db
		va += da * da
		vb += db * db
	}
	denom := math.Sqrt(va * vb)
	if denom == 0 {
		// Both high-pass responses are constant: identical flat structure.
		return 1
	}
	return cov / denom
}
