package imghash2

import cv "github.com/malcolmston/opencv"

const (
	// wHashSize is the side length of the working image fed to the Haar
	// transform. It must be a power of two so the transform can recurse down to
	// the retained low-frequency subband.
	wHashSize = 64
	// wHashLevels is the number of decomposition levels applied before the LL
	// subband is thresholded. Three levels reduce a 64×64 image to an 8×8 LL
	// band, giving a 64-bit hash.
	wHashLevels = 3
)

// WaveletHash implements a wavelet hash (wHash) in the spirit of Krawetz's
// wavelet fingerprint. The image is reduced to grayscale and scaled to 64×64,
// and a multi-level 2-D Haar discrete wavelet transform is applied; after three
// levels of decomposition the low-frequency 8×8 LL subband survives. Each of
// its 64 coefficients becomes one bit, set when it exceeds the median of the
// band. The result is a 64-bit (8-byte) [Hash] compared by Hamming distance.
//
// Like [PHash] the hash works on a compact low-frequency representation and
// thresholds at the median, so it is robust to gamma, uniform brightness shifts
// and mild blur. The Haar basis is spatially localised, so wHash also keys on
// where structure lives, a useful complement to the DCT-based [PHash]. The zero
// value is ready to use; [NewWaveletHash] is provided for symmetry.
type WaveletHash struct{}

// NewWaveletHash returns a ready-to-use [WaveletHash].
func NewWaveletHash() WaveletHash { return WaveletHash{} }

// Name returns the identifier "whash".
func (WaveletHash) Name() string { return "whash" }

// Bits returns 64, the fixed length of a wavelet hash.
func (WaveletHash) Bits() int { return 64 }

// Compute returns the 8-byte wavelet hash of img.
func (WaveletHash) Compute(img *cv.Mat) Hash {
	requireImage(img, "WaveletHash.Compute")
	buf := grayFloats(img, wHashSize, wHashSize)

	// Apply successive Haar decompositions to the shrinking LL subband. After
	// each level the LL band occupies the top-left size×size quadrant.
	size := wHashSize
	for l := 0; l < wHashLevels; l++ {
		sub := extractTopLeft(buf, wHashSize, size)
		t := HaarDWT2D(sub, size)
		insertTopLeft(buf, wHashSize, t, size)
		size /= 2
	}

	ll := extractTopLeft(buf, wHashSize, size)
	thr := Median(ll)
	out := make([]bool, size*size)
	for i, v := range ll {
		out[i] = v > thr
	}
	return packBits(out)
}

// extractTopLeft copies the top-left size×size quadrant of a full×full
// row-major matrix into a fresh size×size row-major slice.
func extractTopLeft(full []float64, fullSide, size int) []float64 {
	out := make([]float64, size*size)
	for y := 0; y < size; y++ {
		copy(out[y*size:y*size+size], full[y*fullSide:y*fullSide+size])
	}
	return out
}

// insertTopLeft writes a size×size row-major sub-matrix back into the top-left
// quadrant of a full×full row-major matrix.
func insertTopLeft(full []float64, fullSide int, sub []float64, size int) {
	for y := 0; y < size; y++ {
		copy(full[y*fullSide:y*fullSide+size], sub[y*size:y*size+size])
	}
}

// Wavelet is a convenience wrapper returning the [WaveletHash] of img.
func Wavelet(img *cv.Mat) Hash { return WaveletHash{}.Compute(img) }
