package imghash

import cv "github.com/malcolmston/opencv"

// wHashSize is the side length of the working image fed to the Haar transform.
// It must be a power of two so the discrete wavelet transform can recurse down
// to the retained low-frequency subband.
const wHashSize = 64

// wHashLevels is the number of decomposition levels applied before the
// low-frequency LL subband is thresholded. Three levels reduce a 64×64 image to
// an 8×8 LL band, giving a 64-bit hash.
const wHashLevels = 3

// haar1DForward applies one level of the orthonormal Haar discrete wavelet
// transform to the first n elements of vec in place. The even-indexed outputs
// become the scaling (approximation) coefficients a = (x0+x1)/√2 and the
// odd-indexed outputs the wavelet (detail) coefficients d = (x0-x1)/√2, but they
// are deinterleaved so the n/2 approximation coefficients occupy the low half
// [0, n/2) and the n/2 detail coefficients the high half [n/2, n). The transform
// is orthonormal, so [haar1DInverse] reconstructs vec exactly.
func haar1DForward(vec []float64, n int) {
	tmp := make([]float64, n)
	half := n / 2
	for i := 0; i < half; i++ {
		a := vec[2*i]
		b := vec[2*i+1]
		tmp[i] = (a + b) * invSqrt2
		tmp[half+i] = (a - b) * invSqrt2
	}
	copy(vec[:n], tmp)
}

// haar1DInverse is the exact inverse of [haar1DForward] for the first n
// elements of vec, reconstructing the interleaved samples from the deinterleaved
// approximation and detail halves.
func haar1DInverse(vec []float64, n int) {
	tmp := make([]float64, n)
	half := n / 2
	for i := 0; i < half; i++ {
		a := vec[i]
		d := vec[half+i]
		tmp[2*i] = (a + d) * invSqrt2
		tmp[2*i+1] = (a - d) * invSqrt2
	}
	copy(vec[:n], tmp)
}

// invSqrt2 is 1/√2, the orthonormal Haar normalisation factor.
const invSqrt2 = 0.70710678118654752440

// HaarDWT2D returns one level of the separable 2-D orthonormal Haar discrete
// wavelet transform of an n×n block stored row-major in a flat slice of length
// n*n. The transform is applied to every row and then to every column, so the
// result is partitioned into four n/2×n/2 subbands: LL (top-left, a coarser
// approximation), LH, HL and HH (the horizontal, vertical and diagonal detail).
// n must be even. The input is not modified; a fresh slice is returned.
//
// The transform is orthonormal and exactly invertible by [HaarIDWT2D], the
// property the wavelet hash and its round-trip test rely on.
func HaarDWT2D(in []float64, n int) []float64 {
	if n%2 != 0 {
		panic("imghash: HaarDWT2D requires an even side length")
	}
	if len(in) != n*n {
		panic("imghash: HaarDWT2D input length must be n*n")
	}
	out := make([]float64, n*n)
	copy(out, in)
	// Transform rows.
	row := make([]float64, n)
	for y := 0; y < n; y++ {
		copy(row, out[y*n:y*n+n])
		haar1DForward(row, n)
		copy(out[y*n:y*n+n], row)
	}
	// Transform columns.
	col := make([]float64, n)
	for x := 0; x < n; x++ {
		for y := 0; y < n; y++ {
			col[y] = out[y*n+x]
		}
		haar1DForward(col, n)
		for y := 0; y < n; y++ {
			out[y*n+x] = col[y]
		}
	}
	return out
}

// HaarIDWT2D is the exact inverse of [HaarDWT2D]: given one level of 2-D Haar
// coefficients for an n×n block it reconstructs the original samples. Columns
// are inverted first and then rows, undoing the forward order. The input is not
// modified; a fresh slice is returned.
func HaarIDWT2D(in []float64, n int) []float64 {
	if n%2 != 0 {
		panic("imghash: HaarIDWT2D requires an even side length")
	}
	if len(in) != n*n {
		panic("imghash: HaarIDWT2D input length must be n*n")
	}
	out := make([]float64, n*n)
	copy(out, in)
	// Invert columns.
	col := make([]float64, n)
	for x := 0; x < n; x++ {
		for y := 0; y < n; y++ {
			col[y] = out[y*n+x]
		}
		haar1DInverse(col, n)
		for y := 0; y < n; y++ {
			out[y*n+x] = col[y]
		}
	}
	// Invert rows.
	row := make([]float64, n)
	for y := 0; y < n; y++ {
		copy(row, out[y*n:y*n+n])
		haar1DInverse(row, n)
		copy(out[y*n:y*n+n], row)
	}
	return out
}

// WaveletHash implements a wavelet hash (wHash) in the spirit of Krawetz's
// wavelet fingerprint. The image is reduced to grayscale and scaled to 64×64,
// and a multi-level 2-D Haar discrete wavelet transform is applied; after three
// levels of decomposition the low-frequency 8×8 LL subband survives. Each of its
// 64 coefficients becomes one bit, set when it exceeds the median of the band.
// The result is a 64-bit (8-byte) fingerprint compared by Hamming distance.
//
// Like [PHash] the hash works on a compact low-frequency representation and
// thresholds at the median, so it is robust to gamma, uniform brightness shifts
// and mild blur. The wavelet basis is spatially localised, so wHash keys on
// where structure lives as well as on its frequency, a useful complement to the
// DCT-based [PHash].
//
// The zero value is ready to use; [NewWaveletHash] is provided for symmetry.
type WaveletHash struct{}

// NewWaveletHash returns a ready-to-use [WaveletHash].
func NewWaveletHash() WaveletHash { return WaveletHash{} }

// Compute returns the 8-byte wavelet hash of img.
func (WaveletHash) Compute(img *cv.Mat) []byte {
	requireImage(img, "WaveletHash.Compute")
	small := grayResize(img, wHashSize, wHashSize)

	buf := make([]float64, wHashSize*wHashSize)
	for i := range buf {
		buf[i] = float64(small.Data[i])
	}

	// Apply successive Haar decompositions to the shrinking LL subband. After
	// each level the LL band occupies the top-left size×size quadrant.
	size := wHashSize
	for l := 0; l < wHashLevels; l++ {
		sub := extractTopLeft(buf, wHashSize, size)
		t := HaarDWT2D(sub, size)
		insertTopLeft(buf, wHashSize, t, size)
		size /= 2
	}

	// size is now the side of the final LL band (8 for the defaults).
	ll := extractTopLeft(buf, wHashSize, size)
	thr := median(ll)
	bitsOut := make([]bool, size*size)
	for i := range ll {
		bitsOut[i] = ll[i] > thr
	}
	return packBits(bitsOut)
}

// Compare returns the Hamming distance between two wavelet hashes.
func (WaveletHash) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "WaveletHash.Compare")
	return float64(hamming(a, b))
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

// insertTopLeft writes a size×size row-major sub matrix back into the top-left
// quadrant of a full×full row-major matrix.
func insertTopLeft(full []float64, fullSide int, sub []float64, size int) {
	for y := 0; y < size; y++ {
		copy(full[y*fullSide:y*fullSide+size], sub[y*size:y*size+size])
	}
}

// Wavelet is a convenience wrapper returning the [WaveletHash] of img.
func Wavelet(img *cv.Mat) []byte { return WaveletHash{}.Compute(img) }
