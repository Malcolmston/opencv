package imghash

import cv "github.com/malcolmston/opencv"

// mhSize is the side length of the working image the LoG filter runs on.
const mhSize = 32

// logKernel is a 5×5 Laplacian-of-Gaussian (Marr–Hildreth) kernel. Its taps sum
// to zero, so it responds to edges and zero-crossings while ignoring the local
// DC level, giving brightness invariance.
var logKernel = [5][5]float64{
	{0, 0, -1, 0, 0},
	{0, -1, -2, -1, 0},
	{-1, -2, 16, -2, -1},
	{0, -1, -2, -1, 0},
	{0, 0, -1, 0, 0},
}

// MarrHildrethHash implements a Marr–Hildreth (Laplacian-of-Gaussian) hash in
// the spirit of Zauner's MH hash and OpenCV's cv::img_hash::MarrHildrethHash.
// The image is reduced to grayscale, scaled to 32×32 and convolved with a 5×5
// LoG kernel; the response is then pooled into an 8×8 grid of block means and
// each block becomes one bit, set when its mean response exceeds the mean of all
// block responses. The result is a 64-bit (8-byte) fingerprint compared by
// Hamming distance.
//
// Filtering with the LoG operator emphasises edge structure and zero-crossings,
// so the hash keys on shape rather than on absolute intensity. This is a
// simplified variant: the exact 72-bit, multi-orientation Marr–Hildreth
// descriptor with its scale/orientation parameters is deferred (see the package
// documentation).
//
// The zero value is ready to use; [NewMarrHildrethHash] is provided for
// symmetry.
type MarrHildrethHash struct{}

// NewMarrHildrethHash returns a ready-to-use [MarrHildrethHash].
func NewMarrHildrethHash() MarrHildrethHash { return MarrHildrethHash{} }

// Compute returns the 8-byte Marr–Hildreth hash of img.
func (MarrHildrethHash) Compute(img *cv.Mat) []byte {
	requireImage(img, "MarrHildrethHash.Compute")
	small := grayResize(img, mhSize, mhSize)

	// Convolve with the LoG kernel, replicating the border.
	resp := make([]float64, mhSize*mhSize)
	for y := 0; y < mhSize; y++ {
		for x := 0; x < mhSize; x++ {
			var acc float64
			for ky := -2; ky <= 2; ky++ {
				sy := clampIndex(y+ky, mhSize)
				for kx := -2; kx <= 2; kx++ {
					sx := clampIndex(x+kx, mhSize)
					acc += logKernel[ky+2][kx+2] * float64(small.Data[sy*mhSize+sx])
				}
			}
			resp[y*mhSize+x] = acc
		}
	}

	// Pool the response into an 8×8 grid of block means.
	const grid = 8
	const block = mhSize / grid // 4
	means := make([]float64, grid*grid)
	for by := 0; by < grid; by++ {
		for bx := 0; bx < grid; bx++ {
			var sum float64
			for y := 0; y < block; y++ {
				for x := 0; x < block; x++ {
					sum += resp[(by*block+y)*mhSize+(bx*block+x)]
				}
			}
			means[by*grid+bx] = sum / float64(block*block)
		}
	}
	thr := mean(means)
	bitsOut := make([]bool, grid*grid)
	for i := range means {
		bitsOut[i] = means[i] > thr
	}
	return packBits(bitsOut)
}

// Compare returns the Hamming distance between two Marr–Hildreth hashes.
func (MarrHildrethHash) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "MarrHildrethHash.Compare")
	return float64(hamming(a, b))
}

// clampIndex clamps i into [0, n) for border replication.
func clampIndex(i, n int) int {
	if i < 0 {
		return 0
	}
	if i >= n {
		return n - 1
	}
	return i
}

// MarrHildreth is a convenience wrapper returning the [MarrHildrethHash] of img.
func MarrHildreth(img *cv.Mat) []byte { return MarrHildrethHash{}.Compute(img) }
