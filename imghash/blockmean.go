package imghash

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// BlockMeanHash implements a block mean-value hash in the spirit of Yang et al.
// and OpenCV's cv::img_hash::BlockMeanHash. The image is reduced to grayscale
// and scaled to a fixed square, partitioned into a grid of equal, non-
// overlapping blocks, and the mean of each block is compared against the mean
// of all block means. Each block contributes one bit, set when its mean exceeds
// the global mean. With the default 16×16 grid the result is a 256-bit
// (32-byte) fingerprint compared by Hamming distance.
//
// A larger grid captures more spatial detail at the cost of a longer hash. The
// hash is robust to uniform brightness shifts because the threshold moves with
// the image mean.
type BlockMeanHash struct {
	// blocks is the number of blocks per side of the grid.
	blocks int
}

// defaultBlockMeanBlocks is the grid resolution used by the zero value and by
// the [BlockMean] convenience function: a 16×16 grid, i.e. 256 bits.
const defaultBlockMeanBlocks = 16

// NewBlockMeanHash returns a [BlockMeanHash] using a blocks×blocks grid. It
// panics if blocks is not positive. Use a value whose square is a multiple of 8
// (for example 8, 16 or 24) so the bit count fills whole bytes.
func NewBlockMeanHash(blocks int) BlockMeanHash {
	if blocks <= 0 {
		panic(fmt.Sprintf("imghash: NewBlockMeanHash requires positive blocks, got %d", blocks))
	}
	return BlockMeanHash{blocks: blocks}
}

// grid returns the configured grid resolution, defaulting the zero value.
func (h BlockMeanHash) grid() int {
	if h.blocks <= 0 {
		return defaultBlockMeanBlocks
	}
	return h.blocks
}

// Compute returns the block mean hash of img. The hash is blocks*blocks bits
// packed into ceil(blocks*blocks/8) bytes.
func (h BlockMeanHash) Compute(img *cv.Mat) []byte {
	requireImage(img, "BlockMeanHash.Compute")
	n := h.grid()
	// Work at an exact multiple of the grid so blocks tile evenly.
	block := 16
	side := n * block
	small := grayResize(img, side, side)

	means := make([]float64, n*n)
	for by := 0; by < n; by++ {
		for bx := 0; bx < n; bx++ {
			var sum int
			for y := 0; y < block; y++ {
				rowBase := (by*block+y)*side + bx*block
				for x := 0; x < block; x++ {
					sum += int(small.Data[rowBase+x])
				}
			}
			means[by*n+bx] = float64(sum) / float64(block*block)
		}
	}
	thr := mean(means)
	bitsOut := make([]bool, n*n)
	for i := range means {
		bitsOut[i] = means[i] > thr
	}
	return packBits(bitsOut)
}

// Compare returns the Hamming distance between two block mean hashes.
func (BlockMeanHash) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "BlockMeanHash.Compare")
	return float64(hamming(a, b))
}

// BlockMean is a convenience wrapper returning the default [BlockMeanHash] of
// img (a 16×16 grid, 32 bytes).
func BlockMean(img *cv.Mat) []byte { return BlockMeanHash{}.Compute(img) }
