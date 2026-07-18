package imghash2

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// defaultBlockMeanBlocks is the grid resolution used by the zero value and the
// [BlockMean] convenience function: a 16×16 grid, i.e. 256 bits.
const defaultBlockMeanBlocks = 16

// blockMeanCell is the pixel side length of each grid cell in the working
// image, so a Blocks×Blocks grid runs on a (Blocks·cell)² image.
const blockMeanCell = 8

// BlockMeanHash implements a block mean-value hash in the spirit of Yang et al.
// and OpenCV's cv::img_hash::BlockMeanHash. The image is reduced to grayscale
// and scaled to a fixed square, partitioned into a Blocks×Blocks grid of equal,
// non-overlapping cells, and the mean of each cell is compared against the mean
// of all cell means. Each cell contributes one bit, set when its mean exceeds
// the global mean. With the default 16×16 grid the result is a 256-bit
// (32-byte) [Hash] compared by Hamming distance.
//
// A larger grid captures more spatial detail at the cost of a longer hash. The
// hash is robust to uniform brightness shifts because the threshold moves with
// the image mean. The zero value uses the 16×16 grid; [NewBlockMeanHash]
// chooses the resolution explicitly.
type BlockMeanHash struct {
	// Blocks is the number of cells per side of the grid. A zero value means
	// the 16×16 default. Choose a value whose square is a multiple of 8 (for
	// example 8, 16 or 24) so the bit count fills whole bytes.
	Blocks int
}

// NewBlockMeanHash returns a [BlockMeanHash] using a blocks×blocks grid. It
// panics if blocks is not positive.
func NewBlockMeanHash(blocks int) BlockMeanHash {
	if blocks <= 0 {
		panic(fmt.Sprintf("imghash2: NewBlockMeanHash requires positive blocks, got %d", blocks))
	}
	return BlockMeanHash{Blocks: blocks}
}

// grid returns the configured grid resolution, applying the default.
func (h BlockMeanHash) grid() int {
	if h.Blocks <= 0 {
		return defaultBlockMeanBlocks
	}
	return h.Blocks
}

// Name returns the identifier "blockmean".
func (BlockMeanHash) Name() string { return "blockmean" }

// Bits returns the number of bits in the hash, the square of the grid
// resolution.
func (h BlockMeanHash) Bits() int {
	n := h.grid()
	return n * n
}

// Compute returns the block mean hash of img, Blocks·Blocks bits packed into
// ceil(Blocks·Blocks/8) bytes.
func (h BlockMeanHash) Compute(img *cv.Mat) Hash {
	requireImage(img, "BlockMeanHash.Compute")
	n := h.grid()
	side := n * blockMeanCell
	small := GrayResize(img, side, side)

	means := make([]float64, n*n)
	for by := 0; by < n; by++ {
		for bx := 0; bx < n; bx++ {
			var sum int
			for y := 0; y < blockMeanCell; y++ {
				rowBase := (by*blockMeanCell+y)*side + bx*blockMeanCell
				for x := 0; x < blockMeanCell; x++ {
					sum += int(small.Data[rowBase+x])
				}
			}
			means[by*n+bx] = float64(sum) / float64(blockMeanCell*blockMeanCell)
		}
	}
	thr := Mean(means)
	out := make([]bool, n*n)
	for i, v := range means {
		out[i] = v > thr
	}
	return packBits(out)
}

// BlockMean is a convenience wrapper returning the default [BlockMeanHash] of
// img (a 16×16 grid, 32 bytes).
func BlockMean(img *cv.Mat) Hash { return BlockMeanHash{}.Compute(img) }
