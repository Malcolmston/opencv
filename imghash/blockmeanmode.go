package imghash

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// Block mean projection modes, matching OpenCV's cv::img_hash::BlockMeanHash
// mode enumeration.
const (
	// BlockMeanMode0 tiles the image with non-overlapping blocks, giving n×n
	// bits for an n-block grid.
	BlockMeanMode0 = 0
	// BlockMeanMode1 slides the block window in half-block steps so adjacent
	// blocks overlap by 50%, giving (2n-1)×(2n-1) bits for an n-block grid. The
	// overlap makes the fingerprint smoother and more robust to small spatial
	// shifts, at the cost of a longer hash.
	BlockMeanMode1 = 1
)

// bmModeBlock is the pixel side of one block; the working image is sized so the
// grid tiles it exactly.
const bmModeBlock = 16

// BlockMeanModeHash implements the block mean-value hash with a selectable
// projection mode, matching OpenCV's cv::img_hash::BlockMeanHash more closely
// than the mode-0-only [BlockMeanHash]. The image is reduced to grayscale and
// scaled to a fixed square, block means are collected either without overlap
// (mode 0) or with 50% overlap (mode 1), and each block mean is compared against
// the mean of all block means. Each block contributes one bit, set when its mean
// exceeds the global mean; the bits are compared by Hamming distance.
//
// For an n-block grid mode 0 yields n² bits and mode 1 yields (2n-1)² bits.
// Hashes of different modes or grids have different lengths and must not be
// compared against one another.
//
// The zero value behaves as a mode-0 16×16 grid, matching [BlockMeanHash];
// [NewBlockMeanModeHash] sets the grid and mode explicitly.
type BlockMeanModeHash struct {
	blocks int
	mode   int
}

// NewBlockMeanModeHash returns a [BlockMeanModeHash] using a blocks×blocks grid
// and the given projection mode ([BlockMeanMode0] or [BlockMeanMode1]). It
// panics if blocks is not positive or mode is unknown.
func NewBlockMeanModeHash(blocks, mode int) BlockMeanModeHash {
	if blocks <= 0 {
		panic(fmt.Sprintf("imghash: NewBlockMeanModeHash requires positive blocks, got %d", blocks))
	}
	if mode != BlockMeanMode0 && mode != BlockMeanMode1 {
		panic(fmt.Sprintf("imghash: NewBlockMeanModeHash unknown mode %d", mode))
	}
	return BlockMeanModeHash{blocks: blocks, mode: mode}
}

func (h BlockMeanModeHash) grid() int {
	if h.blocks <= 0 {
		return defaultBlockMeanBlocks
	}
	return h.blocks
}

// Compute returns the block mean hash of img for the configured grid and mode.
func (h BlockMeanModeHash) Compute(img *cv.Mat) []byte {
	requireImage(img, "BlockMeanModeHash.Compute")
	n := h.grid()
	side := n * bmModeBlock
	small := grayResize(img, side, side)

	// Determine the top-left origins of every block. Mode 0 steps a whole block;
	// mode 1 steps half a block so windows overlap by 50%.
	var step int
	var count int
	switch h.mode {
	case BlockMeanMode1:
		step = bmModeBlock / 2
		count = 2*n - 1
	default:
		step = bmModeBlock
		count = n
	}

	means := make([]float64, 0, count*count)
	for by := 0; by < count; by++ {
		oy := by * step
		for bx := 0; bx < count; bx++ {
			ox := bx * step
			var sum int
			for y := 0; y < bmModeBlock; y++ {
				rowBase := (oy+y)*side + ox
				for x := 0; x < bmModeBlock; x++ {
					sum += int(small.Data[rowBase+x])
				}
			}
			means = append(means, float64(sum)/float64(bmModeBlock*bmModeBlock))
		}
	}

	thr := mean(means)
	bitsOut := make([]bool, len(means))
	for i, m := range means {
		bitsOut[i] = m > thr
	}
	return packBits(bitsOut)
}

// Compare returns the Hamming distance between two block mean hashes.
func (BlockMeanModeHash) Compare(a, b []byte) float64 {
	requireSameLen(a, b, "BlockMeanModeHash.Compare")
	return float64(hamming(a, b))
}

// BlockMeanMode is a convenience wrapper returning the default 16×16
// [BlockMeanModeHash] of img in the given mode.
func BlockMeanMode(img *cv.Mat, mode int) []byte {
	return NewBlockMeanModeHash(defaultBlockMeanBlocks, mode).Compute(img)
}
