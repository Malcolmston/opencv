package stereo2

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// MatchCost selects the pointwise window dissimilarity measure used by a
// [BlockMatcher].
type MatchCost int

const (
	// CostSAD is the sum of absolute differences over the window (lower is better).
	CostSAD MatchCost = iota
	// CostSSD is the sum of squared differences over the window (lower is better).
	CostSSD
	// CostNCC is normalized cross-correlation over the window, converted to a
	// cost of 1-NCC so that, like the others, lower is better.
	CostNCC
)

// BlockMatcher is a local ("winner-take-all") block matcher, the counterpart to
// OpenCV's cv::StereoBM. For every reference pixel it slides a square window
// across the disparity search range and keeps the disparity whose window best
// matches, according to [MatchCost].
//
// The zero value is not useful; construct one with [NewBlockMatcher] or set at
// least NumDisparities and BlockSize.
type BlockMatcher struct {
	// MinDisparity is the smallest disparity searched (commonly 0).
	MinDisparity int
	// NumDisparities is the width of the disparity search range in pixels.
	NumDisparities int
	// BlockSize is the odd side length of the matching window.
	BlockSize int
	// Cost selects the window dissimilarity measure.
	Cost MatchCost
	// UniquenessRatio, when positive, rejects a match unless its cost beats the
	// best cost outside its immediate neighbourhood by at least this many percent.
	UniquenessRatio int
}

// NewBlockMatcher returns a BlockMatcher with the given search range and window
// size and SAD cost. BlockSize is rounded up to the next odd number if even. It
// panics if numDisparities or blockSize is not positive.
func NewBlockMatcher(minDisparity, numDisparities, blockSize int) *BlockMatcher {
	if numDisparities <= 0 || blockSize <= 0 {
		panic(fmt.Sprintf("stereo2: NewBlockMatcher requires positive numDisparities and blockSize, got %d, %d", numDisparities, blockSize))
	}
	if blockSize%2 == 0 {
		blockSize++
	}
	return &BlockMatcher{
		MinDisparity:   minDisparity,
		NumDisparities: numDisparities,
		BlockSize:      blockSize,
		Cost:           CostSAD,
	}
}

// Compute matches left against right and returns the left-referenced disparity
// map: for each left pixel it searches the corresponding right pixel at column
// x-d. Pixels near the border, or that fail the uniqueness test, hold
// [InvalidDisparity]. It panics if the images differ in size.
func (bm *BlockMatcher) Compute(left, right *cv.Mat) *DisparityMap {
	stereo2checkPair(left, right)
	return bm.match(left, right, -1)
}

// ComputeRight matches right against left and returns the right-referenced
// disparity map: for each right pixel it searches the corresponding left pixel
// at column x+d. The result is the map consumed, alongside [BlockMatcher.Compute]'s
// output, by [LeftRightCheck]. It panics if the images differ in size.
func (bm *BlockMatcher) ComputeRight(left, right *cv.Mat) *DisparityMap {
	stereo2checkPair(left, right)
	// Reference = right, target = left, matching pixel at x + d (sign +1).
	return bm.match(right, left, +1)
}

// match performs windowed winner-take-all matching. ref supplies the reference
// window and tgt the searched window; sign is -1 when the target column is
// x-d (left reference) and +1 when it is x+d (right reference).
func (bm *BlockMatcher) match(ref, tgt *cv.Mat, sign int) *DisparityMap {
	if bm.NumDisparities <= 0 || bm.BlockSize <= 0 {
		panic("stereo2: BlockMatcher requires positive NumDisparities and BlockSize")
	}
	bs := bm.BlockSize
	if bs%2 == 0 {
		bs++
	}
	r := bs / 2
	rows, cols := ref.Rows, ref.Cols
	out := NewDisparityMap(rows, cols)
	for y := r; y < rows-r; y++ {
		for x := r; x < cols-r; x++ {
			bestCost := float32(math.MaxFloat32)
			secondCost := float32(math.MaxFloat32)
			bestK := -1
			for k := 0; k < bm.NumDisparities; k++ {
				d := bm.MinDisparity + k
				tx := x + sign*d
				if tx-r < 0 || tx+r >= cols {
					continue
				}
				c := windowCost(ref, tgt, y, x, tx, r, bm.Cost)
				if c < bestCost {
					secondCost = bestCost
					bestCost = c
					bestK = k
				} else if c < secondCost {
					secondCost = c
				}
			}
			if bestK < 0 {
				continue
			}
			if bm.UniquenessRatio > 0 && secondCost != float32(math.MaxFloat32) {
				if bestCost*float32(100+bm.UniquenessRatio) > secondCost*100 {
					continue
				}
			}
			out.Data[y*cols+x] = float32(bm.MinDisparity + bestK)
		}
	}
	return out
}

// windowCost evaluates the dissimilarity between the ref window centred at
// (y,x) and the tgt window centred at (y,tx).
func windowCost(ref, tgt *cv.Mat, y, x, tx, r int, cost MatchCost) float32 {
	switch cost {
	case CostSSD:
		var s float64
		for j := -r; j <= r; j++ {
			for i := -r; i <= r; i++ {
				d := stereo2intensity(ref, y+j, x+i) - stereo2intensity(tgt, y+j, tx+i)
				s += d * d
			}
		}
		return float32(s)
	case CostNCC:
		var sa, sb, saa, sbb, sab float64
		n := 0.0
		for j := -r; j <= r; j++ {
			for i := -r; i <= r; i++ {
				a := stereo2intensity(ref, y+j, x+i)
				b := stereo2intensity(tgt, y+j, tx+i)
				sa += a
				sb += b
				saa += a * a
				sbb += b * b
				sab += a * b
				n++
			}
		}
		cov := sab - sa*sb/n
		va := saa - sa*sa/n
		vb := sbb - sb*sb/n
		denom := math.Sqrt(va * vb)
		if denom <= 1e-12 {
			// A flat window: perfect match only if the other is flat too.
			if math.Abs(cov) <= 1e-12 {
				return 0
			}
			return 2
		}
		ncc := cov / denom
		return float32(1 - ncc)
	default: // CostSAD
		var s float64
		for j := -r; j <= r; j++ {
			for i := -r; i <= r; i++ {
				d := stereo2intensity(ref, y+j, x+i) - stereo2intensity(tgt, y+j, tx+i)
				if d < 0 {
					d = -d
				}
				s += d
			}
		}
		return float32(s)
	}
}

// BlockMatchSAD is a convenience wrapper that runs a sum-of-absolute-differences
// [BlockMatcher] over the pair. blockSize is rounded up to odd. It panics if the
// images differ in size or the parameters are not positive.
func BlockMatchSAD(left, right *cv.Mat, minDisparity, numDisparities, blockSize int) *DisparityMap {
	bm := NewBlockMatcher(minDisparity, numDisparities, blockSize)
	bm.Cost = CostSAD
	return bm.Compute(left, right)
}

// BlockMatchSSD is a convenience wrapper that runs a sum-of-squared-differences
// [BlockMatcher] over the pair. blockSize is rounded up to odd. It panics if the
// images differ in size or the parameters are not positive.
func BlockMatchSSD(left, right *cv.Mat, minDisparity, numDisparities, blockSize int) *DisparityMap {
	bm := NewBlockMatcher(minDisparity, numDisparities, blockSize)
	bm.Cost = CostSSD
	return bm.Compute(left, right)
}

// BlockMatchNCC is a convenience wrapper that runs a normalized-cross-correlation
// [BlockMatcher] over the pair. blockSize is rounded up to odd. It panics if the
// images differ in size or the parameters are not positive.
func BlockMatchNCC(left, right *cv.Mat, minDisparity, numDisparities, blockSize int) *DisparityMap {
	bm := NewBlockMatcher(minDisparity, numDisparities, blockSize)
	bm.Cost = CostNCC
	return bm.Compute(left, right)
}
