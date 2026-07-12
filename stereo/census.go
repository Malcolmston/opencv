package stereo

import (
	"fmt"
	"math/bits"

	cv "github.com/malcolmston/opencv"
)

// CensusTransform computes the census transform of a grayscale (or RGB,
// converted to gray) image, mirroring the classic non-parametric local
// descriptor used by census-based stereo matchers. For each pixel a bit string
// is formed by comparing every neighbour inside a windowW×windowH window against
// the centre value: the bit is 1 when the neighbour is strictly less than the
// centre and 0 otherwise. The result is a []uint64 in row-major order together
// with the image dimensions.
//
// The window sides must be positive odd integers and, because the code is packed
// into a single uint64, the window may hold at most 65 pixels (64 neighbours
// plus the centre), e.g. up to 9×7, 7×9 or 8×8-1. Border pixels replicate the
// image edge. It panics on empty input, an even window side, or a window with
// more than 64 neighbours.
func CensusTransform(m *cv.Mat, windowW, windowH int) (codes []uint64, rows, cols int) {
	requireOdd(windowW, "CensusTransform.windowW")
	requireOdd(windowH, "CensusTransform.windowH")
	if windowW*windowH-1 > 64 {
		panic(fmt.Sprintf("stereo: CensusTransform window %dx%d has more than 64 neighbours", windowW, windowH))
	}
	g := grayMat(m)
	rows, cols = g.Rows, g.Cols
	grid := make([]int, rows*cols)
	for i := range grid {
		grid[i] = int(g.Data[i])
	}
	hw, hh := windowW/2, windowH/2
	codes = make([]uint64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			center := grid[y*cols+x]
			var code uint64
			bit := 0
			for dy := -hh; dy <= hh; dy++ {
				yy := clampInt(y+dy, 0, rows-1)
				for dx := -hw; dx <= hw; dx++ {
					if dy == 0 && dx == 0 {
						continue
					}
					xx := clampInt(x+dx, 0, cols-1)
					if grid[yy*cols+xx] < center {
						code |= uint64(1) << uint(bit)
					}
					bit++
				}
			}
			codes[y*cols+x] = code
		}
	}
	return codes, rows, cols
}

// HammingDistance64 returns the number of differing bits between a and b, i.e.
// the Hamming distance between two census codes.
func HammingDistance64(a, b uint64) int {
	return bits.OnesCount64(a ^ b)
}

// CensusCostVolume builds a [CostVolume] whose data term is the Hamming distance
// between census codes, a robust illumination-invariant matching cost. The left
// pixel (y, x) is compared against right pixel (y, x-(minDisparity+idx)); when
// blockSize exceeds 1 the Hamming distances are summed over an odd blockSize
// window, giving a small aggregated census cost.
//
// It panics if the images differ in size, on an even window or block side, or on
// a non-positive disparity count.
func CensusCostVolume(left, right *cv.Mat, minDisparity, numDisparities, windowW, windowH, blockSize int) *CostVolume {
	if numDisparities <= 0 {
		panic("stereo: CensusCostVolume requires numDisparities > 0")
	}
	requireOdd(blockSize, "CensusCostVolume.blockSize")
	lc, lr, lcols := CensusTransform(left, windowW, windowH)
	rc, rr, rcols := CensusTransform(right, windowW, windowH)
	if lr != rr || lcols != rcols {
		panic(fmt.Sprintf("stereo: CensusCostVolume size mismatch left %dx%d right %dx%d", lr, lcols, rr, rcols))
	}
	rows, cols := lr, lcols
	half := blockSize / 2
	vol := newCostVolume(rows, cols, minDisparity, numDisparities)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			base := (y*cols + x) * numDisparities
			for idx := 0; idx < numDisparities; idx++ {
				d := minDisparity + idx
				vol.Data[base+idx] = int32(censusBlockCost(lc, rc, rows, cols, y, x, d, half))
			}
		}
	}
	return vol
}

// censusBlockCost sums the Hamming distance between the left census codes at
// (y,x) and the right codes at (y,x-d) over an odd window of half-width half.
func censusBlockCost(lc, rc []uint64, rows, cols, y, x, d, half int) int {
	s := 0
	for dy := -half; dy <= half; dy++ {
		yy := clampInt(y+dy, 0, rows-1)
		rowBase := yy * cols
		for dx := -half; dx <= half; dx++ {
			lx := clampInt(x+dx, 0, cols-1)
			rx := clampInt(x-d+dx, 0, cols-1)
			s += HammingDistance64(lc[rowBase+lx], rc[rowBase+rx])
		}
	}
	return s
}
