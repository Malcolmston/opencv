package stitch

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SeamCostMap returns the per-pixel disagreement cost between two same-sized
// images, the energy a seam finder routes around. Each cell is the sum over
// channels of the squared difference of the corresponding samples. It panics if
// the images differ in size or channel count.
func SeamCostMap(a, b *cv.Mat) *cv.FloatMat {
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		panic("stitch: SeamCostMap requires matching image dimensions")
	}
	cost := cv.NewFloatMat(a.Rows, a.Cols)
	ch := a.Channels
	for p := 0; p < a.Rows*a.Cols; p++ {
		base := p * ch
		var s float64
		for c := 0; c < ch; c++ {
			d := float64(a.Data[base+c]) - float64(b.Data[base+c])
			s += d * d
		}
		cost.Data[p] = s
	}
	return cost
}

// FindVerticalSeamDP finds the minimum-cost top-to-bottom seam through the cost
// map by dynamic programming. The seam contains exactly one column per row and
// moves by at most one column between adjacent rows. It returns a slice of length
// cost.Rows giving the seam column in each row. It panics on an empty map.
//
// The cumulative cost of a cell is its own cost plus the least cumulative cost of
// the three cells above it (up-left, up, up-right); the seam is recovered by
// back-tracking from the cheapest cell in the last row.
func FindVerticalSeamDP(cost *cv.FloatMat) []int {
	rows, cols := cost.Rows, cost.Cols
	if rows == 0 || cols == 0 {
		panic("stitch: FindVerticalSeamDP on empty cost map")
	}
	dp := make([]float64, rows*cols)
	back := make([]int, rows*cols)
	for x := 0; x < cols; x++ {
		dp[x] = cost.Data[x]
	}
	for y := 1; y < rows; y++ {
		for x := 0; x < cols; x++ {
			best := dp[(y-1)*cols+x]
			bx := x
			if x > 0 && dp[(y-1)*cols+x-1] < best {
				best = dp[(y-1)*cols+x-1]
				bx = x - 1
			}
			if x < cols-1 && dp[(y-1)*cols+x+1] < best {
				best = dp[(y-1)*cols+x+1]
				bx = x + 1
			}
			dp[y*cols+x] = cost.Data[y*cols+x] + best
			back[y*cols+x] = bx
		}
	}
	endX := 0
	bestEnd := math.Inf(1)
	for x := 0; x < cols; x++ {
		if v := dp[(rows-1)*cols+x]; v < bestEnd {
			bestEnd = v
			endX = x
		}
	}
	seam := make([]int, rows)
	x := endX
	for y := rows - 1; y >= 0; y-- {
		seam[y] = x
		if y > 0 {
			x = back[y*cols+x]
		}
	}
	return seam
}

// FindHorizontalSeamDP finds the minimum-cost left-to-right seam through the cost
// map. The seam contains exactly one row per column and moves by at most one row
// between adjacent columns. It returns a slice of length cost.Cols giving the
// seam row in each column. It panics on an empty map.
func FindHorizontalSeamDP(cost *cv.FloatMat) []int {
	rows, cols := cost.Rows, cost.Cols
	if rows == 0 || cols == 0 {
		panic("stitch: FindHorizontalSeamDP on empty cost map")
	}
	dp := make([]float64, rows*cols)
	back := make([]int, rows*cols)
	for y := 0; y < rows; y++ {
		dp[y*cols] = cost.Data[y*cols]
	}
	for x := 1; x < cols; x++ {
		for y := 0; y < rows; y++ {
			best := dp[y*cols+x-1]
			by := y
			if y > 0 && dp[(y-1)*cols+x-1] < best {
				best = dp[(y-1)*cols+x-1]
				by = y - 1
			}
			if y < rows-1 && dp[(y+1)*cols+x-1] < best {
				best = dp[(y+1)*cols+x-1]
				by = y + 1
			}
			dp[y*cols+x] = cost.Data[y*cols+x] + best
			back[y*cols+x] = by
		}
	}
	endY := 0
	bestEnd := math.Inf(1)
	for y := 0; y < rows; y++ {
		if v := dp[y*cols+cols-1]; v < bestEnd {
			bestEnd = v
			endY = y
		}
	}
	seam := make([]int, cols)
	y := endY
	for x := cols - 1; x >= 0; x-- {
		seam[x] = y
		if x > 0 {
			y = back[y*cols+x]
		}
	}
	return seam
}

// SeamMaskFromColumns builds a single-channel mask of size rows×cols from a
// vertical seam (one column per row, as returned by [FindVerticalSeamDP]).
// Pixels strictly left of the seam column, and the seam column itself, are set to
// 255; pixels to the right are 0. This assigns the left image everything up to
// the seam and the right image everything beyond it.
func SeamMaskFromColumns(seam []int, rows, cols int) *cv.Mat {
	mask := cv.NewMat(rows, cols, 1)
	for y := 0; y < rows && y < len(seam); y++ {
		s := seam[y]
		for x := 0; x <= s && x < cols; x++ {
			mask.Data[y*cols+x] = 255
		}
	}
	return mask
}

// DPSeamFinder routes the boundary between two overlapping images through the
// pixels where they agree most, using dynamic-programming seam finding on the
// per-pixel colour difference. It implements the classic seam step of the
// stitching pipeline.
type DPSeamFinder struct {
	// Horizontal selects a left-to-right seam (images stacked vertically) when
	// true; the default false finds a top-to-bottom seam for side-by-side images.
	Horizontal bool
}

// Find computes the optimal seam between two same-sized, same-channel images a
// and b and returns a single-channel mask: 255 where image a should be used and
// 0 where image b should be used. For a vertical seam, a wins to the left of the
// cut; for a horizontal seam, a wins above it. It panics if the images differ in
// size or channel count.
func (f DPSeamFinder) Find(a, b *cv.Mat) *cv.Mat {
	cost := SeamCostMap(a, b)
	rows, cols := a.Rows, a.Cols
	mask := cv.NewMat(rows, cols, 1)
	if f.Horizontal {
		seam := FindHorizontalSeamDP(cost)
		for x := 0; x < cols; x++ {
			s := seam[x]
			for y := 0; y <= s && y < rows; y++ {
				mask.Data[y*cols+x] = 255
			}
		}
		return mask
	}
	seam := FindVerticalSeamDP(cost)
	return SeamMaskFromColumns(seam, rows, cols)
}
