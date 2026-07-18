package texture

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// LBP computes the Local Binary Pattern code for every pixel of img and returns
// the codes as a rows-by-cols grid (row-major, one uint32 per pixel).
//
// For each pixel the neighbours points evenly spaced on a circle of the given
// radius are sampled with bilinear interpolation. Each neighbour contributes a
// bit that is 1 when its value is greater than or equal to the centre value.
// The bits are packed most-significant-first in the order they are visited,
// starting from angle 0 (to the right) and proceeding counter-clockwise, giving
// a code in [0, 2^points). radius must be >= 1 and points in [1, 32].
//
// Border pixels whose sampling circle extends outside the image are given code
// 0; use the interior region for histograms if this matters.
func LBP(img *cv.Mat, radius, points int) [][]uint32 {
	textureRequire(img, "LBP")
	if radius < 1 {
		panic(fmt.Sprintf("texture: LBP requires radius >= 1, got %d", radius))
	}
	if points < 1 || points > 32 {
		panic(fmt.Sprintf("texture: LBP requires points in [1,32], got %d", points))
	}
	rows, cols := img.Rows, img.Cols
	luma := textureLumaFloat(img)
	// Precompute neighbour sample offsets.
	dx := make([]float64, points)
	dy := make([]float64, points)
	for k := 0; k < points; k++ {
		ang := 2 * math.Pi * float64(k) / float64(points)
		dx[k] = float64(radius) * math.Cos(ang)
		dy[k] = -float64(radius) * math.Sin(ang)
	}
	out := make([][]uint32, rows)
	for y := 0; y < rows; y++ {
		out[y] = make([]uint32, cols)
	}
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			center := luma[y*cols+x]
			var code uint32
			inside := true
			for k := 0; k < points; k++ {
				sx := float64(x) + dx[k]
				sy := float64(y) + dy[k]
				if sx < 0 || sx > float64(cols-1) || sy < 0 || sy > float64(rows-1) {
					inside = false
					break
				}
				v := textureBilinear(luma, cols, rows, sx, sy)
				code <<= 1
				if v >= center {
					code |= 1
				}
			}
			if inside {
				out[y][x] = code
			}
		}
	}
	return out
}

// textureBilinear samples a row-major float plane at fractional coordinates
// using bilinear interpolation. Coordinates are assumed in-bounds.
func textureBilinear(data []float64, cols, rows int, x, y float64) float64 {
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	x1 := x0 + 1
	y1 := y0 + 1
	fx := x - float64(x0)
	fy := y - float64(y0)
	if x1 > cols-1 {
		x1 = cols - 1
	}
	if y1 > rows-1 {
		y1 = rows - 1
	}
	v00 := data[y0*cols+x0]
	v01 := data[y0*cols+x1]
	v10 := data[y1*cols+x0]
	v11 := data[y1*cols+x1]
	top := v00 + fx*(v01-v00)
	bot := v10 + fx*(v11-v10)
	return top + fy*(bot-top)
}

// LBPImage computes the classic 3x3, 8-neighbour Local Binary Pattern and
// returns it as a single-channel [cv.Mat] whose samples are the 8-bit codes.
// This is the common visual form of LBP; for other radii or bit counts use
// [LBP], whose codes may exceed 255. Border pixels are set to 0.
func LBPImage(img *cv.Mat) *cv.Mat {
	textureRequire(img, "LBPImage")
	rows, cols := img.Rows, img.Cols
	luma := textureLuma(img)
	dst := cv.NewMat(rows, cols, 1)
	// 8-neighbour order, counter-clockwise from east.
	nx := []int{1, 1, 0, -1, -1, -1, 0, 1}
	ny := []int{0, -1, -1, -1, 0, 1, 1, 1}
	for y := 1; y < rows-1; y++ {
		for x := 1; x < cols-1; x++ {
			c := luma[y*cols+x]
			var code uint8
			for k := 0; k < 8; k++ {
				code <<= 1
				if luma[(y+ny[k])*cols+(x+nx[k])] >= c {
					code |= 1
				}
			}
			dst.Data[y*cols+x] = code
		}
	}
	return dst
}

// bitTransitions counts the number of 0/1 transitions when the low `points`
// bits of code are read as a circular sequence.
func bitTransitions(code uint32, points int) int {
	var t int
	for k := 0; k < points; k++ {
		b0 := (code >> uint(k)) & 1
		b1 := (code >> uint((k+1)%points)) & 1
		if b0 != b1 {
			t++
		}
	}
	return t
}

// popcountLow returns the number of set bits among the low `points` bits.
func popcountLow(code uint32, points int) int {
	var c int
	for k := 0; k < points; k++ {
		c += int((code >> uint(k)) & 1)
	}
	return c
}

// IsUniform reports whether the low `points` bits of code form a uniform
// pattern, i.e. contain at most two circular 0/1 transitions. Uniform patterns
// (edges, corners, spots, flat regions) make up the great majority of patterns
// in natural images and are the basis of the compact uniform LBP histogram.
func IsUniform(code uint32, points int) bool {
	return bitTransitions(code, points) <= 2
}

// UniformBinCount returns the number of histogram bins used by the uniform LBP
// mapping for the given neighbour count: one bin per uniform pattern
// (points*(points-1)+2 of them) plus a single catch-all bin for every
// non-uniform pattern.
func UniformBinCount(points int) int {
	return points*(points-1) + 2 + 1
}

// MapUniform maps an LBP code to its uniform-LBP bin index in
// [0, UniformBinCount(points)). Uniform patterns are labelled by their number
// of set bits combined with their rotation, and all non-uniform patterns share
// the final bin. This is the standard rotation-variant uniform mapping used for
// texture classification.
func MapUniform(code uint32, points int) int {
	if !IsUniform(code, points) {
		return points*(points-1) + 2 // the single non-uniform bin (index)
	}
	// Enumerate uniform codes in a stable order to assign indices. Uniform
	// patterns are: all-zeros, all-ones, and runs of ones. We build the label
	// as: number of leading ... use a direct rank.
	return uniformRank(code, points)
}

// uniformRank returns a stable index in [0, points*(points-1)+2) for a uniform
// code. The all-zero pattern is 0; a pattern with n set bits (1<=n<=points-1)
// starting its run of ones at rotation r gets index 1 + (n-1)*points + r; the
// all-ones pattern is the last index points*(points-1)+1.
func uniformRank(code uint32, points int) int {
	n := popcountLow(code, points)
	if n == 0 {
		return 0
	}
	if n == points {
		return points*(points-1) + 1
	}
	// Find the rotation r such that bit r is 1 and bit r-1 is 0 (start of the
	// single run of ones).
	start := 0
	for r := 0; r < points; r++ {
		cur := (code >> uint(r)) & 1
		prev := (code >> uint((r-1+points)%points)) & 1
		if cur == 1 && prev == 0 {
			start = r
			break
		}
	}
	return 1 + (n-1)*points + start
}

// LBPUniform computes uniform LBP labels for every pixel of img and returns
// them as a rows-by-cols grid. Each label lies in
// [0, UniformBinCount(points)); non-uniform patterns collapse to a single
// label. See [MapUniform].
func LBPUniform(img *cv.Mat, radius, points int) [][]int {
	codes := LBP(img, radius, points)
	out := make([][]int, len(codes))
	for y := range codes {
		row := make([]int, len(codes[y]))
		for x, c := range codes[y] {
			row[x] = MapUniform(c, points)
		}
		out[y] = row
	}
	return out
}

// RotateMinimum returns the rotation-invariant canonical form of the low
// `points` bits of code: the numerically smallest value obtainable by any
// circular bit rotation. Two patterns that differ only by rotation map to the
// same value, giving rotation invariance.
func RotateMinimum(code uint32, points int) uint32 {
	mask := uint32((1 << uint(points)) - 1)
	code &= mask
	min := code
	cur := code
	for k := 1; k < points; k++ {
		// rotate right by 1 within `points` bits
		lsb := cur & 1
		cur = (cur >> 1) | (lsb << uint(points-1))
		cur &= mask
		if cur < min {
			min = cur
		}
	}
	return min
}

// LBPRotationInvariant computes rotation-invariant LBP codes for every pixel of
// img via [RotateMinimum] and returns them as a rows-by-cols grid.
func LBPRotationInvariant(img *cv.Mat, radius, points int) [][]uint32 {
	codes := LBP(img, radius, points)
	for y := range codes {
		for x, c := range codes[y] {
			codes[y][x] = RotateMinimum(c, points)
		}
	}
	return codes
}

// MapUniformRotationInvariant maps an LBP code to its rotation-invariant
// uniform label in [0, points+1]: a uniform pattern is labelled by its number
// of set bits (0..points), and every non-uniform pattern shares the label
// points+1. This is the compact "riu2" descriptor of Ojala et al., the most
// widely used LBP variant.
func MapUniformRotationInvariant(code uint32, points int) int {
	if IsUniform(code, points) {
		return popcountLow(code, points)
	}
	return points + 1
}

// LBPUniformRotationInvariant computes the rotation-invariant uniform ("riu2")
// label for every pixel and returns them as a rows-by-cols grid, each label in
// [0, points+1]. See [MapUniformRotationInvariant].
func LBPUniformRotationInvariant(img *cv.Mat, radius, points int) [][]int {
	codes := LBP(img, radius, points)
	out := make([][]int, len(codes))
	for y := range codes {
		row := make([]int, len(codes[y]))
		for x, c := range codes[y] {
			row[x] = MapUniformRotationInvariant(c, points)
		}
		out[y] = row
	}
	return out
}

// Histogram tallies the labels of a grid (as returned by [LBPUniform] or
// [LBPUniformRotationInvariant]) into a normalised histogram with bins bins.
// Labels outside [0, bins) are ignored. The returned slice sums to 1 unless
// every label was out of range, in which case it is all zeros.
func Histogram(labels [][]int, bins int) []float64 {
	if bins < 1 {
		panic(fmt.Sprintf("texture: Histogram requires bins >= 1, got %d", bins))
	}
	h := make([]float64, bins)
	var total float64
	for _, row := range labels {
		for _, v := range row {
			if v >= 0 && v < bins {
				h[v]++
				total++
			}
		}
	}
	if total > 0 {
		for i := range h {
			h[i] /= total
		}
	}
	return h
}

// LBPUniformHistogram is a convenience routine returning the normalised uniform
// LBP histogram of img: it computes uniform labels with [LBPUniform] and bins
// them with [Histogram]. The result has UniformBinCount(points) entries and is
// a ready-to-use texture feature vector.
func LBPUniformHistogram(img *cv.Mat, radius, points int) []float64 {
	labels := LBPUniform(img, radius, points)
	return Histogram(labels, UniformBinCount(points))
}

// LBPRotationInvariantUniformHistogram returns the normalised rotation-invariant
// uniform ("riu2") histogram of img, with points+2 entries. It is compact,
// rotation invariant and a standard texture descriptor.
func LBPRotationInvariantUniformHistogram(img *cv.Mat, radius, points int) []float64 {
	labels := LBPUniformRotationInvariant(img, radius, points)
	return Histogram(labels, points+2)
}
