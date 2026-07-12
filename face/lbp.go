package face

import (
	cv "github.com/malcolmston/opencv"
)

// lbpOffsets and lbpWeights define the classic 3×3, 8-neighbour Local Binary
// Pattern sampling used by [LBP] and [LBPUniform]. The eight neighbours are
// visited clockwise starting at the top-left corner, and each contributes its
// weight to the code when its value is greater than or equal to the centre
// pixel:
//
//	  1    2    4
//	128    C    8
//	 64   32   16
//
// so, for example, a code of 1 means only the top-left neighbour met the
// threshold. This weighting is the standard Ojala et al. convention; it is
// fixed and documented so that hand-computed codes match exactly.
var (
	lbpOffsets = [8][2]int{
		{-1, -1}, {-1, 0}, {-1, 1}, // top-left, top, top-right
		{0, 1},                  // right
		{1, 1}, {1, 0}, {1, -1}, // bottom-right, bottom, bottom-left
		{0, -1}, // left
	}
	lbpWeights = [8]int{1, 2, 4, 8, 16, 32, 64, 128}
)

// LBP computes the basic 3×3 Local Binary Pattern code image of img. The image
// is first reduced to luma; each interior pixel is compared against its eight
// neighbours (see the package-level weighting) to form an 8-bit code in
// [0,255]. Because the pattern is undefined on the outer border, the result is a
// single-channel Mat of size (Rows−2)×(Cols−2): output pixel (y,x) holds the
// code of input pixel (y+1,x+1). In particular the LBP of a 3×3 image is a 1×1
// Mat containing that single code.
//
// LBP is illumination-robust by construction: adding a constant to every pixel
// preserves the neighbour ordering and therefore leaves the codes unchanged.
// It panics if img is smaller than 3×3.
func LBP(img *cv.Mat) *cv.Mat {
	return lbpImage(img, false)
}

// LBPUniform computes the uniform-pattern LBP label image of img. It is
// identical to [LBP] except that each 8-bit code is mapped to a compact label:
// the 58 "uniform" patterns (those with at most two 0↔1 transitions in their
// circular bit string) receive distinct labels 0–57, and every non-uniform
// pattern collapses to label 58, for 59 labels total. Uniform patterns capture
// the fundamental local textures (edges, corners, spots) and yield much shorter
// histograms. The result is a single-channel Mat of size (Rows−2)×(Cols−2). It
// panics if img is smaller than 3×3.
func LBPUniform(img *cv.Mat) *cv.Mat {
	return lbpImage(img, true)
}

// lbpImage is the shared kernel for LBP and LBPUniform.
func lbpImage(img *cv.Mat, uniform bool) *cv.Mat {
	g := toGrayMat(img)
	if g.Rows < 3 || g.Cols < 3 {
		panic("face: LBP requires an image of at least 3x3")
	}
	outRows := g.Rows - 2
	outCols := g.Cols - 2
	out := cv.NewMat(outRows, outCols, 1)
	cols := g.Cols
	for y := 1; y < g.Rows-1; y++ {
		for x := 1; x < g.Cols-1; x++ {
			center := g.Data[y*cols+x]
			code := 0
			for k := 0; k < 8; k++ {
				ny := y + lbpOffsets[k][0]
				nx := x + lbpOffsets[k][1]
				if g.Data[ny*cols+nx] >= center {
					code |= lbpWeights[k]
				}
			}
			if uniform {
				code = lbpUniformMap[code]
			}
			out.Data[(y-1)*outCols+(x-1)] = uint8(code)
		}
	}
	return out
}

// lbpUniformMap maps each of the 256 possible 8-bit LBP codes to its uniform
// label in [0,58]. It is built once at package initialisation.
var lbpUniformMap = buildUniformMap()

// buildUniformMap assigns sequential labels 0–57 to the uniform patterns (in
// increasing code order) and label 58 to every non-uniform pattern.
func buildUniformMap() [256]int {
	var table [256]int
	next := 0
	for code := 0; code < 256; code++ {
		if bitTransitions(code) <= 2 {
			table[code] = next
			next++
		} else {
			table[code] = 58
		}
	}
	return table
}

// bitTransitions counts the number of 0↔1 changes in the circular 8-bit
// representation of code (bit 7 wraps around to bit 0).
func bitTransitions(code int) int {
	count := 0
	for i := 0; i < 8; i++ {
		a := (code >> i) & 1
		b := (code >> ((i + 1) % 8)) & 1
		if a != b {
			count++
		}
	}
	return count
}
