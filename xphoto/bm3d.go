package xphoto

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// bm3d tuning constants for the single-step (basic) estimate.
const (
	bm3dBlock       = 4    // block edge length (blocks are bm3dBlock x bm3dBlock)
	bm3dStep        = 3    // stride between reference blocks
	bm3dSearchRad   = 8    // half-size of the block-matching search window
	bm3dMaxGroup    = 16   // maximum number of blocks stacked in a group
	bm3dLambdaThr   = 2.7  // hard-threshold factor (in units of sigma)
	bm3dMatchExtra  = 25.0 // additive slack on the block-match distance
	bm3dMatchFactor = 2.7  // multiplicative (sigma^2) term on the match distance
)

// Bm3dDenoising denoises src with a single-step (hard-threshold) BM3D filter,
// approximating cv::xphoto::bm3dDenoising's first stage (BM3D_STEP1). h is the
// assumed noise standard deviation (sigma); larger h removes more noise.
//
// The algorithm works in three phases, repeated on a grid of reference blocks:
//
//  1. Block matching: similar blocks within a local search window are found by
//     sum-of-squared difference and stacked into a 3D group.
//  2. Collaborative filtering: a separable 2D-DCT is applied to each block and a
//     1D-DCT across the stack (the collaborative transform); the resulting
//     coefficients are hard-thresholded at bm3dLambdaThr*sigma, except the group
//     DC term which is preserved so flat-region means are not disturbed; then
//     the transforms are inverted.
//  3. Aggregation: the filtered blocks are written back into accumulation
//     buffers weighted by 1/(number of retained coefficients), and the final
//     image is the weighted average.
//
// src may be single- or three-channel; each channel is denoised independently.
// See the package Deferred note: the second, empirical-Wiener BM3D stage is not
// implemented, so this is the weaker basic estimate.
func Bm3dDenoising(src *cv.Mat, h float64) *cv.Mat {
	requireNonEmpty(src, "Bm3dDenoising")
	if h <= 0 {
		h = 1
	}
	dst := cv.NewMat(src.Rows, src.Cols, src.Channels)
	for c := 0; c < src.Channels; c++ {
		bm3dChannel(src, dst, c, h)
	}
	return dst
}

// bm3dChannel runs the single-step BM3D estimate on channel c of src, writing
// the result into the same channel of dst.
func bm3dChannel(src, dst *cv.Mat, c int, sigma float64) {
	rows, cols := src.Rows, src.Cols
	B := bm3dBlock
	if rows < B || cols < B {
		// Too small to block-process; copy through.
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				dst.Set(y, x, c, src.At(y, x, c))
			}
		}
		return
	}
	// Load channel into a float plane once.
	plane := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			plane[y*cols+x] = float64(src.At(y, x, c))
		}
	}

	numAcc := make([]float64, rows*cols)
	denAcc := make([]float64, rows*cols)

	thr := bm3dLambdaThr * sigma
	matchTau := bm3dMatchFactor*sigma*sigma + bm3dMatchExtra

	cos := dctBasis(B)

	// Reference block positions on a grid; the last row/column of blocks is
	// clamped so the whole image is covered.
	for ry := 0; ry <= rows-B; ry += bm3dStep {
		ry2 := ry
		if ry2 > rows-B {
			ry2 = rows - B
		}
		for rx := 0; rx <= cols-B; rx += bm3dStep {
			rx2 := rx
			if rx2 > cols-B {
				rx2 = cols - B
			}
			processReference(plane, rows, cols, ry2, rx2, B, cos, thr, matchTau, numAcc, denAcc)
		}
	}
	// Ensure the far edges are covered even if the stride skipped them.
	if (rows-B)%bm3dStep != 0 {
		for rx := 0; rx <= cols-B; rx += bm3dStep {
			processReference(plane, rows, cols, rows-B, rx, B, cos, thr, matchTau, numAcc, denAcc)
		}
	}
	if (cols-B)%bm3dStep != 0 {
		for ry := 0; ry <= rows-B; ry += bm3dStep {
			processReference(plane, rows, cols, ry, cols-B, B, cos, thr, matchTau, numAcc, denAcc)
		}
	}

	for i := 0; i < rows*cols; i++ {
		v := plane[i]
		if denAcc[i] > 0 {
			v = numAcc[i] / denAcc[i]
		}
		y, x := i/cols, i%cols
		dst.Set(y, x, c, clampU8(v))
	}
}

// processReference performs block matching, collaborative filtering and
// aggregation for one reference block at (ry,rx).
func processReference(plane []float64, rows, cols, ry, rx, B int, cos [][]float64, thr, matchTau float64, numAcc, denAcc []float64) {
	// --- Block matching ---
	type cand struct {
		y, x int
		dist float64
	}
	var cands []cand
	y0 := ry - bm3dSearchRad
	y1 := ry + bm3dSearchRad
	x0 := rx - bm3dSearchRad
	x1 := rx + bm3dSearchRad
	for sy := y0; sy <= y1; sy++ {
		if sy < 0 || sy > rows-B {
			continue
		}
		for sx := x0; sx <= x1; sx++ {
			if sx < 0 || sx > cols-B {
				continue
			}
			d := blockDist(plane, cols, ry, rx, sy, sx, B)
			if sy == ry && sx == rx {
				cands = append(cands, cand{sy, sx, -1}) // reference always first
			} else if d <= matchTau {
				cands = append(cands, cand{sy, sx, d})
			}
		}
	}
	// Deterministically keep the closest bm3dMaxGroup blocks.
	sort.SliceStable(cands, func(i, j int) bool {
		if cands[i].dist != cands[j].dist {
			return cands[i].dist < cands[j].dist
		}
		if cands[i].y != cands[j].y {
			return cands[i].y < cands[j].y
		}
		return cands[i].x < cands[j].x
	})
	if len(cands) > bm3dMaxGroup {
		cands = cands[:bm3dMaxGroup]
	}
	g := len(cands)

	// --- Collaborative transform ---
	// group[b] holds the 2D-DCT coefficients of block b, length B*B.
	group := make([][]float64, g)
	for b := 0; b < g; b++ {
		block := make([]float64, B*B)
		for i := 0; i < B; i++ {
			for j := 0; j < B; j++ {
				block[i*B+j] = plane[(cands[b].y+i)*cols+(cands[b].x+j)]
			}
		}
		group[b] = dct2d(block, B, cos)
	}
	// 1D-DCT across the group for every coefficient position, then hard-
	// threshold, then invert the 1D transform.
	col := make([]float64, g)
	retained := 0
	for pos := 0; pos < B*B; pos++ {
		for b := 0; b < g; b++ {
			col[b] = group[b][pos]
		}
		coef := dct1d(col, g)
		for w := 0; w < g; w++ {
			// Preserve the overall DC term (pos 0, w 0) so the mean of a flat
			// region survives; hard-threshold everything else.
			if pos == 0 && w == 0 {
				retained++
				continue
			}
			if math.Abs(coef[w]) < thr {
				coef[w] = 0
			} else {
				retained++
			}
		}
		inv := idct1d(coef, g)
		for b := 0; b < g; b++ {
			group[b][pos] = inv[b]
		}
	}
	// Confidence weight: fewer retained coefficients -> more reliable estimate.
	weight := 1.0
	if retained > 0 {
		weight = 1.0 / float64(retained)
	}

	// --- Inverse 2D-DCT and aggregation ---
	for b := 0; b < g; b++ {
		block := idct2d(group[b], B, cos)
		by, bx := cands[b].y, cands[b].x
		for i := 0; i < B; i++ {
			for j := 0; j < B; j++ {
				idx := (by+i)*cols + (bx + j)
				numAcc[idx] += weight * block[i*B+j]
				denAcc[idx] += weight
			}
		}
	}
}

// blockDist returns the mean squared difference between two BxB blocks.
func blockDist(plane []float64, cols, ay, ax, by, bx, B int) float64 {
	var s float64
	for i := 0; i < B; i++ {
		for j := 0; j < B; j++ {
			d := plane[(ay+i)*cols+(ax+j)] - plane[(by+i)*cols+(bx+j)]
			s += d * d
		}
	}
	return s / float64(B*B)
}

// dctBasis precomputes the orthonormal DCT-II cosine matrix of size n: the
// entry cos[k][m] is the weight of input sample m in output coefficient k.
func dctBasis(n int) [][]float64 {
	c := make([][]float64, n)
	for k := 0; k < n; k++ {
		row := make([]float64, n)
		var ck float64
		if k == 0 {
			ck = math.Sqrt(1.0 / float64(n))
		} else {
			ck = math.Sqrt(2.0 / float64(n))
		}
		for m := 0; m < n; m++ {
			row[m] = ck * math.Cos(math.Pi*(2*float64(m)+1)*float64(k)/(2*float64(n)))
		}
		c[k] = row
	}
	return c
}

// dct1d applies the orthonormal DCT-II to the first n elements of vec, using a
// freshly derived basis. It is used across the group where n = group size.
func dct1d(vec []float64, n int) []float64 {
	basis := dctBasis(n)
	out := make([]float64, n)
	for k := 0; k < n; k++ {
		var s float64
		bk := basis[k]
		for m := 0; m < n; m++ {
			s += bk[m] * vec[m]
		}
		out[k] = s
	}
	return out
}

// idct1d inverts dct1d (applies the DCT-III / transpose of the DCT-II basis).
func idct1d(coef []float64, n int) []float64 {
	basis := dctBasis(n)
	out := make([]float64, n)
	for m := 0; m < n; m++ {
		var s float64
		for k := 0; k < n; k++ {
			s += basis[k][m] * coef[k]
		}
		out[m] = s
	}
	return out
}

// dct2d applies a separable orthonormal 2D-DCT to a BxB block using the
// precomputed cosine matrix.
func dct2d(block []float64, B int, cos [][]float64) []float64 {
	tmp := make([]float64, B*B)
	// Rows.
	for i := 0; i < B; i++ {
		for k := 0; k < B; k++ {
			var s float64
			ck := cos[k]
			for j := 0; j < B; j++ {
				s += ck[j] * block[i*B+j]
			}
			tmp[i*B+k] = s
		}
	}
	out := make([]float64, B*B)
	// Columns.
	for j := 0; j < B; j++ {
		for k := 0; k < B; k++ {
			var s float64
			ck := cos[k]
			for i := 0; i < B; i++ {
				s += ck[i] * tmp[i*B+j]
			}
			out[k*B+j] = s
		}
	}
	return out
}

// idct2d inverts dct2d.
func idct2d(coef []float64, B int, cos [][]float64) []float64 {
	tmp := make([]float64, B*B)
	// Inverse over columns.
	for j := 0; j < B; j++ {
		for i := 0; i < B; i++ {
			var s float64
			for k := 0; k < B; k++ {
				s += cos[k][i] * coef[k*B+j]
			}
			tmp[i*B+j] = s
		}
	}
	out := make([]float64, B*B)
	// Inverse over rows.
	for i := 0; i < B; i++ {
		for j := 0; j < B; j++ {
			var s float64
			for k := 0; k < B; k++ {
				s += cos[k][j] * tmp[i*B+k]
			}
			out[i*B+j] = s
		}
	}
	return out
}
