package xphoto

import (
	cv "github.com/malcolmston/opencv"
)

// FSRMode selects the quality/speed trade-off of [InpaintFSR], mirroring
// OpenCV's INPAINT_FSR_BEST and INPAINT_FSR_FAST inpainting flags.
type FSRMode int

const (
	// FSRBest favours reconstruction quality: it uses a larger support border
	// and more model-building iterations per block. This is OpenCV's
	// INPAINT_FSR_BEST.
	FSRBest FSRMode = iota
	// FSRFast favours speed with a smaller border and fewer iterations,
	// mirroring OpenCV's INPAINT_FSR_FAST.
	FSRFast
)

// fsrGamma is the orthogonality-deficiency compensation factor: only this
// fraction of each selected atom's least-squares coefficient is added to the
// model per iteration, which keeps the greedy expansion stable (the classic FSR
// value is 0.5).
const fsrGamma = 0.5

// InpaintFSR fills the masked region of src by Frequency-Selective
// Reconstruction, porting the FSR variant of cv::xphoto::inpaint
// (INPAINT_FSR_BEST / INPAINT_FSR_FAST). The image is processed in overlapping
// blocks; within each block the known samples are modelled as a sparse
// superposition of 2D-DCT basis functions, built one atom at a time by weighted
// matching pursuit (each iteration adds the basis that best explains the
// weighted residual over the known pixels). The converged model is then
// evaluated at the unknown pixels, so missing content is extrapolated from the
// local spatial-frequency structure rather than copied, which reconstructs
// smooth gradients and periodic texture well.
//
// mask must be single-channel and the same size as src; every non-zero mask
// pixel is treated as unknown. src may be single- or three-channel. mode trades
// quality for speed. Any pixels left unresolved by blocks that lacked support
// are finished with a boundary-mean diffusion pass so the result is always
// fully defined. src and mask are not modified.
func InpaintFSR(src, mask *cv.Mat, mode FSRMode) *cv.Mat {
	requireNonEmpty(src, "InpaintFSR")
	requireNonEmpty(mask, "InpaintFSR")
	requireChannels(mask, 1, "InpaintFSR mask")
	requireSameSize(src, mask, "InpaintFSR")

	rows, cols, ch := src.Rows, src.Cols, src.Channels
	out := src.Clone()

	known := make([]bool, rows*cols)
	unknown := 0
	for i := 0; i < rows*cols; i++ {
		if mask.Data[i] == 0 {
			known[i] = true
		} else {
			unknown++
		}
	}
	if unknown == 0 {
		return out
	}

	blk, border, iters := 8, 8, 40
	if mode == FSRFast {
		blk, border, iters = 8, 4, 18
	}

	// Float planes per channel for accumulation.
	planes := make([][]float64, ch)
	for c := 0; c < ch; c++ {
		p := make([]float64, rows*cols)
		for i := 0; i < rows*cols; i++ {
			p[i] = float64(out.Data[i*ch+c])
		}
		planes[c] = p
	}

	filled := make([]bool, rows*cols)
	copy(filled, known)

	for by := 0; by < rows; by += blk {
		for bx := 0; bx < cols; bx += blk {
			// Does this block contain any unknown pixel?
			hasUnknown := false
			for y := by; y < by+blk && y < rows && !hasUnknown; y++ {
				for x := bx; x < bx+blk && x < cols; x++ {
					if !known[y*cols+x] {
						hasUnknown = true
						break
					}
				}
			}
			if !hasUnknown {
				continue
			}
			fsrProcessBlock(planes, known, filled, out, rows, cols, ch, by, bx, blk, border, iters)
		}
	}

	// Diffusion fallback for any pixel a block could not resolve (no support).
	fsrDiffusionFill(out, filled, rows, cols, ch)
	return out
}

// fsrProcessBlock reconstructs the unknown pixels of the central block at
// (by,bx) from the known samples in the surrounding window.
func fsrProcessBlock(planes [][]float64, known, filled []bool, out *cv.Mat, rows, cols, ch, by, bx, blk, border, iters int) {
	// Window bounds (clamped to the image).
	wy0 := by - border
	wx0 := bx - border
	wy1 := by + blk + border
	wx1 := bx + blk + border
	if wy0 < 0 {
		wy0 = 0
	}
	if wx0 < 0 {
		wx0 = 0
	}
	if wy1 > rows {
		wy1 = rows
	}
	if wx1 > cols {
		wx1 = cols
	}
	H := wy1 - wy0
	W := wx1 - wx0
	if H < 2 || W < 2 {
		return
	}

	// Binary support weight and count of known samples in the window.
	weight := make([]float64, H*W)
	nKnown := 0
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			if known[(wy0+y)*cols+(wx0+x)] {
				weight[y*W+x] = 1
				nKnown++
			}
		}
	}
	if nKnown < 4 {
		return // not enough support; leave for the diffusion fallback
	}

	basisH := dctBasis(H)
	basisW := dctBasis(W)

	// den[u*W+v] = sum weight * phi^2 depends only on the weight and is shared
	// across channels; precompute it once.
	// wRow2[y*W+v] = sum_x weight[y,x] * cosW[v][x]^2
	wRow2 := make([]float64, H*W)
	for y := 0; y < H; y++ {
		for v := 0; v < W; v++ {
			var s float64
			bv := basisW[v]
			for x := 0; x < W; x++ {
				w := weight[y*W+x]
				if w != 0 {
					s += w * bv[x] * bv[x]
				}
			}
			wRow2[y*W+v] = s
		}
	}
	den := make([]float64, H*W)
	for u := 0; u < H; u++ {
		bu := basisH[u]
		for v := 0; v < W; v++ {
			var s float64
			for y := 0; y < H; y++ {
				s += bu[y] * bu[y] * wRow2[y*W+v]
			}
			den[u*W+v] = s
		}
	}

	// Reconstruct each channel with weighted matching pursuit.
	rowT := make([]float64, H*W) // per-row DCT of the weighted residual
	num := make([]float64, H*W)
	for c := 0; c < ch; c++ {
		plane := planes[c]
		// Weighted residual r = weight * (value where known, else 0).
		resid := make([]float64, H*W)
		for y := 0; y < H; y++ {
			for x := 0; x < W; x++ {
				if weight[y*W+x] != 0 {
					resid[y*W+x] = plane[(wy0+y)*cols+(wx0+x)]
				}
			}
		}
		model := make([]float64, H*W)

		for it := 0; it < iters; it++ {
			// rowT[y*W+v] = sum_x resid[y,x] * cosW[v][x]
			for y := 0; y < H; y++ {
				base := y * W
				for v := 0; v < W; v++ {
					var s float64
					bv := basisW[v]
					for x := 0; x < W; x++ {
						s += resid[base+x] * bv[x]
					}
					rowT[base+v] = s
				}
			}
			// num[u*W+v] = sum_y cosH[u][y] * rowT[y,v]; pick the best atom.
			bestU, bestV := 0, 0
			bestScore := -1.0
			var bestNum float64
			for u := 0; u < H; u++ {
				bu := basisH[u]
				for v := 0; v < W; v++ {
					var s float64
					for y := 0; y < H; y++ {
						s += bu[y] * rowT[y*W+v]
					}
					num[u*W+v] = s
					d := den[u*W+v]
					if d <= 1e-12 {
						continue
					}
					score := s * s / d
					if score > bestScore {
						bestScore = score
						bestU, bestV = u, v
						bestNum = s
					}
				}
			}
			if bestScore <= 1e-12 {
				break
			}
			coef := fsrGamma * bestNum / den[bestU*W+bestV]
			bu := basisH[bestU]
			bv := basisW[bestV]
			// Update the model and the weighted residual.
			for y := 0; y < H; y++ {
				phiY := bu[y]
				for x := 0; x < W; x++ {
					phi := phiY * bv[x]
					model[y*W+x] += coef * phi
					if weight[y*W+x] != 0 {
						resid[y*W+x] -= coef * phi
					}
				}
			}
		}

		// Write the model into the unknown pixels of the central block only.
		for y := by; y < by+blk && y < rows; y++ {
			for x := bx; x < bx+blk && x < cols; x++ {
				if known[y*cols+x] {
					continue
				}
				val := model[(y-wy0)*W+(x-wx0)]
				out.Data[(y*cols+x)*ch+c] = clampU8(val)
				plane[y*cols+x] = val
			}
		}
	}
	// Mark the central block's unknown pixels as filled.
	for y := by; y < by+blk && y < rows; y++ {
		for x := bx; x < bx+blk && x < cols; x++ {
			if !known[y*cols+x] {
				filled[y*cols+x] = true
			}
		}
	}
}

// fsrDiffusionFill resolves any pixel not yet filled by propagating the mean of
// its filled 8-neighbours inward until none remain, guaranteeing termination.
func fsrDiffusionFill(out *cv.Mat, filled []bool, rows, cols, ch int) {
	remaining := 0
	for i := range filled {
		if !filled[i] {
			remaining++
		}
	}
	for remaining > 0 {
		type fill struct {
			idx int
			val []uint8
		}
		var layer []fill
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				idx := y*cols + x
				if filled[idx] {
					continue
				}
				val := neighbourMean(out, filled, rows, cols, y, x, ch)
				layer = append(layer, fill{idx, val})
			}
		}
		if len(layer) == 0 {
			break
		}
		for _, f := range layer {
			copy(out.Data[f.idx*ch:f.idx*ch+ch], f.val)
			filled[f.idx] = true
			remaining--
		}
	}
}
