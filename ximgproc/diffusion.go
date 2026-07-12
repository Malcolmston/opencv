package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// AnisotropicDiffusion applies Perona–Malik anisotropic diffusion to img and
// returns a new Mat of the same shape. It is an iterative edge-preserving
// smoother: each step nudges every pixel toward its four neighbours, but the
// conduction coefficient shrinks across strong gradients so that edges resist
// blurring while flat regions are averaged.
//
// The update per iteration, applied independently to each channel, is
//
//	I ← I + alpha·Σ_{d∈{N,S,E,W}} c(∇_d I)·∇_d I,
//	c(∇) = exp(−(|∇|/k)²),
//
// where ∇_d I is the intensity difference to the neighbour in direction d.
//
// alpha is the integration rate; for the 4-neighbour stencil it must satisfy
// 0 < alpha ≤ 0.25 for numerical stability. k is the conduction gradient
// threshold on the native [0,255] intensity scale: gradients well below k are
// smoothed, gradients well above k (edges) are preserved. iters is the number
// of iterations (a value < 1 returns a copy of img). Borders use edge
// replication so the outward gradient there is zero. It panics if alpha or k is
// non-positive.
func AnisotropicDiffusion(img *cv.Mat, alpha, k float64, iters int) *cv.Mat {
	if alpha <= 0 {
		panic("ximgproc: AnisotropicDiffusion requires alpha > 0")
	}
	if k <= 0 {
		panic("ximgproc: AnisotropicDiffusion requires k > 0")
	}
	rows, cols, ch := img.Rows, img.Cols, img.Channels

	if iters < 1 {
		return img.Clone()
	}

	// Per-channel float working buffers.
	planes := make([][]float64, ch)
	for c := 0; c < ch; c++ {
		planes[c] = make([]float64, rows*cols)
		for i := 0; i < rows*cols; i++ {
			planes[c][i] = float64(img.Data[i*ch+c])
		}
	}

	invK2 := 1.0 / (k * k)
	cond := func(grad float64) float64 {
		return math.Exp(-grad * grad * invK2)
	}

	next := make([]float64, rows*cols)
	for c := 0; c < ch; c++ {
		cur := planes[c]
		for it := 0; it < iters; it++ {
			for y := 0; y < rows; y++ {
				for x := 0; x < cols; x++ {
					i := y*cols + x
					center := cur[i]
					north := center
					if y > 0 {
						north = cur[i-cols]
					}
					south := center
					if y < rows-1 {
						south = cur[i+cols]
					}
					west := center
					if x > 0 {
						west = cur[i-1]
					}
					east := center
					if x < cols-1 {
						east = cur[i+1]
					}
					gn := north - center
					gs := south - center
					gw := west - center
					ge := east - center
					next[i] = center + alpha*(cond(gn)*gn+cond(gs)*gs+cond(gw)*gw+cond(ge)*ge)
				}
			}
			cur, next = next, cur
		}
		planes[c] = cur
	}

	out := cv.NewMat(rows, cols, ch)
	for c := 0; c < ch; c++ {
		for i := 0; i < rows*cols; i++ {
			out.Data[i*ch+c] = clampU8(planes[c][i])
		}
	}
	return out
}
