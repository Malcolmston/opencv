package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// FastGlobalSmootherFilter applies the fast global smoother of Min, Choi, Lu, Xu
// and Wang ("Fast Global Image Smoothing Based on Weighted Least Squares", 2014)
// to src, guided by guide, and returns a new Mat shaped like src.
//
// It approximates the weighted-least-squares energy
//
//	Σ_p (u_p − f_p)² + lambda · Σ_p Σ_{q∈N(p)} w_{p,q}·(u_p − u_q)²
//
// whose minimiser is a smoothed image u that stays close to the input f while
// penalising differences across low-weight neighbours. The affinity weight
//
//	w_{p,q} = exp(−‖guide_p − guide_q‖₁ / sigmaColor)
//
// is small across guide edges, so those edges are preserved. Rather than solving
// the full 2-D system, the method performs, for each of iters passes, exact 1-D
// weighted-least-squares solves along every row and then every column using the
// tridiagonal Thomas algorithm — an O(N) separable approximation that converges
// in a handful of passes.
//
// lambda is the smoothing strength (larger ⇒ smoother); sigmaColor is the
// edge-stopping scale on the native [0,255] guide range; iters is the number of
// horizontal+vertical sweeps (three is typical, and must be ≥ 1). guide and src
// must share width and height; guide may be 1- or 3-channel. If guide is nil,
// src guides itself. It panics on a size mismatch, non-positive lambda/sigmaColor
// or iters < 1. The filter is deterministic.
func FastGlobalSmootherFilter(src, guide *cv.Mat, lambda, sigmaColor float64, iters int) *cv.Mat {
	if guide == nil {
		guide = src
	}
	if src.Rows != guide.Rows || src.Cols != guide.Cols {
		panic("ximgproc: FastGlobalSmootherFilter src and guide must share dimensions")
	}
	if lambda <= 0 || sigmaColor <= 0 {
		panic("ximgproc: FastGlobalSmootherFilter requires positive lambda and sigmaColor")
	}
	if iters < 1 {
		panic("ximgproc: FastGlobalSmootherFilter requires iters >= 1")
	}
	rows, cols := src.Rows, src.Cols
	gp := planesFromMat(guide)

	// Horizontal affinity wH[y*cols+x] links (x-1,y)-(x,y); vertical wV links
	// (x,y-1)-(x,y). Column 0 / row 0 entries are unused.
	wH := make([]float64, rows*cols)
	wV := make([]float64, rows*cols)
	invSig := 1.0 / sigmaColor
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if x > 0 {
				var s float64
				for c := range gp {
					s += math.Abs(gp[c][i] - gp[c][i-1])
				}
				wH[i] = math.Exp(-s * invSig)
			}
			if y > 0 {
				var s float64
				for c := range gp {
					s += math.Abs(gp[c][i] - gp[c][i-cols])
				}
				wV[i] = math.Exp(-s * invSig)
			}
		}
	}

	planes := planesFromMat(src)

	// Scratch buffers for the Thomas solver, sized to the longer dimension.
	maxLen := cols
	if rows > cols {
		maxLen = rows
	}
	a := make([]float64, maxLen)
	b := make([]float64, maxLen)
	cc := make([]float64, maxLen)
	rhs := make([]float64, maxLen)

	for it := 0; it < iters; it++ {
		// Per-iteration lambda schedule from the paper.
		lam := lambda * 1.5 * math.Exp2(float64(2*(iters-it-1))) /
			(math.Exp2(float64(2*iters)) - 1)
		for _, p := range planes {
			// Horizontal solves (rows).
			for y := 0; y < rows; y++ {
				base := y * cols
				for x := 0; x < cols; x++ {
					i := base + x
					var lo, hi float64
					if x > 0 {
						lo = wH[i]
					}
					if x < cols-1 {
						hi = wH[i+1]
					}
					a[x] = -lam * lo
					cc[x] = -lam * hi
					b[x] = 1 + lam*(lo+hi)
					rhs[x] = p[i]
				}
				thomas(a[:cols], b[:cols], cc[:cols], rhs[:cols])
				for x := 0; x < cols; x++ {
					p[base+x] = rhs[x]
				}
			}
			// Vertical solves (columns).
			for x := 0; x < cols; x++ {
				for y := 0; y < rows; y++ {
					i := y*cols + x
					var lo, hi float64
					if y > 0 {
						lo = wV[i]
					}
					if y < rows-1 {
						hi = wV[i+cols]
					}
					a[y] = -lam * lo
					cc[y] = -lam * hi
					b[y] = 1 + lam*(lo+hi)
					rhs[y] = p[i]
				}
				thomas(a[:rows], b[:rows], cc[:rows], rhs[:rows])
				for y := 0; y < rows; y++ {
					p[y*cols+x] = rhs[y]
				}
			}
		}
	}
	return matFromPlanes(planes, rows, cols)
}

// thomas solves the tridiagonal system with sub-diagonal a, diagonal b,
// super-diagonal c and right-hand side d in place: on return d holds the
// solution. a[0] and c[n-1] are ignored. The systems here are diagonally
// dominant, so no pivoting is needed.
func thomas(a, b, c, d []float64) {
	n := len(b)
	if n == 0 {
		return
	}
	cp := make([]float64, n)
	dp := make([]float64, n)
	cp[0] = c[0] / b[0]
	dp[0] = d[0] / b[0]
	for i := 1; i < n; i++ {
		m := b[i] - a[i]*cp[i-1]
		cp[i] = c[i] / m
		dp[i] = (d[i] - a[i]*dp[i-1]) / m
	}
	d[n-1] = dp[n-1]
	for i := n - 2; i >= 0; i-- {
		d[i] = dp[i] - cp[i]*d[i+1]
	}
}
