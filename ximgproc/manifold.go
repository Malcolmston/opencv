package ximgproc

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// AdaptiveManifoldFilter applies an adaptive-manifold edge-preserving filter to
// src and returns a new Mat of the same shape. It is a fast high-dimensional
// Gaussian filter in the spirit of Gastal and Oliveira ("Adaptive Manifolds for
// Real-Time High-Dimensional Filtering", 2012): instead of blurring in the full
// 5-D space-plus-colour domain, it evaluates a small set of adaptive manifolds —
// low-dimensional sheets that follow the image content — and blends them.
//
// The manifolds are built as a binary tree. The root is a spatial low-pass of
// the guidance (luma) image; at each level every manifold is split into two by
// the sign of its residual against the guidance, and each child is re-estimated
// as a range-weighted spatial low-pass. For every manifold η the filter splats
// src weighted by the range kernel exp(−(luma−η)²/2σ_r²), blurs the splatted
// signal and the weights spatially with σ_s, and gathers the normalised result
// back to each pixel weighted again by its affinity to η. Summed over the tree
// and normalised, this yields an edge-preserving average whose cost is
// independent of σ_r.
//
// sigmaSpace (σ_s) is the spatial smoothing radius in pixels; sigmaColor (σ_r)
// is the range/edge scale on the native [0,255] luma range. src may be 1- or
// 3-channel. It panics on non-positive sigmas. The number of manifolds is chosen
// automatically from σ_r (fewer for large σ_r). The filter is deterministic.
func AdaptiveManifoldFilter(src *cv.Mat, sigmaSpace, sigmaColor float64) *cv.Mat {
	if sigmaSpace <= 0 || sigmaColor <= 0 {
		panic("ximgproc: AdaptiveManifoldFilter requires positive sigmas")
	}
	rows, cols := src.Rows, src.Cols
	n := rows * cols
	ch := src.Channels

	luma := channelPlane(toGray(src), 0)
	srcP := planesFromMat(src)

	// Tree depth: more manifolds when sigmaColor is small (sharper range kernel).
	depth := int(math.Round(math.Log2(255.0/sigmaColor))) + 1
	if depth < 1 {
		depth = 1
	}
	if depth > 4 {
		depth = 4
	}

	numAcc := make([][]float64, ch) // per-channel gathered numerator
	for c := range numAcc {
		numAcc[c] = make([]float64, n)
	}
	denAcc := make([]float64, n) // gathered denominator

	invR2 := 1.0 / (2 * sigmaColor * sigmaColor)

	// Recursive manifold processing. mask (nil == all pixels) restricts which
	// pixels define this manifold's low-pass.
	var process func(eta []float64, level int)
	process = func(eta []float64, level int) {
		// Range affinity of each pixel to this manifold.
		w := make([]float64, n)
		for i := 0; i < n; i++ {
			d := luma[i] - eta[i]
			w[i] = math.Exp(-d * d * invR2)
		}
		blurW := gaussianBlurFloat(w, rows, cols, sigmaSpace)
		// Splat + blur each channel, then gather.
		for c := 0; c < ch; c++ {
			splat := make([]float64, n)
			for i := 0; i < n; i++ {
				splat[i] = w[i] * srcP[c][i]
			}
			blurS := gaussianBlurFloat(splat, rows, cols, sigmaSpace)
			for i := 0; i < n; i++ {
				if blurW[i] > 1e-12 {
					numAcc[c][i] += w[i] * (blurS[i] / blurW[i])
				}
			}
		}
		for i := 0; i < n; i++ {
			denAcc[i] += w[i]
		}

		if level >= depth {
			return
		}
		// Blurred manifold value of the guidance, used to split residuals.
		splL := make([]float64, n)
		for i := 0; i < n; i++ {
			splL[i] = w[i] * luma[i]
		}
		blurL := gaussianBlurFloat(splL, rows, cols, sigmaSpace)
		etaVal := make([]float64, n)
		for i := 0; i < n; i++ {
			if blurW[i] > 1e-12 {
				etaVal[i] = blurL[i] / blurW[i]
			} else {
				etaVal[i] = eta[i]
			}
		}
		// Split by residual sign and re-estimate a manifold on each side.
		etaPlus := maskedLowpass(luma, etaVal, rows, cols, sigmaSpace, true)
		etaMinus := maskedLowpass(luma, etaVal, rows, cols, sigmaSpace, false)
		process(etaPlus, level+1)
		process(etaMinus, level+1)
	}

	root := gaussianBlurFloat(luma, rows, cols, sigmaSpace)
	process(root, 1)

	out := make([][]float64, ch)
	for c := 0; c < ch; c++ {
		out[c] = make([]float64, n)
		for i := 0; i < n; i++ {
			if denAcc[i] > 1e-12 {
				out[c][i] = numAcc[c][i] / denAcc[i]
			} else {
				out[c][i] = srcP[c][i]
			}
		}
	}
	return matFromPlanes(out, rows, cols)
}

// maskedLowpass returns a spatial low-pass of luma computed only over the pixels
// whose residual (luma − etaVal) has the requested sign, with the rest gated to
// zero and re-normalised by the blurred mask. It is used to grow the two child
// manifolds around the current level.
func maskedLowpass(luma, etaVal []float64, rows, cols int, sigma float64, positive bool) []float64 {
	n := rows * cols
	mask := make([]float64, n)
	masked := make([]float64, n)
	for i := 0; i < n; i++ {
		r := luma[i] - etaVal[i]
		in := r >= 0
		if !positive {
			in = r < 0
		}
		if in {
			mask[i] = 1
			masked[i] = luma[i]
		}
	}
	bm := gaussianBlurFloat(mask, rows, cols, sigma)
	bx := gaussianBlurFloat(masked, rows, cols, sigma)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		if bm[i] > 1e-6 {
			out[i] = bx[i] / bm[i]
		} else {
			out[i] = etaVal[i]
		}
	}
	return out
}
