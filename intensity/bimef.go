package intensity

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Constants of the fitted camera response (brightness-transfer) function used by
// BIMEF, from Ying et al. (2017). The response g(i,k) = exp(b·(1−k^a))·i^(k^a)
// maps a normalised intensity i in [0,1] to the value it would take under an
// exposure ratio k.
const (
	bimefA  = -0.3293 // exponent of the exposure ratio
	bimefB  = 1.1258  // brightness offset
	bimefMu = 0.5     // fusion-weight exponent applied to the illumination map
)

// bimefKMin and bimefKMax bound the estimated exposure ratio so a very dark
// scene cannot request an unbounded amount of amplification.
const (
	bimefKMin = 1.0
	bimefKMax = 7.0
)

// btf evaluates the brightness-transfer function g(i,k) for a normalised
// intensity i in [0,1] and exposure ratio k >= 1. The result is the intensity
// after virtually re-exposing the pixel by the factor k.
func btf(i, k float64) float64 {
	gamma := math.Pow(k, bimefA)
	beta := math.Exp((1 - gamma) * bimefB)
	return beta * math.Pow(i, gamma)
}

// BIMEF enhances a low-light image with a Bio-Inspired Multi-Exposure Fusion
// pipeline after Ying et al. (2017), "A Bio-Inspired Multi-Exposure Fusion
// Framework for Low-light Image Enhancement". It accepts a single- or
// three-channel img and returns a new Mat of the same shape.
//
// The pipeline is:
//
//  1. Illumination estimate. A per-pixel scene-illumination map t is taken as
//     the maximum of the (normalised) channels — the bright-channel prior —
//     giving t in [0,1]; dark pixels have small t.
//  2. Exposure ratio. A single virtual exposure ratio k in [1,7] is chosen from
//     the mean illumination of the under-exposed region (pixels with t < 0.5),
//     so darker scenes are amplified more. Each channel is then re-exposed
//     through the fitted camera response [btf], producing a synthetic
//     well-exposed image.
//  3. Weighted fusion. The original and re-exposed images are blended per pixel
//     with weight w = t^mu (mu = 0.5): well-lit pixels (large t) keep their
//     original value while dark pixels take mostly the brightened value, so
//     detail is recovered in shadows without washing out highlights.
//
// # Approximation and deferred work
//
// This is a faithful but simplified realisation of the framework. The
// illumination map is the raw bright-channel prior rather than the
// edge-preserving, weighted-least-squares–refined map of the paper, and the
// exposure ratio is derived analytically from the dark-region mean rather than
// by the entropy-maximising optimisation the paper performs. Full illumination
// refinement and exposure-ratio / fusion-weight optimisation are deferred. The
// output is deterministic.
func BIMEF(img *cv.Mat) *cv.Mat {
	requireImage(img, "BIMEF")
	ch := img.Channels
	n := img.Total()

	// Per-pixel illumination map t (bright-channel prior) and the mean
	// illumination of the under-exposed region.
	t := make([]float64, n)
	var darkSum float64
	var darkCount int
	for p := 0; p < n; p++ {
		base := p * ch
		mx := img.Data[base]
		for c := 1; c < ch; c++ {
			if v := img.Data[base+c]; v > mx {
				mx = v
			}
		}
		tv := float64(mx) / 255
		t[p] = tv
		if tv < 0.5 {
			darkSum += tv
			darkCount++
		}
	}

	// Exposure ratio: brighten more when the dark region is darker. With no
	// under-exposed region the image needs no amplification.
	k := 1.0
	if darkCount > 0 {
		k = 1 / (darkSum/float64(darkCount) + 1e-3)
		if k < bimefKMin {
			k = bimefKMin
		} else if k > bimefKMax {
			k = bimefKMax
		}
	}

	dst := cv.NewMat(img.Rows, img.Cols, ch)
	for p := 0; p < n; p++ {
		w := math.Pow(t[p], bimefMu) // fusion weight toward the original
		base := p * ch
		for c := 0; c < ch; c++ {
			orig := float64(img.Data[base+c]) / 255
			enh := btf(orig, k)
			fused := w*orig + (1-w)*enh
			dst.Data[base+c] = clampToUint8(fused*255 + 0.5)
		}
	}
	return dst
}
