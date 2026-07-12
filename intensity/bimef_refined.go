package intensity

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// BIMEFParams configures the refined Bio-Inspired Multi-Exposure Fusion pipeline
// [BIMEFWithParams]. The zero value is not useful; obtain a sensible starting
// point from [DefaultBIMEFParams].
type BIMEFParams struct {
	// Mu is the exponent applied to the illumination map to form the fusion
	// weight w = t^Mu; larger Mu keeps more of the original in mid-lit regions.
	Mu float64
	// Lambda is the weighted-least-squares smoothness weight: larger Lambda
	// yields a smoother, more strongly regularised illumination estimate.
	Lambda float64
	// Sharpness is the edge sensitivity (alpha) of the WLS affinities; larger
	// values make the illumination map cling more tightly to guidance edges.
	Sharpness float64
	// Iterations is the number of Gauss-Seidel sweeps used to solve the WLS
	// linear system. More sweeps converge the illumination map further.
	Iterations int
	// KMin and KMax bound the exposure ratio searched by the entropy
	// maximisation; KMin must be ≥ 1.
	KMin, KMax float64
	// EntropySteps is the number of candidate exposure ratios evaluated in
	// [KMin,KMax]; it must be ≥ 2.
	EntropySteps int
}

// DefaultBIMEFParams returns the parameters used by [BIMEFRefined]: a mild
// fusion exponent, a moderately smooth weighted-least-squares illumination map,
// and an entropy search over 40 exposure ratios in [1,7].
func DefaultBIMEFParams() BIMEFParams {
	return BIMEFParams{
		Mu:           0.5,
		Lambda:       0.5,
		Sharpness:    1.0,
		Iterations:   30,
		KMin:         1.0,
		KMax:         7.0,
		EntropySteps: 40,
	}
}

// wlsRefine solves the edge-preserving weighted-least-squares system
//
//	(I + A) t = t0,
//
// where A is the graph Laplacian whose edge weights fall off with the guidance
// gradient, so t is a smooth version of the prior t0 that still snaps to the
// structural edges of guide (all slices length rows·cols, values in [0,1]). The
// system is solved with a fixed number of in-place Gauss-Seidel sweeps, which is
// deterministic. The affinity between neighbouring pixels is
// lambda / (|Δguide|^sharpness + eps).
func wlsRefine(t0, guide []float64, rows, cols int, lambda, sharpness float64, iters int) []float64 {
	const eps = 1e-4
	// Horizontal edge weights: wx[i] couples pixel i to its left neighbour.
	wx := make([]float64, rows*cols)
	wy := make([]float64, rows*cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			i := y*cols + x
			if x > 0 {
				d := math.Abs(guide[i] - guide[i-1])
				wx[i] = lambda / (math.Pow(d, sharpness) + eps)
			}
			if y > 0 {
				d := math.Abs(guide[i] - guide[i-cols])
				wy[i] = lambda / (math.Pow(d, sharpness) + eps)
			}
		}
	}

	t := make([]float64, len(t0))
	copy(t, t0)
	for sweep := 0; sweep < iters; sweep++ {
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				i := y*cols + x
				var num, den float64
				num = t0[i]
				den = 1
				if x > 0 {
					w := wx[i]
					num += w * t[i-1]
					den += w
				}
				if x < cols-1 {
					w := wx[i+1] // left-edge weight of the right neighbour
					num += w * t[i+1]
					den += w
				}
				if y > 0 {
					w := wy[i]
					num += w * t[i-cols]
					den += w
				}
				if y < rows-1 {
					w := wy[i+cols]
					num += w * t[i+cols]
					den += w
				}
				t[i] = num / den
			}
		}
	}
	return t
}

// entropyBlurSigma is the small blur applied to the luminance before the
// exposure-ratio entropy search. Averaging neighbouring pixels turns the
// discrete integer intensities into a near-continuous signal, which is what lets
// the nonlinear re-exposure reshape the histogram (and so the entropy) — the
// role played by the down-sampling step in the original BIMEF. Without it a
// monotone re-exposure would leave a discrete histogram's entropy unchanged.
const entropyBlurSigma = 1.0

// bestExposureRatio scans EntropySteps exposure ratios in [KMin,KMax] and returns
// the one whose re-exposed luminance histogram has the greatest Shannon entropy —
// the ratio that best fills the tonal range, and so reveals the most detail.
// lumaNorm holds the (blurred, near-continuous) luminance in [0,1].
func bestExposureRatio(lumaNorm []float64, kMin, kMax float64, steps int) float64 {
	bestK := kMin
	bestE := math.Inf(-1)
	n := len(lumaNorm)
	for s := 0; s < steps; s++ {
		k := kMin + (kMax-kMin)*float64(s)/float64(steps-1)
		bright := make([]float64, n)
		for i, l := range lumaNorm {
			bright[i] = btf(l, k) * 255
		}
		e := entropy256(histFloat(bright), n)
		if e > bestE {
			bestE = e
			bestK = k
		}
	}
	return bestK
}

// BIMEFRefined enhances a low-light image with the full Bio-Inspired
// Multi-Exposure Fusion pipeline of Ying et al. (2017), using the default
// [BIMEFParams]. It is the fully realised counterpart to [BIMEF]: the
// illumination map is refined by edge-preserving weighted least squares and the
// exposure ratio is chosen by entropy maximisation rather than analytically. See
// [BIMEFWithParams] for the algorithm and parameters. The output is
// deterministic.
func BIMEFRefined(img *cv.Mat) *cv.Mat {
	return BIMEFWithParams(img, DefaultBIMEFParams())
}

// BIMEFWithParams runs the refined BIMEF pipeline with explicit parameters. It
// accepts a single- or three-channel img and returns a new [cv.Mat] of the same
// shape. The pipeline is:
//
//  1. Illumination prior. The bright-channel prior t0 (the per-pixel channel
//     maximum, normalised to [0,1]) estimates the scene illumination.
//  2. WLS refinement. t0 is smoothed by an edge-preserving weighted-least-squares
//     solve (see [wlsRefine]) guided by the image luminance, giving an
//     illumination map t that is flat within objects but respects their
//     boundaries — the paper's refinement that the simpler [BIMEF] omits.
//  3. Entropy-maximising exposure. A single virtual exposure ratio k in
//     [KMin,KMax] is selected to maximise the entropy of the re-exposed
//     luminance, so the amplification recovers the most detail without washing
//     the image out. Each channel is re-exposed through the fitted camera
//     response.
//  4. Weighted fusion. The original and re-exposed images are blended with the
//     per-pixel weight w = t^Mu, so well-lit pixels keep their value and dark
//     pixels take the brightened one.
//
// It panics on an empty image or invalid parameters (Mu, Lambda, Sharpness
// negative; Iterations < 1; KMin < 1; KMax < KMin; EntropySteps < 2). A dark,
// textured scene is markedly brightened while the ordering and edges of its
// content are preserved. The output is deterministic.
func BIMEFWithParams(img *cv.Mat, p BIMEFParams) *cv.Mat {
	requireImage(img, "BIMEFWithParams")
	if p.Mu < 0 || p.Lambda < 0 || p.Sharpness < 0 {
		panic(fmt.Sprintf("intensity: BIMEFWithParams requires non-negative Mu, Lambda, Sharpness, got %+v", p))
	}
	if p.Iterations < 1 {
		panic("intensity: BIMEFWithParams requires Iterations >= 1")
	}
	if p.KMin < 1 || p.KMax < p.KMin {
		panic(fmt.Sprintf("intensity: BIMEFWithParams requires 1 <= KMin <= KMax, got KMin=%v KMax=%v", p.KMin, p.KMax))
	}
	if p.EntropySteps < 2 {
		panic("intensity: BIMEFWithParams requires EntropySteps >= 2")
	}

	rows, cols, ch := img.Rows, img.Cols, img.Channels
	n := img.Total()

	// 1. Bright-channel prior t0 and normalised luminance guidance.
	t0 := make([]float64, n)
	guide := make([]float64, n)
	luma := lumaFloat(img)
	for pix := 0; pix < n; pix++ {
		base := pix * ch
		mx := img.Data[base]
		for c := 1; c < ch; c++ {
			if v := img.Data[base+c]; v > mx {
				mx = v
			}
		}
		t0[pix] = float64(mx) / 255
		guide[pix] = luma[pix] / 255
	}

	// 2. Edge-preserving WLS refinement of the illumination map.
	t := wlsRefine(t0, guide, rows, cols, p.Lambda, p.Sharpness, p.Iterations)

	// 3. Entropy-maximising exposure ratio. The luminance is lightly blurred
	// first so the re-exposure can reshape a near-continuous histogram.
	blurLuma := blurPlaneFloat(luma, rows, cols, entropyBlurSigma)
	for i := range blurLuma {
		blurLuma[i] /= 255
	}
	k := bestExposureRatio(blurLuma, p.KMin, p.KMax, p.EntropySteps)

	// 4. Weighted multi-exposure fusion.
	dst := cv.NewMat(rows, cols, ch)
	for pix := 0; pix < n; pix++ {
		tv := t[pix]
		if tv < 0 {
			tv = 0
		} else if tv > 1 {
			tv = 1
		}
		w := math.Pow(tv, p.Mu)
		base := pix * ch
		for c := 0; c < ch; c++ {
			orig := float64(img.Data[base+c]) / 255
			enh := btf(orig, k)
			fused := w*orig + (1-w)*enh
			dst.Data[base+c] = clampToUint8(fused*255 + 0.5)
		}
	}
	return dst
}
