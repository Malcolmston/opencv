package intensity

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// GammaLUT builds and returns the 256-entry power-law lookup table
//
//	lut[i] = round(255 · (i/255)^gamma),
//
// the same table [GammaCorrection] applies. It is exported so callers can cache
// a table, compose it with other maps, or feed it to [ApplyLUT] directly. gamma
// must be finite and positive; it panics otherwise.
func GammaLUT(gamma float64) []uint8 {
	if gamma <= 0 || math.IsNaN(gamma) || math.IsInf(gamma, 0) {
		panic(fmt.Sprintf("intensity: GammaLUT requires gamma > 0, got %v", gamma))
	}
	return buildLUT(func(i int) float64 {
		return 255 * math.Pow(float64(i)/255, gamma)
	})
}

// ApplyLUT maps every sample of every channel of img through the 256-entry
// lookup table lut and returns a new [cv.Mat] of the same shape. It is the
// exported companion to the table builders such as [GammaLUT] and
// [ToneCurveLUT]. It panics on an empty image or a table that is not exactly 256
// entries long.
func ApplyLUT(img *cv.Mat, lut []uint8) *cv.Mat {
	requireImage(img, "ApplyLUT")
	return applyLUT(img, lut)
}

// AutoGamma applies an automatic power-law correction whose exponent is chosen
// from the image's own statistics so that the mean luminance is driven toward
// the mid-grey target 0.5. Writing m for the normalised mean brightness, the
// exponent is
//
//	gamma = log(0.5) / log(m),
//
// which is < 1 (brightening) for a dark image and > 1 (darkening) for a bright
// one; a mid-grey image is left essentially unchanged. The exponent is clamped
// to [0.1, 10] for numerical safety and the same table is applied to every
// channel so colours are not shifted. See [AutoGammaValue] for the exponent
// alone.
func AutoGamma(img *cv.Mat) *cv.Mat {
	requireImage(img, "AutoGamma")
	return applyLUT(img, GammaLUT(AutoGammaValue(img)))
}

// AutoGammaValue returns the exponent [AutoGamma] would use for img without
// applying it: gamma = log(0.5)/log(mean), where mean is the average luminance
// normalised to (0,1). Degenerate means (fully black or white) return 1. The
// result is clamped to [0.1, 10].
func AutoGammaValue(img *cv.Mat) float64 {
	requireImage(img, "AutoGammaValue")
	mean, _ := meanStd(lumaFloat(img))
	m := mean / 255
	if m <= 0 || m >= 1 {
		return 1
	}
	gamma := math.Log(0.5) / math.Log(m)
	if gamma < 0.1 {
		gamma = 0.1
	} else if gamma > 10 {
		gamma = 10
	}
	return gamma
}

// AGCWD applies Adaptive Gamma Correction with Weighting Distribution, after
// Huang, Cheng and Chiu (2013), "Efficient Contrast Enhancement Using Adaptive
// Gamma Correction With Weighting Distribution". Unlike a single global
// exponent, AGCWD uses a per-intensity exponent derived from a reshaped
// histogram, which lifts dark, low-contrast images while resisting the
// over-enhancement of already well-exposed ones.
//
// The method builds the luminance histogram, forms a weighted probability
// density
//
//	pdf_w(l) = pdf_max · ((pdf(l) − pdf_min) / (pdf_max − pdf_min))^alpha,
//
// accumulates it into a normalised CDF, and maps each level through
// out(l) = 255·(l/255)^(1 − cdf_w(l)). alpha in (0,1] controls how strongly the
// distribution is flattened (0.5 is the paper's default); it panics unless
// 0 < alpha ≤ 1. The intensity-indexed table is applied to every channel, so a
// colour image keeps its hue while its brightness structure is enhanced. A
// perfectly flat image is returned unchanged.
func AGCWD(img *cv.Mat, alpha float64) *cv.Mat {
	requireImage(img, "AGCWD")
	if !(alpha > 0 && alpha <= 1) {
		panic(fmt.Sprintf("intensity: AGCWD requires 0 < alpha <= 1, got %v", alpha))
	}
	luma := lumaFloat(img)
	hist := histFloat(luma)
	total := len(luma)

	// Probability density and its extremes. Following Huang et al., pdf_min and
	// pdf_max are taken over the whole 0..255 range (so empty bins pull pdf_min
	// to zero); this keeps the weighting smooth across the occupied levels rather
	// than letting the single most-frequent level dominate.
	pdf := make([]float64, 256)
	pdfMin := math.Inf(1)
	pdfMax := 0.0
	distinct := 0
	for i := 0; i < 256; i++ {
		p := float64(hist[i]) / float64(total)
		pdf[i] = p
		if p < pdfMin {
			pdfMin = p
		}
		if p > pdfMax {
			pdfMax = p
		}
		if hist[i] > 0 {
			distinct++
		}
	}
	if distinct <= 1 || pdfMax <= pdfMin {
		return img.Clone() // degenerate / single-value / perfectly flat histogram.
	}

	// Weighted (reshaped) density and its cumulative sum.
	span := pdfMax - pdfMin
	wpdf := make([]float64, 256)
	var wsum float64
	for i := 0; i < 256; i++ {
		if pdf[i] <= 0 {
			continue
		}
		w := pdfMax * math.Pow((pdf[i]-pdfMin)/span, alpha)
		wpdf[i] = w
		wsum += w
	}
	if wsum <= 0 {
		return img.Clone()
	}

	lut := make([]uint8, 256)
	var acc float64
	for i := 0; i < 256; i++ {
		acc += wpdf[i]
		cdf := acc / wsum
		gamma := 1 - cdf
		lut[i] = clampToUint8(255*math.Pow(float64(i)/255, gamma) + 0.5)
	}
	return applyLUT(img, lut)
}
