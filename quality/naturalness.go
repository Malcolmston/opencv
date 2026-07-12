package quality

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// gammaRatioTable holds r(α) = Γ(2/α)² / (Γ(1/α)·Γ(3/α)) sampled on a fine grid
// of shape parameters α. Moment-matching estimators invert this monotone map by
// nearest-neighbour lookup, avoiding a special-function root solve. It is built
// once at package initialisation.
var gammaRatioTable = buildGammaRatioTable()

const (
	gammaAlphaMin  = 0.20
	gammaAlphaMax  = 10.0
	gammaAlphaStep = 0.001
)

// buildGammaRatioTable tabulates r(α) over [gammaAlphaMin, gammaAlphaMax].
func buildGammaRatioTable() []float64 {
	n := int((gammaAlphaMax-gammaAlphaMin)/gammaAlphaStep) + 1
	t := make([]float64, n)
	for i := 0; i < n; i++ {
		a := gammaAlphaMin + float64(i)*gammaAlphaStep
		g1 := math.Gamma(1 / a)
		g2 := math.Gamma(2 / a)
		g3 := math.Gamma(3 / a)
		t[i] = (g2 * g2) / (g1 * g3)
	}
	return t
}

// alphaForRatio returns the shape parameter α whose r(α) is closest to the
// target ratio.
func alphaForRatio(ratio float64) float64 {
	best := 0
	bestErr := math.Inf(1)
	for i, v := range gammaRatioTable {
		e := math.Abs(v - ratio)
		if e < bestErr {
			bestErr = e
			best = i
		}
	}
	return gammaAlphaMin + float64(best)*gammaAlphaStep
}

// estimateGGD fits a zero-mean generalised Gaussian distribution to xs by
// moment matching, returning the shape parameter α and the variance σ².
func estimateGGD(xs []float64) (alpha, sigmaSq float64) {
	var sqSum, absSum float64
	for _, v := range xs {
		sqSum += v * v
		absSum += math.Abs(v)
	}
	n := float64(len(xs))
	sigmaSq = sqSum / n
	mAbs := absSum / n
	if sigmaSq == 0 {
		return 2, 0
	}
	rho := (mAbs * mAbs) / sigmaSq
	return alphaForRatio(rho), sigmaSq
}

// estimateAGGD fits an asymmetric generalised Gaussian distribution to xs,
// returning the shape parameter α, the distribution mean, and the left and
// right scale parameters. These are the four features BRISQUE/NIQE extract from
// each oriented paired-product map.
func estimateAGGD(xs []float64) (alpha, mean, leftStd, rightStd float64) {
	var leftSq, rightSq, absSum, sqSum float64
	var leftN, rightN int
	for _, v := range xs {
		absSum += math.Abs(v)
		sqSum += v * v
		if v < 0 {
			leftSq += v * v
			leftN++
		} else {
			rightSq += v * v
			rightN++
		}
	}
	n := float64(len(xs))
	if leftN > 0 {
		leftStd = math.Sqrt(leftSq / float64(leftN))
	}
	if rightN > 0 {
		rightStd = math.Sqrt(rightSq / float64(rightN))
	}
	if rightStd == 0 {
		return 2, 0, leftStd, 0
	}
	gammaHat := leftStd / rightStd
	meanSq := sqSum / n
	if meanSq == 0 {
		return 2, 0, 0, 0
	}
	rHat := (absSum / n) * (absSum / n) / meanSq
	num := gammaHat*gammaHat*gammaHat + 1
	den := gammaHat*gammaHat + 1
	rHatNorm := rHat * num * (gammaHat + 1) / (den * den)
	alpha = alphaForRatio(rHatNorm)

	g1 := math.Gamma(1 / alpha)
	g2 := math.Gamma(2 / alpha)
	g3 := math.Gamma(3 / alpha)
	constant := math.Sqrt(g1 / g3)
	mean = (rightStd - leftStd) * (g2 / g1) * constant
	return alpha, mean, leftStd, rightStd
}

// niqeScaleFeatures extracts the 18 natural-scene-statistics features of one
// scale: the two GGD parameters of the MSCN coefficients followed by the four
// AGGD parameters of each of the four (horizontal, vertical, and two diagonal)
// paired-product maps.
func niqeScaleFeatures(g grid) []float64 {
	m := mscn(g)
	feats := make([]float64, 0, 18)
	a, s := estimateGGD(m.data)
	feats = append(feats, a, s)

	type shift struct{ dy, dx int }
	for _, sh := range []shift{{0, 1}, {1, 0}, {1, 1}, {1, -1}} {
		prod := make([]float64, 0, len(m.data))
		for y := 0; y < m.rows; y++ {
			ny := y + sh.dy
			if ny < 0 || ny >= m.rows {
				continue
			}
			for x := 0; x < m.cols; x++ {
				nx := x + sh.dx
				if nx < 0 || nx >= m.cols {
					continue
				}
				prod = append(prod, m.data[m.idx(y, x)]*m.data[m.idx(ny, nx)])
			}
		}
		aa, mm, ls, rs := estimateAGGD(prod)
		feats = append(feats, aa, mm, ls, rs)
	}
	return feats
}

// niqeFeatures returns the full 36-dimensional NIQE feature vector of a
// luminance grid: the 18 per-scale features at full resolution followed by the
// 18 at half resolution.
func niqeFeatures(g grid) []float64 {
	f := niqeScaleFeatures(g)
	return append(f, niqeScaleFeatures(downsample2(g))...)
}

// niqePristine is the mean feature vector of the opinion-unaware natural-scene
// model, fixed at build time. It was obtained by fitting the 36 NIQE features to
// a deterministic pristine reference (a smooth, gently textured natural-like
// image); distortions such as noise and blur move an image's features away from
// this vector, increasing the reported distance. Because the model is fixed and
// derived only from undistorted statistics, NIQE needs no human opinion scores.
var niqePristine = []float64{
	// full-scale: ggd(alpha,sigma) + 4×aggd(alpha,mean,left,right)
	1.4820, 0.0237, 0.5470, 0.0157, 0.0078, 0.0353,
	0.4760, 0.0174, 0.0070, 0.0396, 0.6060, 0.0129,
	0.0100, 0.0315, 0.5570, 0.0159, 0.0066, 0.0341,
	// half-scale
	3.1850, 0.0702, 0.8980, 0.0578, 0.0154, 0.0996,
	0.7840, 0.0631, 0.0104, 0.1061, 1.0730, 0.0388,
	0.0282, 0.0822, 0.8970, 0.0587, 0.0151, 0.1006,
}

// niqeRelVar and niqeVarFloor define the fixed diagonal covariance of the
// pristine model as var_k = max((relVar·ν_k)², floor), so that each feature is
// scored on its own natural scale.
const (
	niqeRelVar   = 0.5
	niqeVarFloor = 0.02
)

// NIQE returns the Natural Image Quality Evaluator score of img (Mittal et al.
// 2013): an opinion-unaware, no-reference measure of how far the image's natural
// scene statistics stray from those of undistorted natural images. It fits a
// generalised-Gaussian model to the mean-subtracted contrast-normalised (MSCN)
// coefficients and their oriented paired products at two scales, forming a
// 36-dimensional feature vector, then reports its (diagonal-covariance)
// Mahalanobis distance to a fixed pristine model. Lower is better: a pristine
// image scores near zero and blur or noise raises the score.
//
// This is a self-contained, opinion-unaware implementation with documented
// fixed parameters ([niqePristine] and a fixed diagonal covariance); it does not
// reproduce the exact full-covariance model distributed with OpenCV, but it
// captures the same natural-scene-statistics principle. img is reduced to
// luminance first. It panics on an empty image.
func NIQE(img *cv.Mat) float64 {
	requireImage(img, "NIQE")
	f := niqeFeatures(toGray(img))
	var sum float64
	for k := range f {
		d := f[k] - niqePristine[k]
		v := niqeRelVar * niqePristine[k]
		variance := v * v
		if variance < niqeVarFloor {
			variance = niqeVarFloor
		}
		sum += d * d / variance
	}
	return math.Sqrt(sum)
}

// PIQE block size and thresholds (Venkatanath et al. 2015). Blocks whose MSCN
// activity falls below piqeActivityThresh are treated as flat and excluded from
// pooling; the noise and blockiness criteria are stabilised by piqeC.
const (
	piqeBlock          = 16
	piqeActivityThresh = 0.1
	piqeC              = 1.0
)

// PIQE returns the Perception-based Image Quality Evaluator score of img
// (Venkatanath et al. 2015): an opinion-unaware, no-reference measure built by
// scoring only the spatially active 16×16 blocks of the MSCN coefficient map.
// Each active block is penalised for noticeable noise (excess MSCN activity) and
// for blockiness (discontinuities at its borders); the block scores are pooled
// into a single value. Lower is better: a clean image scores low and additive
// noise or blocking artefacts raise the score.
//
// img is reduced to luminance first. It panics on an empty image.
func PIQE(img *cv.Mat) float64 {
	requireImage(img, "PIQE")
	m := mscn(toGray(img))

	var scoreSum float64
	var active int
	for by := 0; by+piqeBlock <= m.rows; by += piqeBlock {
		for bx := 0; bx+piqeBlock <= m.cols; bx += piqeBlock {
			// Block MSCN statistics.
			var sum, sumSq float64
			cnt := float64(piqeBlock * piqeBlock)
			for y := 0; y < piqeBlock; y++ {
				for x := 0; x < piqeBlock; x++ {
					v := m.data[m.idx(by+y, bx+x)]
					sum += v
					sumSq += v * v
				}
			}
			mean := sum / cnt
			variance := sumSq/cnt - mean*mean
			if variance < 0 {
				variance = 0
			}
			std := math.Sqrt(variance)
			if std < piqeActivityThresh {
				continue // flat block: no perceptible distortion here
			}
			active++

			// Noise criterion: excess MSCN activity above the natural baseline.
			noise := std

			// Blockiness criterion: mean absolute luminance step across the
			// block's right and bottom borders.
			block := blockiness(m, by, bx)

			scoreSum += noise + block
		}
	}
	if active == 0 {
		return 0
	}
	return 100 * (scoreSum + piqeC) / (float64(active) + piqeC)
}

// blockiness measures the mean absolute MSCN discontinuity across the right and
// bottom edges of the block anchored at (by, bx), a proxy for blocking artefacts.
func blockiness(m grid, by, bx int) float64 {
	var sum float64
	var n int
	if bx+piqeBlock < m.cols {
		for y := 0; y < piqeBlock; y++ {
			l := m.data[m.idx(by+y, bx+piqeBlock-1)]
			r := m.data[m.idx(by+y, bx+piqeBlock)]
			sum += math.Abs(l - r)
			n++
		}
	}
	if by+piqeBlock < m.rows {
		for x := 0; x < piqeBlock; x++ {
			t := m.data[m.idx(by+piqeBlock-1, bx+x)]
			b := m.data[m.idx(by+piqeBlock, bx+x)]
			sum += math.Abs(t - b)
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}
