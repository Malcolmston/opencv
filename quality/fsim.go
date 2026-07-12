package quality

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// FSIM similarity-map stabilisation constants (Zhang et al. 2011). T1 acts on
// the phase-congruency term (which lies in [0, 1]); T2 on the Sobel gradient
// magnitude; T3/T4 on the I/Q chrominance similarity of [FSIMc]. lambdaChroma
// weights the chrominance contribution.
const (
	fsimT1      = 0.85
	fsimT2      = 160.0
	fsimT3      = 200.0
	fsimT4      = 200.0
	lambdaColor = 0.03
)

// pcSigmas are the standard deviations of the Gaussian bank whose successive
// differences form the difference-of-Gaussians bandpass channels used by the
// phase-congruency approximation.
var pcSigmas = []float64{1, 2, 4, 8}

// oddKernel returns the smallest odd kernel size that comfortably covers a
// Gaussian of the given sigma (roughly ±3σ), with a floor of 3.
func oddKernel(sigma float64) int {
	k := int(6*sigma) | 1
	if k < 3 {
		k = 3
	}
	return k
}

// phaseCongruency returns a phase-congruency-like feature map of the luminance
// grid g in the range [0, 1]. True phase congruency locates points where the
// Fourier components across scale share a common phase; this approximation
// captures the same idea cheaply by measuring how coherently a bank of
// difference-of-Gaussians bandpass responses add up: the magnitude of their
// sum (local energy) divided by the sum of their magnitudes (local amplitude).
// Where the bands reinforce — at edges and lines — the ratio approaches 1;
// where they cancel it approaches 0.
func phaseCongruency(g grid) grid {
	blurs := make([]grid, len(pcSigmas))
	for i, s := range pcSigmas {
		blurs[i] = gaussBlur(g, oddKernel(s), s)
	}
	sumAmp := newGrid(g.rows, g.cols)
	for j := 0; j+1 < len(blurs); j++ {
		band := sub(blurs[j], blurs[j+1])
		for i := range sumAmp.data {
			sumAmp.data[i] += math.Abs(band.data[i])
		}
	}
	// The signed band sum telescopes to the coarsest difference of Gaussians.
	energy := sub(blurs[0], blurs[len(blurs)-1])
	out := newGrid(g.rows, g.cols)
	const eps = 0.01
	for i := range out.data {
		out.data[i] = math.Abs(energy.data[i]) / (sumAmp.data[i] + eps)
	}
	return out
}

// fsimAccumulate computes the numerator and denominator of the FSIM pooling for
// two luminance grids, optionally folding in the I/Q chrominance similarity of
// the two source images when color is true. It also returns the per-pixel local
// similarity map (S_L, in [0, 1]) for callers that want a spatial quality image.
func fsimAccumulate(g1, g2 grid, a, b *cv.Mat, color bool) (num, den float64, slMap grid) {
	slMap = newGrid(g1.rows, g1.cols)
	pc1 := phaseCongruency(g1)
	pc2 := phaseCongruency(g2)
	gm1 := gradientMag(g1, sobelX, sobelY)
	gm2 := gradientMag(g2, sobelX, sobelY)

	var i1, q1, i2, q2 grid
	if color {
		i1, q1, _ = chromaIQ(a)
		i2, q2, _ = chromaIQ(b)
	}

	for i := range g1.data {
		sPC := (2*pc1.data[i]*pc2.data[i] + fsimT1) /
			(pc1.data[i]*pc1.data[i] + pc2.data[i]*pc2.data[i] + fsimT1)
		sG := (2*gm1.data[i]*gm2.data[i] + fsimT2) /
			(gm1.data[i]*gm1.data[i] + gm2.data[i]*gm2.data[i] + fsimT2)
		sl := sPC * sG
		if color {
			sI := (2*i1.data[i]*i2.data[i] + fsimT3) /
				(i1.data[i]*i1.data[i] + i2.data[i]*i2.data[i] + fsimT3)
			sQ := (2*q1.data[i]*q2.data[i] + fsimT4) /
				(q1.data[i]*q1.data[i] + q2.data[i]*q2.data[i] + fsimT4)
			sc := sI * sQ
			if sc < 0 {
				sc = 0
			}
			sl *= math.Pow(sc, lambdaColor)
		}
		slMap.data[i] = sl
		pcm := pc1.data[i]
		if pc2.data[i] > pcm {
			pcm = pc2.data[i]
		}
		num += sl * pcm
		den += pcm
	}
	return num, den, slMap
}

// FSIM returns the feature-similarity index between reference a and candidate b
// (Zhang et al. 2011). It combines two perceptually salient low-level features —
// phase congruency (a contrast-invariant edge/line detector) and gradient
// magnitude — into a per-pixel similarity that is then pooled with the phase
// congruency itself as the weight, emphasising structurally important pixels.
// The score is 1 for identical images and decreases monotonically with
// distortion.
//
// The phase-congruency term is a difference-of-Gaussians approximation (see
// [phaseCongruency]); it is exact enough to make identical images score 1 and
// to rank distortions consistently, but it is not the log-Gabor phase
// congruency of the original paper. FSIM is evaluated on luminance. It panics
// unless the two images share a size and channel count.
func FSIM(a, b *cv.Mat) float64 {
	requireComparable(a, b, "FSIM")
	num, den, _ := fsimAccumulate(toGray(a), toGray(b), a, b, false)
	if den == 0 {
		return 1
	}
	return num / den
}

// FSIMc returns the colour feature-similarity index between reference a and
// candidate b: [FSIM] augmented with a chrominance-similarity term computed on
// the I and Q channels of the YIQ colour space. It rewards agreement in colour
// as well as structure. For single-channel inputs there is no chrominance and
// FSIMc reduces to [FSIM]. The score is 1 for identical images and decreases
// monotonically with distortion. It panics unless the two images share a size
// and channel count.
func FSIMc(a, b *cv.Mat) float64 {
	requireComparable(a, b, "FSIMc")
	if a.Channels != 3 {
		return FSIM(a, b)
	}
	num, den, _ := fsimAccumulate(toGray(a), toGray(b), a, b, true)
	if den == 0 {
		return 1
	}
	return num / den
}
