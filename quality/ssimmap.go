package quality

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SSIMMap returns the per-pixel structural-similarity map of a versus b as a
// single-channel [cv.Mat], with each SSIM value clamped to [0, 1] and scaled to
// the 8-bit range for visualisation. It is the same map [SSIM] returns as its
// second result, exposed on its own for callers that want the spatial quality
// image without the pooled mean. Identical images produce an all-255 map.
//
// The map is computed on luminance with the 11×11 Gaussian window. It panics
// unless the two images share a size and channel count.
func SSIMMap(a, b *cv.Mat) *cv.Mat {
	requireComparable(a, b, "SSIMMap")
	lMap, csMap := ssimMaps(toGray(a), toGray(b))
	m := newGrid(lMap.rows, lMap.cols)
	for i := range m.data {
		m.data[i] = lMap.data[i] * csMap.data[i]
	}
	return similarityMapToMat(m)
}

// iwStats returns, for two luminance grids, the SSIM luminance map (l), the
// contrast-structure map (cs) and an information-content weight map (iw). The
// weight is log((1+σ₁²/C)(1+σ₂²/C)): it is zero in flat regions and grows in
// textured, information-rich ones, so pooling with it emphasises the parts of
// the image that carry the most visual information.
func iwStats(a, b grid) (l, cs, iw grid) {
	c1 := (ssimK1 * dynamicRange) * (ssimK1 * dynamicRange)
	c2 := (ssimK2 * dynamicRange) * (ssimK2 * dynamicRange)

	muA := gaussBlur(a, ssimWindow, ssimSigma)
	muB := gaussBlur(b, ssimWindow, ssimSigma)
	sigAA := gaussBlur(mul(a, a), ssimWindow, ssimSigma)
	sigBB := gaussBlur(mul(b, b), ssimWindow, ssimSigma)
	sigAB := gaussBlur(mul(a, b), ssimWindow, ssimSigma)

	l = newGrid(a.rows, a.cols)
	cs = newGrid(a.rows, a.cols)
	iw = newGrid(a.rows, a.cols)
	for i := range l.data {
		muAA := muA.data[i] * muA.data[i]
		muBB := muB.data[i] * muB.data[i]
		muAB := muA.data[i] * muB.data[i]
		vAA := sigAA.data[i] - muAA
		vBB := sigBB.data[i] - muBB
		vAB := sigAB.data[i] - muAB
		l.data[i] = (2*muAB + c1) / (muAA + muBB + c1)
		cs.data[i] = (2*vAB + c2) / (vAA + vBB + c2)
		if vAA < 0 {
			vAA = 0
		}
		if vBB < 0 {
			vBB = 0
		}
		iw.data[i] = math.Log((1+vAA/c2)*(1+vBB/c2)) + iwFloor
	}
	return l, cs, iw
}

// iwFloor is a small positive baseline weight so that flat regions (where the
// log term is zero) still contribute, keeping the weighted pooling well-defined.
const iwFloor = 1e-3

// weightedMean returns the weighted arithmetic mean of vals under weights w,
// falling back to the unweighted mean when the weights sum to zero.
func weightedMean(vals, w []float64) float64 {
	var num, den float64
	for i := range vals {
		num += w[i] * vals[i]
		den += w[i]
	}
	if den == 0 {
		return meanOf(vals)
	}
	return num / den
}

// IWSSIM returns the information-content-weighted multi-scale structural
// similarity index between a and b (Wang & Li 2011). It follows the same image
// pyramid as [MSSSIM] but, instead of averaging each scale's similarity map
// uniformly, pools it with an information-content weight that up-weights
// textured, high-variance regions and down-weights flat ones. This tracks
// perceived quality more closely than plain MS-SSIM. The score is 1 for
// identical images and decreases monotonically with distortion.
//
// IWSSIM is evaluated on luminance. It panics unless the two images share a
// size and channel count.
func IWSSIM(a, b *cv.Mat) float64 {
	requireComparable(a, b, "IWSSIM")
	ga, gb := toGray(a), toGray(b)

	const minSide = 16
	scales := len(msssimWeights)
	for scales > 1 {
		side := scales - 1
		if (ga.rows>>uint(side)) >= minSide && (ga.cols>>uint(side)) >= minSide {
			break
		}
		scales--
	}
	weights := msssimWeights[:scales]
	var wsum float64
	for _, w := range weights {
		wsum += w
	}

	product := 1.0
	curA, curB := ga, gb
	for s := 0; s < scales; s++ {
		l, cs, iw := iwStats(curA, curB)
		w := weights[s] / wsum
		product *= math.Pow(relu(weightedMean(cs.data, iw.data)), w)
		if s == scales-1 {
			product *= math.Pow(relu(weightedMean(l.data, iw.data)), w)
		}
		if s < scales-1 {
			curA = downsample2(curA)
			curB = downsample2(curB)
		}
	}
	return product
}

// cwssimK stabilises the complex-wavelet SSIM ratios against tiny magnitudes.
const cwssimK = 0.01

// cwDerivX and cwDerivY are central-difference kernels used to form the
// odd-symmetric (quadrature) component of the complex band-pass coefficients.
var (
	cwDerivX = [9]float64{0, 0, 0, -0.5, 0, 0.5, 0, 0, 0}
	cwDerivY = [9]float64{0, -0.5, 0, 0, 0, 0, 0, 0.5, 0}
)

// cwOrientation returns the mean local complex-wavelet SSIM of two luminance
// grids for a single orientation, whose odd-symmetric component is formed with
// the given derivative kernel.
func cwOrientation(g1, g2 grid, deriv [9]float64) float64 {
	even1 := sub(gaussBlur(g1, oddKernel(1), 1), gaussBlur(g1, oddKernel(2), 2))
	even2 := sub(gaussBlur(g2, oddKernel(1), 1), gaussBlur(g2, oddKernel(2), 2))
	smooth1 := gaussBlur(g1, oddKernel(1.5), 1.5)
	smooth2 := gaussBlur(g2, oddKernel(1.5), 1.5)
	odd1 := conv3(smooth1, deriv)
	odd2 := conv3(smooth2, deriv)

	n := len(g1.data)
	mag1sq := newGrid(g1.rows, g1.cols)
	mag2sq := newGrid(g1.rows, g1.cols)
	magProd := newGrid(g1.rows, g1.cols)
	crossRe := newGrid(g1.rows, g1.cols)
	crossIm := newGrid(g1.rows, g1.cols)
	for i := 0; i < n; i++ {
		a1, b1 := even1.data[i], odd1.data[i]
		a2, b2 := even2.data[i], odd2.data[i]
		m1 := a1*a1 + b1*b1
		m2 := a2*a2 + b2*b2
		mag1sq.data[i] = m1
		mag2sq.data[i] = m2
		magProd.data[i] = math.Sqrt(m1) * math.Sqrt(m2)
		crossRe.data[i] = a1*a2 + b1*b2
		crossIm.data[i] = b1*a2 - a1*b2
	}

	const win = 7
	const sigma = 7.0 / 6.0
	sM1 := gaussBlur(mag1sq, win, sigma)
	sM2 := gaussBlur(mag2sq, win, sigma)
	sMP := gaussBlur(magProd, win, sigma)
	sRe := gaussBlur(crossRe, win, sigma)
	sIm := gaussBlur(crossIm, win, sigma)

	var sum float64
	for i := 0; i < n; i++ {
		absCross := math.Hypot(sRe.data[i], sIm.data[i])
		term1 := (2*sMP.data[i] + cwssimK) / (sM1.data[i] + sM2.data[i] + cwssimK)
		term2 := (2*absCross + cwssimK) / (2*sMP.data[i] + cwssimK)
		sum += term1 * term2
	}
	return sum / float64(n)
}

// CWSSIM returns the complex-wavelet structural similarity index between a and
// b (Wang & Simoncelli 2005). Unlike SSIM it compares the magnitude and the
// relative phase of local complex band-pass coefficients, which makes it robust
// to small geometric distortions (a slight shift or blur changes coefficient
// phase little). This implementation uses a two-orientation quadrature filter
// bank as a steerable-pyramid approximation. The score is 1 for identical images
// and decreases with distortion.
//
// CWSSIM is evaluated on luminance. It panics unless the two images share a size
// and channel count.
func CWSSIM(a, b *cv.Mat) float64 {
	requireComparable(a, b, "CWSSIM")
	g1, g2 := toGray(a), toGray(b)
	x := cwOrientation(g1, g2, cwDerivX)
	y := cwOrientation(g1, g2, cwDerivY)
	return (x + y) / 2
}
