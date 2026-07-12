package quality

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// dynamicRange is the peak sample value L for 8-bit images.
const dynamicRange = 255.0

// SSIM window and stabilisation constants (Wang et al. 2004).
const (
	ssimWindow = 11
	ssimSigma  = 1.5
	ssimK1     = 0.01
	ssimK2     = 0.03
)

// MSE returns the mean squared error between a and b, one value per channel.
// Lower is better and identical images score exactly zero. It panics unless the
// two images share a size and channel count.
func MSE(a, b *cv.Mat) []float64 {
	requireComparable(a, b, "MSE")
	ch := a.Channels
	out := make([]float64, ch)
	n := a.Total()
	for p := 0; p < n; p++ {
		base := p * ch
		for c := 0; c < ch; c++ {
			d := float64(a.Data[base+c]) - float64(b.Data[base+c])
			out[c] += d * d
		}
	}
	inv := 1.0 / float64(n)
	for c := range out {
		out[c] *= inv
	}
	return out
}

// MAE returns the mean absolute error between a and b, one value per channel.
// Lower is better and identical images score zero. It panics unless the two
// images share a size and channel count.
func MAE(a, b *cv.Mat) []float64 {
	requireComparable(a, b, "MAE")
	ch := a.Channels
	out := make([]float64, ch)
	n := a.Total()
	for p := 0; p < n; p++ {
		base := p * ch
		for c := 0; c < ch; c++ {
			d := float64(a.Data[base+c]) - float64(b.Data[base+c])
			out[c] += math.Abs(d)
		}
	}
	inv := 1.0 / float64(n)
	for c := range out {
		out[c] *= inv
	}
	return out
}

// PSNR returns the peak signal-to-noise ratio between a and b in decibels,
// pooling the error across every channel. Higher is better; identical images
// yield +Inf (their pooled MSE is zero). It panics unless the two images share
// a size and channel count.
func PSNR(a, b *cv.Mat) float64 {
	requireComparable(a, b, "PSNR")
	perCh := MSE(a, b)
	var pooled float64
	for _, v := range perCh {
		pooled += v
	}
	pooled /= float64(len(perCh))
	if pooled == 0 {
		return math.Inf(1)
	}
	return 10 * math.Log10(dynamicRange*dynamicRange/pooled)
}

// ssimMaps computes, for two luminance grids of equal size, the per-pixel
// luminance term (lMap) and contrast-structure term (csMap) of SSIM using an
// 11×11 Gaussian window. Their product is the SSIM map.
func ssimMaps(a, b grid) (lMap, csMap grid) {
	c1 := (ssimK1 * dynamicRange) * (ssimK1 * dynamicRange)
	c2 := (ssimK2 * dynamicRange) * (ssimK2 * dynamicRange)

	muA := gaussBlur(a, ssimWindow, ssimSigma)
	muB := gaussBlur(b, ssimWindow, ssimSigma)
	muAA := mul(muA, muA)
	muBB := mul(muB, muB)
	muAB := mul(muA, muB)

	sigAA := gaussBlur(mul(a, a), ssimWindow, ssimSigma)
	sigBB := gaussBlur(mul(b, b), ssimWindow, ssimSigma)
	sigAB := gaussBlur(mul(a, b), ssimWindow, ssimSigma)

	lMap = newGrid(a.rows, a.cols)
	csMap = newGrid(a.rows, a.cols)
	for i := range lMap.data {
		vAA := sigAA.data[i] - muAA.data[i]
		vBB := sigBB.data[i] - muBB.data[i]
		vAB := sigAB.data[i] - muAB.data[i]
		lMap.data[i] = (2*muAB.data[i] + c1) / (muAA.data[i] + muBB.data[i] + c1)
		csMap.data[i] = (2*vAB + c2) / (vAA + vBB + c2)
	}
	return lMap, csMap
}

// SSIM returns the mean structural similarity index between a and b and the
// per-pixel SSIM map. The score lies in [-1, 1]; it is 1 for identical images
// and falls as distortion grows. The map is a single-channel [cv.Mat] with each
// SSIM value clamped to [0, 1] and scaled to [0, 255] for visualisation.
//
// SSIM is evaluated on luminance: three-channel inputs are reduced to gray
// first. It panics unless the two images share a size and channel count.
func SSIM(a, b *cv.Mat) (float64, *cv.Mat) {
	requireComparable(a, b, "SSIM")
	ga, gb := toGray(a), toGray(b)
	lMap, csMap := ssimMaps(ga, gb)

	ssim := newGrid(ga.rows, ga.cols)
	var sum float64
	for i := range ssim.data {
		v := lMap.data[i] * csMap.data[i]
		ssim.data[i] = v
		sum += v
	}
	mean := sum / float64(len(ssim.data))

	// Render the map: clamp to [0,1] then scale to the 8-bit range.
	vis := newGrid(ssim.rows, ssim.cols)
	for i, v := range ssim.data {
		if v < 0 {
			v = 0
		} else if v > 1 {
			v = 1
		}
		vis.data[i] = v * 255
	}
	return mean, grayMapToMat(vis)
}

// msssimWeights are the standard five-scale weights from Wang et al. (2003).
var msssimWeights = []float64{0.0448, 0.2856, 0.3001, 0.2363, 0.1333}

// MSSSIM returns the multi-scale structural similarity index between a and b.
// It aggregates the contrast-structure term over an image pyramid and the
// luminance term at the coarsest scale, following Wang et al. (2003). The score
// is 1 for identical images and decreases with distortion.
//
// Up to five scales are used; the pyramid stops early on small images so that
// the coarsest level stays usable, and the scale weights are renormalised over
// the levels actually visited. MSSSIM is evaluated on luminance. It panics
// unless the two images share a size and channel count.
func MSSSIM(a, b *cv.Mat) float64 {
	requireComparable(a, b, "MSSSIM")
	ga, gb := toGray(a), toGray(b)

	// Choose how many scales the image can support (keep the coarsest level
	// at least 16 px on a side so the 11×11 window remains meaningful).
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
		lMap, csMap := ssimMaps(curA, curB)
		w := weights[s] / wsum

		// Contrast-structure at every scale; luminance only at the coarsest.
		cs := relu(meanOf(csMap.data))
		product *= math.Pow(cs, w)
		if s == scales-1 {
			l := relu(meanOf(lMap.data))
			product *= math.Pow(l, w)
		}

		if s < scales-1 {
			curA = downsample2(curA)
			curB = downsample2(curB)
		}
	}
	return product
}

// relu clamps negative values to zero so they can be raised to a fractional
// power without producing NaN.
func relu(v float64) float64 {
	if v < 0 {
		return 0
	}
	return v
}

// gmsdT is the stabilisation constant for GMSD on images in the [0, 255] range
// (Xue et al. 2014).
const gmsdT = 170.0

// GMSD returns the gradient magnitude similarity deviation between a and b and
// the per-pixel gradient-magnitude-similarity (GMS) map. GMSD is the standard
// deviation of the GMS map: lower is better and identical images score zero.
// The map is a single-channel [cv.Mat] with the GMS value (in [0, 1]) scaled to
// [0, 255].
//
// GMSD is evaluated on luminance using Prewitt gradients. It panics unless the
// two images share a size and channel count.
func GMSD(a, b *cv.Mat) (float64, *cv.Mat) {
	requireComparable(a, b, "GMSD")
	ga, gb := toGray(a), toGray(b)
	gmA := gradientMag(ga, prewittX, prewittY)
	gmB := gradientMag(gb, prewittX, prewittY)

	gms := newGrid(ga.rows, ga.cols)
	for i := range gms.data {
		na, nb := gmA.data[i], gmB.data[i]
		gms.data[i] = (2*na*nb + gmsdT) / (na*na + nb*nb + gmsdT)
	}
	dev := popStdDev(gms.data)

	vis := newGrid(gms.rows, gms.cols)
	for i, v := range gms.data {
		vis.data[i] = v * 255
	}
	return dev, grayMapToMat(vis)
}

// uqiWindow is the side length of the sliding window used by UQI.
const uqiWindow = 8

// UQI returns the universal quality index between a and b (Wang & Bovik 2002),
// the predecessor of SSIM. It is the mean local quality over every fully
// contained uqiWindow×uqiWindow window (or the whole image when it is smaller
// than the window). The score lies in [-1, 1]; it is 1 for identical images and
// falls with distortion.
//
// UQI is evaluated on luminance. It panics unless the two images share a size
// and channel count.
func UQI(a, b *cv.Mat) float64 {
	requireComparable(a, b, "UQI")
	ga, gb := toGray(a), toGray(b)

	win := uqiWindow
	if win > ga.rows {
		win = ga.rows
	}
	if win > ga.cols {
		win = ga.cols
	}
	n := float64(win * win)
	// Divisor for the unbiased sample variance; guard the degenerate 1-px case.
	d := n - 1
	if d <= 0 {
		d = 1
	}

	var sum float64
	var count int
	for y := 0; y+win <= ga.rows; y++ {
		for x := 0; x+win <= ga.cols; x++ {
			var sa, sb, saa, sbb, sab float64
			for wy := 0; wy < win; wy++ {
				for wx := 0; wx < win; wx++ {
					va := ga.data[ga.idx(y+wy, x+wx)]
					vb := gb.data[gb.idx(y+wy, x+wx)]
					sa += va
					sb += vb
					saa += va * va
					sbb += vb * vb
					sab += va * vb
				}
			}
			muA := sa / n
			muB := sb / n
			// Sample (unbiased) variance/covariance, as in the original paper.
			varA := (saa - sa*muA) / d
			varB := (sbb - sb*muB) / d
			covAB := (sab - sa*muB) / d

			denom := (varA + varB) * (muA*muA + muB*muB)
			var q float64
			if denom == 0 {
				// Both windows flat and equal in mean ⇒ perfect; otherwise the
				// numerator is zero too and the window is undefined ⇒ treat as 1.
				q = 1
			} else {
				q = (4 * covAB * muA * muB) / denom
			}
			sum += q
			count++
		}
	}
	if count == 0 {
		return 1
	}
	return sum / float64(count)
}
