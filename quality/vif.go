package quality

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// vifNoiseVar is the variance of the assumed additive neural noise in the HVS
// model of VIF, for signals in the [0, 255] range (Sheikh & Bovik 2006).
const vifNoiseVar = 2.0

// vifEps guards divisions by tiny local variances.
const vifEps = 1e-10

// vifStats accumulates the reference-information and distorted-information sums
// of the visual-information-fidelity criterion for one band/scale. mu, variance
// and covariance are estimated with a Gaussian window of the given size and
// sigma; the returned num/den are the summed log2 information terms of the
// distorted and reference channels respectively.
func vifStats(r, d grid, ksize int, sigma float64) (num, den float64) {
	mu1 := gaussBlur(r, ksize, sigma)
	mu2 := gaussBlur(d, ksize, sigma)
	sig1 := gaussBlur(mul(r, r), ksize, sigma)
	sig2 := gaussBlur(mul(d, d), ksize, sigma)
	sig12 := gaussBlur(mul(r, d), ksize, sigma)

	for i := range r.data {
		sigma1sq := sig1.data[i] - mu1.data[i]*mu1.data[i]
		sigma2sq := sig2.data[i] - mu2.data[i]*mu2.data[i]
		sigma12 := sig12.data[i] - mu1.data[i]*mu2.data[i]
		if sigma1sq < 0 {
			sigma1sq = 0
		}
		if sigma2sq < 0 {
			sigma2sq = 0
		}

		g := sigma12 / (sigma1sq + vifEps)
		svSq := sigma2sq - g*sigma12

		switch {
		case sigma1sq < vifEps:
			g = 0
			svSq = sigma2sq
			sigma1sq = 0
		case sigma2sq < vifEps:
			g = 0
			svSq = 0
		}
		if g < 0 {
			svSq = sigma2sq
			g = 0
		}
		if svSq < vifEps {
			svSq = vifEps
		}

		num += math.Log2(1 + g*g*sigma1sq/(svSq+vifNoiseVar))
		den += math.Log2(1 + sigma1sq/vifNoiseVar)
	}
	return num, den
}

// VIFP returns the pixel-domain visual information fidelity between reference a
// and candidate b (Sheikh & Bovik 2006). It models both images as the output of
// a Gaussian-scale-mixture source passed through the human visual system and
// reports the ratio of information the candidate shares with the reference to
// the information present in the reference itself. The score is 1 for identical
// images, lies in [0, 1] for typical distortions, and falls monotonically as
// blur or noise destroys information.
//
// The criterion is pooled over four dyadic scales, each analysed with a
// Gaussian window whose size shrinks with scale. VIFP is evaluated on
// luminance. It panics unless the two images share a size and channel count.
func VIFP(a, b *cv.Mat) float64 {
	requireComparable(a, b, "VIFP")
	r := toGray(a)
	d := toGray(b)

	var num, den float64
	for scale := 1; scale <= 4; scale++ {
		n := (1 << uint(4-scale+1)) + 1 // 17, 9, 5, 3
		sd := float64(n) / 5.0
		if scale > 1 {
			r = decimate2(gaussBlur(r, n, sd))
			d = decimate2(gaussBlur(d, n, sd))
		}
		sn, sd2 := vifStats(r, d, n, sd)
		num += sn
		den += sd2
	}
	if den == 0 {
		return 1
	}
	return num / den
}

// VIF returns a subband-domain approximation of visual information fidelity
// between reference a and candidate b. Where [VIFP] works in the pixel domain,
// VIF first decomposes each image into a Laplacian (difference-of-Gaussians)
// bandpass pyramid and applies the Gaussian-scale-mixture information criterion
// to every band, which better isolates the scale-selective behaviour of the
// original wavelet-domain VIF. The score is 1 for identical images and
// decreases monotonically with distortion.
//
// VIF is evaluated on luminance over four bandpass levels. It panics unless the
// two images share a size and channel count.
func VIF(a, b *cv.Mat) float64 {
	requireComparable(a, b, "VIF")
	cr := toGray(a)
	cd := toGray(b)

	const levels = 4
	var num, den float64
	for s := 0; s < levels; s++ {
		lowR := gaussBlur(cr, 11, 1.5)
		lowD := gaussBlur(cd, 11, 1.5)
		bandR := sub(cr, lowR)
		bandD := sub(cd, lowD)
		sn, sd := vifStats(bandR, bandD, 7, 7.0/6.0)
		num += sn
		den += sd
		if s < levels-1 {
			cr = downsample2(lowR)
			cd = downsample2(lowD)
		}
	}
	if den == 0 {
		return 1
	}
	return num / den
}
