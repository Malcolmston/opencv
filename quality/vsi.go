package quality

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// VSI similarity-map stabilisation constants (Zhang et al. 2014). C1 acts on the
// saliency term, C2 on the gradient term, C3 on the chrominance term. alphaVSI
// and betaVSI weight the gradient and chrominance contributions.
const (
	vsiC1    = 1.27
	vsiC2    = 386.0
	vsiC3    = 130.0
	alphaVSI = 0.40
	betaVSI  = 0.02
)

// sdspSaliency computes an SDSP-style visual-saliency map (Zhang et al. 2013)
// for the source image m in the range [0, 1]. It is the product of three
// priors: a frequency prior (a difference-of-Gaussians band-pass of luminance
// that responds to conspicuous structure), a colour prior (chroma magnitude in
// YIQ, constant for grey images), and a location prior (a Gaussian favouring the
// image centre, where observers fixate). The map is normalised to [0, 1].
func sdspSaliency(m *cv.Mat) grid {
	g := toGray(m)

	// Frequency prior: band-pass magnitude of luminance.
	band := sub(gaussBlur(g, oddKernel(2), 2), gaussBlur(g, oddKernel(10), 10))
	freq := absGrid(band)

	// Colour prior: chroma magnitude.
	iCh, qCh, hasColor := chromaIQ(m)

	// Location prior: centred Gaussian.
	cy := float64(g.rows-1) / 2
	cx := float64(g.cols-1) / 2
	sigmaD := 0.25 * math.Hypot(float64(g.rows), float64(g.cols))
	twoSigD2 := 2 * sigmaD * sigmaD

	out := newGrid(g.rows, g.cols)
	var maxV float64
	for y := 0; y < g.rows; y++ {
		for x := 0; x < g.cols; x++ {
			i := g.idx(y, x)
			loc := math.Exp(-((float64(y)-cy)*(float64(y)-cy) +
				(float64(x)-cx)*(float64(x)-cx)) / twoSigD2)
			col := 1.0
			if hasColor {
				chroma := math.Hypot(iCh.data[i], qCh.data[i])
				// Squash chroma into a bounded saliency multiplier in (0, 1].
				col = 1 - math.Exp(-chroma/128.0)
			}
			v := freq.data[i] * loc * (0.5 + 0.5*col)
			out.data[i] = v
			if v > maxV {
				maxV = v
			}
		}
	}
	if maxV > 0 {
		for i := range out.data {
			out.data[i] /= maxV
		}
	}
	return out
}

// VSI returns the visual-saliency-induced index between reference a and
// candidate b (Zhang et al. 2014). It builds a visual-saliency map for each
// image and combines saliency similarity, gradient-magnitude similarity and
// (for colour inputs) chrominance similarity into a per-pixel score, pooled with
// the element-wise maximum saliency as the weight so that changes in conspicuous
// regions matter most. The score is 1 for identical images and decreases
// monotonically with distortion.
//
// The saliency map is an SDSP approximation (see [sdspSaliency]). VSI uses the
// I/Q chrominance channels for colour inputs and omits the chrominance term for
// single-channel inputs. It panics unless the two images share a size and
// channel count.
func VSI(a, b *cv.Mat) float64 {
	requireComparable(a, b, "VSI")
	vs1 := sdspSaliency(a)
	vs2 := sdspSaliency(b)
	gm1 := gradientMag(toGray(a), sobelX, sobelY)
	gm2 := gradientMag(toGray(b), sobelX, sobelY)

	color := a.Channels == 3
	var i1, q1, i2, q2 grid
	if color {
		i1, q1, _ = chromaIQ(a)
		i2, q2, _ = chromaIQ(b)
	}

	var num, den float64
	for i := range vs1.data {
		sVS := (2*vs1.data[i]*vs2.data[i] + vsiC1) /
			(vs1.data[i]*vs1.data[i] + vs2.data[i]*vs2.data[i] + vsiC1)
		sG := (2*gm1.data[i]*gm2.data[i] + vsiC2) /
			(gm1.data[i]*gm1.data[i] + gm2.data[i]*gm2.data[i] + vsiC2)
		s := sVS * math.Pow(sG, alphaVSI)
		if color {
			sC := (2*(i1.data[i]*i2.data[i]+q1.data[i]*q2.data[i]) + vsiC3) /
				(i1.data[i]*i1.data[i] + q1.data[i]*q1.data[i] +
					i2.data[i]*i2.data[i] + q2.data[i]*q2.data[i] + vsiC3)
			if sC < 0 {
				sC = 0
			}
			s *= math.Pow(sC, betaVSI)
		}
		w := vs1.data[i]
		if vs2.data[i] > w {
			w = vs2.data[i]
		}
		num += s * w
		den += w
	}
	if den == 0 {
		return 1
	}
	return num / den
}
