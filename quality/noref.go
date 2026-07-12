package quality

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// LaplacianVariance returns the variance of the Laplacian response of img, the
// classic passive auto-focus / sharpness measure: a sharply focused image has a
// strong, high-variance Laplacian, while blur suppresses it. img is reduced to
// luminance first. It panics on an empty image.
func LaplacianVariance(img *cv.Mat) float64 {
	requireImage(img, "LaplacianVariance")
	g := toGray(img)
	lap := conv3(g, laplacian4)
	return variance(lap.data)
}

// Sharpness is an alias for [LaplacianVariance]: it reports the variance of the
// Laplacian response, higher meaning sharper. It is provided under this name
// because "image sharpness" is the metric's most common use.
func Sharpness(img *cv.Mat) float64 {
	return LaplacianVariance(img)
}

// Tenengrad returns the Tenengrad focus measure of img: the mean squared Sobel
// gradient magnitude (gx²+gy²) over all pixels. Like [LaplacianVariance] it is
// larger for sharper images. img is reduced to luminance first. It panics on an
// empty image.
func Tenengrad(img *cv.Mat) float64 {
	requireImage(img, "Tenengrad")
	g := toGray(img)
	gx := conv3(g, sobelX)
	gy := conv3(g, sobelY)
	var sum float64
	for i := range gx.data {
		sum += gx.data[i]*gx.data[i] + gy.data[i]*gy.data[i]
	}
	return sum / float64(len(gx.data))
}

// mscn returns the mean-subtracted contrast-normalised (MSCN) coefficients of
// the luminance grid g, the local features that BRISQUE is built on. Each
// coefficient is (v-μ)/(σ+1) where μ and σ are the local Gaussian-weighted mean
// and standard deviation over a 7×7 window.
func mscn(g grid) grid {
	const win = 7
	const sigma = 7.0 / 6.0
	mu := gaussBlur(g, win, sigma)
	muSq := mul(mu, mu)
	meanSq := gaussBlur(mul(g, g), win, sigma)

	out := newGrid(g.rows, g.cols)
	for i := range out.data {
		v := meanSq.data[i] - muSq.data[i]
		if v < 0 {
			v = 0
		}
		sd := math.Sqrt(v)
		out.data[i] = (g.data[i] - mu.data[i]) / (sd + 1)
	}
	return out
}

// BrisqueScore returns a lightweight, no-reference BRISQUE-style heuristic for
// img: the variance of its mean-subtracted contrast-normalised (MSCN)
// coefficients. Sharper, more textured images carry more high-frequency
// structure and score higher; heavy blur drives the score toward zero.
//
// This is deliberately a "lite" proxy. The calibrated OpenCV BRISQUE score
// summarises the MSCN distribution with 36 generalised-Gaussian features fed to
// a trained support-vector regressor; without that model this function reports
// a single interpretable statistic instead. img is reduced to luminance first.
// It panics on an empty image.
func BrisqueScore(img *cv.Mat) float64 {
	requireImage(img, "BrisqueScore")
	return variance(mscn(toGray(img)).data)
}

// variance returns the population variance of xs.
func variance(xs []float64) float64 {
	s := popStdDev(xs)
	return s * s
}
