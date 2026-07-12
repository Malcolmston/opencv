package photo

import (
	cv "github.com/malcolmston/opencv"
)

// Decolor converts a three-channel RGB image to grayscale in a
// contrast-preserving way and also returns a colour-boosted version of the
// input. Instead of the fixed BT.601 luma — which collapses iso-luminant colours
// to the same gray and destroys their contrast — Decolor searches over
// non-negative channel weights that sum to one and picks the mixture maximising
// the variance (global contrast) of the resulting grayscale. This keeps distinct
// colours distinguishable in the gray output.
//
// The returned gray is single-channel. The returned boost is a three-channel
// image with its saturation increased, a companion output analogous to OpenCV's
// color_boost. img must be three-channel.
func Decolor(img *cv.Mat) (gray *cv.Mat, boost *cv.Mat) {
	if img == nil || img.Empty() {
		panic("photo: Decolor given an empty image")
	}
	requireChannels(img, 3, "Decolor")

	rows, cols := img.Rows, img.Cols
	n := rows * cols

	// Pre-extract channels as floats.
	r := make([]float64, n)
	g := make([]float64, n)
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		r[i] = float64(img.Data[i*3+0])
		g[i] = float64(img.Data[i*3+1])
		b[i] = float64(img.Data[i*3+2])
	}

	// Search the weight simplex (step 0.1) for the mixture with maximum variance.
	const step = 0.1
	bestVar := -1.0
	var bestWr, bestWg, bestWb float64
	for iwr := 0; iwr <= 10; iwr++ {
		wr := float64(iwr) * step
		for iwg := 0; iwg <= 10-iwr; iwg++ {
			wg := float64(iwg) * step
			wb := 1 - wr - wg
			// Compute mean and variance of the candidate grayscale.
			var mean float64
			for i := 0; i < n; i++ {
				mean += wr*r[i] + wg*g[i] + wb*b[i]
			}
			mean /= float64(n)
			var variance float64
			for i := 0; i < n; i++ {
				v := wr*r[i] + wg*g[i] + wb*b[i]
				d := v - mean
				variance += d * d
			}
			variance /= float64(n)
			if variance > bestVar {
				bestVar, bestWr, bestWg, bestWb = variance, wr, wg, wb
			}
		}
	}

	gray = cv.NewMat(rows, cols, 1)
	for i := 0; i < n; i++ {
		gray.Data[i] = clampU8(bestWr*r[i] + bestWg*g[i] + bestWb*b[i])
	}

	boost = boostSaturation(img, 1.5)
	return gray, boost
}

// boostSaturation returns img with its HSV saturation multiplied by factor.
func boostSaturation(img *cv.Mat, factor float64) *cv.Mat {
	hsv := cv.CvtColor(img, cv.ColorRGB2HSV)
	for p := 0; p < hsv.Total(); p++ {
		hsv.Data[p*3+1] = clampU8(float64(hsv.Data[p*3+1]) * factor)
	}
	return cv.CvtColor(hsv, cv.ColorHSV2RGB)
}
