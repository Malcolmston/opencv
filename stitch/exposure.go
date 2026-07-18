package stitch

import (
	cv "github.com/malcolmston/opencv"
)

// MeanIntensityMasked returns the mean grayscale intensity of img over the pixels
// whose weight is strictly positive, together with the number of such pixels. It
// returns (0, 0) if no pixel is covered. img and weight must have the same
// dimensions.
func MeanIntensityMasked(img *cv.Mat, weight *cv.FloatMat) (mean float64, count int) {
	var sum float64
	for p := 0; p < img.Rows*img.Cols; p++ {
		if weight.Data[p] <= 0 {
			continue
		}
		sum += grayValue(img, p%img.Cols, p/img.Cols)
		count++
	}
	if count == 0 {
		return 0, 0
	}
	return sum / float64(count), count
}

// ApplyGain returns a copy of img with every sample multiplied by gain and
// clamped to the valid 8-bit range. A gain of 1 reproduces the input.
func ApplyGain(img *cv.Mat, gain float64) *cv.Mat {
	out := cv.NewMat(img.Rows, img.Cols, img.Channels)
	for i, v := range img.Data {
		out.Data[i] = clampByte(float64(v) * gain)
	}
	return out
}

// EstimateGains solves for the per-image exposure gains that best equalise
// brightness across a mosaic, following Brown and Lowe's gain-compensation model.
// means and counts are n×n matrices: means[i][j] is the mean intensity of image i
// over its overlap with image j (the diagonal means[i][i] is the mean over the
// whole image), and counts[i][j] is the number of overlapping pixels
// (counts[i][i] is the image area). sigmaN is the standard deviation of the
// intensity error and sigmaG that of the gain prior; smaller sigmaG pulls the
// gains harder toward 1.
//
// It returns one gain per image. If the normal equations are singular (for
// example no overlaps at all) it returns all-ones gains.
func EstimateGains(means, counts [][]float64, sigmaN, sigmaG float64) []float64 {
	n := len(means)
	if n == 0 {
		return nil
	}
	alpha := 1 / (sigmaN * sigmaN)
	beta := 1 / (sigmaG * sigmaG)
	a := make([][]float64, n)
	b := make([]float64, n)
	for i := 0; i < n; i++ {
		a[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			nij := counts[i][j]
			if nij == 0 {
				continue
			}
			b[i] += beta * nij
			a[i][i] += beta * nij
			if j == i {
				continue
			}
			a[i][i] += 2 * alpha * means[i][j] * means[i][j] * nij
			a[i][j] -= 2 * alpha * means[i][j] * means[j][i] * nij
		}
	}
	g, ok := solveLinear(a, b)
	if !ok {
		g = make([]float64, n)
		for i := range g {
			g[i] = 1
		}
	}
	return g
}

// GainCompensator estimates and applies per-image exposure gains so that
// overlapping images share a consistent brightness. Feed it the warped layers of
// a mosaic, then read the gains or apply them to each layer.
type GainCompensator struct {
	// SigmaN is the assumed standard deviation of the per-pixel intensity error.
	// Zero selects a sensible default (10).
	SigmaN float64
	// SigmaG is the standard deviation of the gain prior; smaller values keep
	// gains closer to 1. Zero selects a sensible default (0.1).
	SigmaG float64

	gains []float64
}

// NewGainCompensator returns a compensator with default noise and gain-prior
// parameters.
func NewGainCompensator() *GainCompensator {
	return &GainCompensator{SigmaN: 10, SigmaG: 0.1}
}

// Feed measures the pairwise overlap statistics of the layers and solves for the
// exposure gains, storing them for [GainCompensator.Gains] and
// [GainCompensator.Apply]. All layers must share the same canvas size. It returns
// the estimated gains.
func (g *GainCompensator) Feed(layers []Layer) []float64 {
	sigmaN := g.SigmaN
	if sigmaN <= 0 {
		sigmaN = 10
	}
	sigmaG := g.SigmaG
	if sigmaG <= 0 {
		sigmaG = 0.1
	}
	n := len(layers)
	means := make([][]float64, n)
	counts := make([][]float64, n)
	for i := 0; i < n; i++ {
		means[i] = make([]float64, n)
		counts[i] = make([]float64, n)
	}
	for i := 0; i < n; i++ {
		li := layers[i]
		total := li.Image.Rows * li.Image.Cols
		// Diagonal: whole-image statistics.
		var sSelf float64
		var cSelf int
		for p := 0; p < total; p++ {
			if li.Weight.Data[p] <= 0 {
				continue
			}
			sSelf += grayValue(li.Image, p%li.Image.Cols, p/li.Image.Cols)
			cSelf++
		}
		if cSelf > 0 {
			means[i][i] = sSelf / float64(cSelf)
		}
		counts[i][i] = float64(cSelf)
		for j := 0; j < n; j++ {
			if j == i {
				continue
			}
			lj := layers[j]
			var sum float64
			var cnt int
			for p := 0; p < total; p++ {
				if li.Weight.Data[p] <= 0 || lj.Weight.Data[p] <= 0 {
					continue
				}
				sum += grayValue(li.Image, p%li.Image.Cols, p/li.Image.Cols)
				cnt++
			}
			if cnt > 0 {
				means[i][j] = sum / float64(cnt)
			}
			counts[i][j] = float64(cnt)
		}
	}
	g.gains = EstimateGains(means, counts, sigmaN, sigmaG)
	return g.gains
}

// Gains returns the exposure gains computed by the most recent
// [GainCompensator.Feed], one per layer, or nil if Feed has not been called.
func (g *GainCompensator) Gains() []float64 {
	return g.gains
}

// Apply multiplies the colour of layer index by its estimated gain and returns
// the corrected layer (sharing the original weight map). It returns the layer
// unchanged if no gain is available for that index.
func (g *GainCompensator) Apply(index int, layer Layer) Layer {
	if index < 0 || index >= len(g.gains) {
		return layer
	}
	return Layer{Image: ApplyGain(layer.Image, g.gains[index]), Weight: layer.Weight}
}
