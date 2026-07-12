package saliency

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SaliencyToHeatmap renders a single-channel saliency map as a three-channel
// RGB pseudo-colour heatmap using a "jet"-style colormap (low saliency maps to
// blue, mid to green/yellow, high to red). It is a visualisation aid, analogous
// to applying cv::applyColorMap with COLORMAP_JET. It panics if saliency is nil,
// empty or not single-channel.
func SaliencyToHeatmap(saliency *cv.Mat) *cv.Mat {
	if saliency == nil || saliency.Empty() {
		panic("saliency: SaliencyToHeatmap given an empty map")
	}
	if saliency.Channels != 1 {
		panic("saliency: SaliencyToHeatmap requires a single-channel map")
	}
	out := cv.NewMat(saliency.Rows, saliency.Cols, 3)
	for i, v := range saliency.Data {
		r, g, b := jetColor(float64(v) / 255)
		out.Data[i*3+0] = r
		out.Data[i*3+1] = g
		out.Data[i*3+2] = b
	}
	return out
}

// jetColor maps t in [0,1] to an 8-bit RGB triple along a jet colormap.
func jetColor(t float64) (r, g, b uint8) {
	if t < 0 {
		t = 0
	} else if t > 1 {
		t = 1
	}
	// Piecewise-linear jet: blue -> cyan -> green -> yellow -> red.
	clamp := func(v float64) uint8 {
		v = math.Round(v * 255)
		if v < 0 {
			v = 0
		} else if v > 255 {
			v = 255
		}
		return uint8(v)
	}
	fr := math.Min(math.Max(1.5-math.Abs(4*t-3), 0), 1)
	fg := math.Min(math.Max(1.5-math.Abs(4*t-2), 0), 1)
	fb := math.Min(math.Max(1.5-math.Abs(4*t-1), 0), 1)
	return clamp(fr), clamp(fg), clamp(fb)
}

// AdaptiveBinaryMap thresholds a single-channel saliency map at factor times its
// mean value (clamped to the 8-bit range), the adaptive threshold of Achanta et
// al. (CVPR 2009); pixels at or above the threshold become 255 and the rest 0.
// The customary factor is 2. It panics if saliency is nil, empty or not
// single-channel, or if factor is not positive.
func AdaptiveBinaryMap(saliency *cv.Mat, factor float64) *cv.Mat {
	if saliency == nil || saliency.Empty() {
		panic("saliency: AdaptiveBinaryMap given an empty map")
	}
	if saliency.Channels != 1 {
		panic("saliency: AdaptiveBinaryMap requires a single-channel map")
	}
	if factor <= 0 {
		panic("saliency: AdaptiveBinaryMap requires a positive factor")
	}
	var sum float64
	for _, v := range saliency.Data {
		sum += float64(v)
	}
	mean := sum / float64(len(saliency.Data))
	thr := factor * mean
	if thr > 255 {
		thr = 255
	}
	out := cv.NewMat(saliency.Rows, saliency.Cols, 1)
	for i, v := range saliency.Data {
		if float64(v) >= thr {
			out.Data[i] = 255
		}
	}
	return out
}

// CenterBiasPrior returns a rows×cols single-channel Gaussian center-prior map
// (brightest at the image centre, falling off toward the edges), normalised to
// the 8-bit range. sigmaFrac sets the Gaussian's standard deviation as a
// fraction of the smaller image dimension (<=0 uses 0.35). Multiplying a
// saliency map by this prior models the well-known human tendency to fixate near
// image centres. It panics if either dimension is not positive.
func CenterBiasPrior(rows, cols int, sigmaFrac float64) *cv.Mat {
	if rows <= 0 || cols <= 0 {
		panic("saliency: CenterBiasPrior requires positive dimensions")
	}
	if sigmaFrac <= 0 {
		sigmaFrac = 0.35
	}
	minDim := rows
	if cols < minDim {
		minDim = cols
	}
	sigma := sigmaFrac * float64(minDim)
	twoSig2 := 2 * sigma * sigma
	cy := float64(rows-1) / 2
	cx := float64(cols-1) / 2
	p := newPlane(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			dy := float64(y) - cy
			dx := float64(x) - cx
			p.data[y*cols+x] = math.Exp(-(dy*dy + dx*dx) / twoSig2)
		}
	}
	return p.normalizedMat()
}

// ApplyCenterBias multiplies a single-channel saliency map by a Gaussian
// center-prior (see [CenterBiasPrior]) and renormalises to the 8-bit range,
// returning a new map that suppresses off-centre responses. It panics if
// saliency is nil, empty or not single-channel.
func ApplyCenterBias(saliency *cv.Mat, sigmaFrac float64) *cv.Mat {
	if saliency == nil || saliency.Empty() {
		panic("saliency: ApplyCenterBias given an empty map")
	}
	if saliency.Channels != 1 {
		panic("saliency: ApplyCenterBias requires a single-channel map")
	}
	prior := CenterBiasPrior(saliency.Rows, saliency.Cols, sigmaFrac)
	out := newPlane(saliency.Rows, saliency.Cols)
	for i := range out.data {
		out.data[i] = float64(saliency.Data[i]) * float64(prior.Data[i])
	}
	return out.normalizedMat()
}
