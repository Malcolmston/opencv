package hdr

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Tonemap is the common interface for operators that compress a linear
// high-dynamic-range [Radiance] map into a displayable 8-bit [cv.Mat] with
// samples in [0,255]. Implementations are deterministic.
type Tonemap interface {
	// Process tonemaps the radiance map and returns an 8-bit image with the
	// same dimensions and channel count.
	Process(r *Radiance) *cv.Mat
}

// logAvgLuminance returns the log-average (geometric mean) luminance of a plane
// and its maximum, the two anchors most operators key off. eps guards log(0).
func logAvgLuminance(lum *plane) (logAvg, maxL float64) {
	const eps = 1e-6
	var sumLog float64
	maxL = eps
	for _, v := range lum.data {
		if v < 0 {
			v = 0
		}
		sumLog += math.Log(v + eps)
		if v > maxL {
			maxL = v
		}
	}
	logAvg = math.Exp(sumLog / float64(len(lum.data)))
	return
}

// applyLuminanceMap rebuilds an 8-bit image from a new luminance plane ld,
// re-introducing colour from the original radiance via a per-channel ratio
// raised to the saturation power, then applying the display gamma. For a
// single-channel image it simply gamma-encodes the mapped luminance.
func applyLuminanceMap(r *Radiance, oldLum, ld *plane, saturation, gamma float64) *cv.Mat {
	out := cv.NewMat(r.Rows, r.Cols, r.Channels)
	invGamma := 1.0
	if gamma > 0 {
		invGamma = 1.0 / gamma
	}
	total := r.Rows * r.Cols
	for p := 0; p < total; p++ {
		l := oldLum.data[p]
		nd := ld.data[p]
		if r.Channels == 1 {
			out.Data[p] = clamp8(math.Pow(clamp01(nd), invGamma) * 255)
			continue
		}
		for c := 0; c < r.Channels; c++ {
			val := r.Data[p*r.Channels+c]
			var ratio float64
			if l > 1e-9 {
				ratio = val / l
			}
			mapped := math.Pow(ratio, saturation) * nd
			out.Data[p*r.Channels+c] = clamp8(math.Pow(clamp01(mapped), invGamma) * 255)
		}
	}
	return out
}

// normalizeGamma applies per-channel min/max normalisation followed by a gamma
// curve — the behaviour of the plain gamma tonemapper.
func normalizeGamma(r *Radiance, gamma float64) *cv.Mat {
	out := cv.NewMat(r.Rows, r.Cols, r.Channels)
	// Global min/max over all channels keeps colours balanced.
	minV, maxV := math.Inf(1), math.Inf(-1)
	for _, v := range r.Data {
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	rng := maxV - minV
	if rng <= 0 {
		rng = 1
	}
	invGamma := 1.0
	if gamma > 0 {
		invGamma = 1.0 / gamma
	}
	for i, v := range r.Data {
		n := (v - minV) / rng
		out.Data[i] = clamp8(math.Pow(clamp01(n), invGamma) * 255)
	}
	return out
}

// TonemapGamma is the simplest operator: it linearly normalises the radiance to
// [0,1] and applies a gamma curve. Gamma greater than 1 brightens midtones;
// gamma of 1 is a plain linear stretch.
type TonemapGamma struct {
	// Gamma is the display gamma exponent; non-positive is treated as 1.
	Gamma float64
}

// NewTonemapGamma returns a gamma tonemapper. Pass a non-positive gamma for a
// linear (gamma 1) stretch.
func NewTonemapGamma(gamma float64) *TonemapGamma {
	if gamma <= 0 {
		gamma = 1
	}
	return &TonemapGamma{Gamma: gamma}
}

// Process implements [Tonemap].
func (t *TonemapGamma) Process(r *Radiance) *cv.Mat {
	return normalizeGamma(r, t.Gamma)
}

// TonemapReinhard implements Reinhard et al. (2002) photographic tone
// reproduction. The global operator maps luminance L to L/(1+L) after scaling
// by a key derived from Intensity; enabling Local adds a dodge-and-burn term
// using a Gaussian surround so local contrast is preserved.
type TonemapReinhard struct {
	// Gamma is the display gamma; non-positive is treated as 1.
	Gamma float64
	// Intensity shifts the overall key (exposure): larger is brighter. A value
	// of 0 corresponds to the classic 0.18 middle-grey key.
	Intensity float64
	// Saturation controls colour vividness when re-introducing chroma (1 keeps
	// the original saturation).
	Saturation float64
	// Local enables the local (surround-based) operator.
	Local bool
	// LocalSigma is the Gaussian surround radius (in pixels) for the local
	// operator; non-positive selects a default.
	LocalSigma float64
}

// NewTonemapReinhard returns a Reinhard tonemapper with sensible defaults
// (global operator, gamma 2.2, unit saturation).
func NewTonemapReinhard() *TonemapReinhard {
	return &TonemapReinhard{Gamma: 2.2, Intensity: 0, Saturation: 1, Local: false, LocalSigma: 0}
}

// Process implements [Tonemap].
func (t *TonemapReinhard) Process(r *Radiance) *cv.Mat {
	lum := r.luminance()
	logAvg, _ := logAvgLuminance(lum)
	// Key: 0.18 middle-grey scaled by the intensity exposure control.
	key := 0.18 * math.Exp(t.Intensity)
	if logAvg <= 0 {
		logAvg = 1e-6
	}
	scale := key / logAvg

	scaled := newPlane(lum.rows, lum.cols)
	for i, v := range lum.data {
		if v < 0 {
			v = 0
		}
		scaled.data[i] = v * scale
	}

	ld := newPlane(lum.rows, lum.cols)
	if t.Local {
		sigma := t.LocalSigma
		if sigma <= 0 {
			sigma = float64(minInt(lum.rows, lum.cols)) / 20.0
			if sigma < 1 {
				sigma = 1
			}
		}
		surround := scaled.blur(sigma)
		for i := range ld.data {
			ld.data[i] = scaled.data[i] / (1 + surround.data[i])
		}
	} else {
		for i := range ld.data {
			ld.data[i] = scaled.data[i] / (1 + scaled.data[i])
		}
	}

	sat := t.Saturation
	if sat <= 0 {
		sat = 1
	}
	return applyLuminanceMap(r, lum, ld, sat, gammaOrOne(t.Gamma))
}

// TonemapDrago implements Drago et al. (2003) adaptive logarithmic mapping. The
// Bias parameter (0,1) controls how the logarithm base varies with luminance:
// smaller values darken, the paper's default is 0.85.
type TonemapDrago struct {
	// Gamma is the display gamma; non-positive is treated as 1.
	Gamma float64
	// Bias shapes the adaptive logarithm base; clamped to (0,1), default 0.85.
	Bias float64
	// Saturation controls colour vividness (1 keeps the original).
	Saturation float64
}

// NewTonemapDrago returns a Drago tonemapper with the paper's default
// parameters (gamma 2.2, bias 0.85, unit saturation).
func NewTonemapDrago() *TonemapDrago {
	return &TonemapDrago{Gamma: 2.2, Bias: 0.85, Saturation: 1}
}

// Process implements [Tonemap].
func (t *TonemapDrago) Process(r *Radiance) *cv.Mat {
	lum := r.luminance()
	logAvg, maxL := logAvgLuminance(lum)
	// Normalise luminance by the log-average so the key is scene-independent.
	if logAvg <= 0 {
		logAvg = 1e-6
	}
	lwMax := maxL / logAvg
	bias := t.Bias
	if bias <= 0 || bias >= 1 {
		bias = 0.85
	}
	logBias := math.Log(bias) / math.Log(0.5)
	denom := math.Log10(1 + lwMax)
	if denom <= 0 {
		denom = 1e-6
	}
	ld := newPlane(lum.rows, lum.cols)
	for i, v := range lum.data {
		if v < 0 {
			v = 0
		}
		lw := v / logAvg
		var ratio float64
		if lwMax > 0 {
			ratio = lw / lwMax
		}
		// Drago's interpolated logarithm base.
		base := 2 + 8*math.Pow(ratio, logBias)
		ld.data[i] = (math.Log(1+lw) / math.Log(base)) / denom
	}
	sat := t.Saturation
	if sat <= 0 {
		sat = 1
	}
	return applyLuminanceMap(r, lum, ld, sat, gammaOrOne(t.Gamma))
}

// TonemapMantiuk is a local-contrast approximation of Mantiuk et al. (2006).
// The true operator scales luminance gradients in a multi-resolution transform
// and reconstructs the image by solving a Poisson equation; this
// implementation instead compresses the log-luminance deviation from a Gaussian
// local mean by the Scale factor, which reproduces the characteristic
// contrast-equalising look without the gradient-domain solve. See the package
// Deferred notes.
type TonemapMantiuk struct {
	// Gamma is the display gamma; non-positive is treated as 1.
	Gamma float64
	// Scale is the contrast compression factor in (0,1]; smaller compresses
	// harder. Default 0.7.
	Scale float64
	// Saturation controls colour vividness (1 keeps the original).
	Saturation float64
}

// NewTonemapMantiuk returns a Mantiuk-style tonemapper with default parameters
// (gamma 2.2, scale 0.7, unit saturation).
func NewTonemapMantiuk() *TonemapMantiuk {
	return &TonemapMantiuk{Gamma: 2.2, Scale: 0.7, Saturation: 1}
}

// Process implements [Tonemap].
func (t *TonemapMantiuk) Process(r *Radiance) *cv.Mat {
	lum := r.luminance()
	const eps = 1e-6
	// Work in the log domain.
	logL := newPlane(lum.rows, lum.cols)
	for i, v := range lum.data {
		if v < 0 {
			v = 0
		}
		logL.data[i] = math.Log(v + eps)
	}
	sigma := float64(minInt(lum.rows, lum.cols)) / 12.0
	if sigma < 1 {
		sigma = 1
	}
	localMean := logL.blur(sigma)
	scale := t.Scale
	if scale <= 0 || scale > 1 {
		scale = 0.7
	}
	// Compress deviation from the local mean, then renormalise to [0,1].
	comp := newPlane(lum.rows, lum.cols)
	minV, maxV := math.Inf(1), math.Inf(-1)
	for i := range comp.data {
		v := localMean.data[i] + scale*(logL.data[i]-localMean.data[i])
		comp.data[i] = v
		if v < minV {
			minV = v
		}
		if v > maxV {
			maxV = v
		}
	}
	rng := maxV - minV
	if rng <= 0 {
		rng = 1
	}
	ld := newPlane(lum.rows, lum.cols)
	for i := range ld.data {
		// Back to linear display luminance in [0,1].
		ld.data[i] = (comp.data[i] - minV) / rng
	}
	sat := t.Saturation
	if sat <= 0 {
		sat = 1
	}
	return applyLuminanceMap(r, lum, ld, sat, gammaOrOne(t.Gamma))
}

func gammaOrOne(g float64) float64 {
	if g <= 0 {
		return 1
	}
	return g
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
