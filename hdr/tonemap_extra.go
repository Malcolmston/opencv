package hdr

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// TonemapDurand implements Durand & Dorsey's (2002) fast bilateral-filtering
// tone reproduction, the operator OpenCV exposes as createTonemapDurand. The
// log-luminance is split by a bilateral filter into a piecewise-smooth base
// layer and a detail layer; only the base layer's contrast is compressed, so
// edges and fine texture survive intact. Colour is re-introduced from the
// original radiance and a display gamma is applied.
type TonemapDurand struct {
	// Gamma is the display gamma; non-positive is treated as 1.
	Gamma float64
	// Contrast is the target dynamic range of the compressed base layer (the
	// ratio between its brightest and darkest values). Larger keeps more global
	// contrast. Non-positive selects 4.
	Contrast float64
	// SigmaSpace is the bilateral spatial radius in pixels; non-positive scales
	// with the image size.
	SigmaSpace float64
	// SigmaColor is the bilateral range sigma in log-luminance units;
	// non-positive selects 0.4.
	SigmaColor float64
	// Saturation controls colour vividness (1 keeps the original).
	Saturation float64
}

// NewTonemapDurand returns a Durand tonemapper with sensible defaults (gamma
// 2.2, target contrast 4, range sigma 0.4, unit saturation, automatic spatial
// sigma).
func NewTonemapDurand() *TonemapDurand {
	return &TonemapDurand{Gamma: 2.2, Contrast: 4, SigmaSpace: 0, SigmaColor: 0.4, Saturation: 1}
}

// Process implements [Tonemap].
func (t *TonemapDurand) Process(r *Radiance) *cv.Mat {
	const eps = 1e-6
	lum := r.luminance()
	logL := newPlane(lum.rows, lum.cols)
	for i, v := range lum.data {
		if v < 0 {
			v = 0
		}
		logL.data[i] = math.Log(v + eps)
	}

	sigmaS := t.SigmaSpace
	if sigmaS <= 0 {
		sigmaS = float64(minInt(lum.rows, lum.cols)) / 16.0
		if sigmaS < 1 {
			sigmaS = 1
		}
	}
	sigmaC := t.SigmaColor
	if sigmaC <= 0 {
		sigmaC = 0.4
	}
	base := bilateralPlane(logL, sigmaS, sigmaC)

	// Range of the base layer drives the compression factor.
	minB, maxB := math.Inf(1), math.Inf(-1)
	for _, v := range base.data {
		if v < minB {
			minB = v
		}
		if v > maxB {
			maxB = v
		}
	}
	contrast := t.Contrast
	if contrast <= 0 {
		contrast = 4
	}
	rng := maxB - minB
	if rng <= 0 {
		rng = 1
	}
	// Compression maps the base range onto log(contrast); detail is preserved.
	cf := math.Log(contrast) / rng

	ld := newPlane(lum.rows, lum.cols)
	for i := range ld.data {
		detail := logL.data[i] - base.data[i]
		outLog := (base.data[i]-maxB)*cf + detail
		ld.data[i] = math.Exp(outLog)
	}
	sat := t.Saturation
	if sat <= 0 {
		sat = 1
	}
	return applyLuminanceMap(r, lum, ld, sat, gammaOrOne(t.Gamma))
}

// TonemapMantiukGradient is a gradient-domain HDR compressor in the spirit of
// Fattal et al. (2002) and Mantiuk et al. (2006). Unlike the local-contrast
// [TonemapMantiuk], it operates on the gradient field of the log-luminance:
// large gradients (high-contrast edges) are attenuated more than small ones,
// and the image is reconstructed by solving the resulting Poisson equation
// ∇²I = div(G') with Gauss–Seidel iteration. This genuinely equalises contrast
// across scales rather than approximating it with a single Gaussian surround.
type TonemapMantiukGradient struct {
	// Gamma is the display gamma; non-positive is treated as 1.
	Gamma float64
	// Beta is the gradient attenuation exponent in (0,1]; smaller compresses
	// high-contrast edges harder. Non-positive or >1 selects 0.85.
	Beta float64
	// Saturation controls colour vividness (1 keeps the original).
	Saturation float64
	// Iterations is the number of Gauss–Seidel sweeps of the Poisson solve;
	// non-positive selects 400.
	Iterations int
}

// NewTonemapMantiukGradient returns a gradient-domain tonemapper with default
// parameters (gamma 2.2, beta 0.85, unit saturation, 400 solver iterations).
func NewTonemapMantiukGradient() *TonemapMantiukGradient {
	return &TonemapMantiukGradient{Gamma: 2.2, Beta: 0.85, Saturation: 1, Iterations: 400}
}

// Process implements [Tonemap].
func (t *TonemapMantiukGradient) Process(r *Radiance) *cv.Mat {
	const eps = 1e-6
	rows, cols := r.Rows, r.Cols
	lum := r.luminance()
	logL := newPlane(rows, cols)
	for i, v := range lum.data {
		if v < 0 {
			v = 0
		}
		logL.data[i] = math.Log(v + eps)
	}

	// Forward gradients (zero on the trailing edges).
	gx := newPlane(rows, cols)
	gy := newPlane(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			if x < cols-1 {
				gx.set(y, x, logL.at(y, x+1)-logL.at(y, x))
			}
			if y < rows-1 {
				gy.set(y, x, logL.at(y+1, x)-logL.at(y, x))
			}
		}
	}

	// Attenuation scale alpha from the mean gradient magnitude (Fattal).
	var sumMag float64
	n := rows * cols
	for i := 0; i < n; i++ {
		sumMag += math.Hypot(gx.data[i], gy.data[i])
	}
	alpha := 0.1 * (sumMag / float64(n))
	if alpha <= 0 {
		alpha = 1
	}
	beta := t.Beta
	if beta <= 0 || beta > 1 {
		beta = 0.85
	}

	// Attenuate each gradient: phi = (mag/alpha)^(beta-1), so larger gradients
	// are scaled down when beta < 1.
	for i := 0; i < n; i++ {
		mag := math.Hypot(gx.data[i], gy.data[i]) + eps
		phi := math.Pow(mag/alpha, beta-1)
		gx.data[i] *= phi
		gy.data[i] *= phi
	}

	// Divergence of the attenuated field: div = dGx/dx + dGy/dy (backward diff).
	div := newPlane(rows, cols)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			var d float64
			d += gx.at(y, x)
			if x > 0 {
				d -= gx.at(y, x-1)
			}
			d += gy.at(y, x)
			if y > 0 {
				d -= gy.at(y-1, x)
			}
			div.set(y, x, d)
		}
	}

	// Solve ∇²I = div with Gauss–Seidel and Neumann (reflect) boundaries.
	iters := t.Iterations
	if iters <= 0 {
		iters = 400
	}
	img := newPlane(rows, cols)
	for it := 0; it < iters; it++ {
		for y := 0; y < rows; y++ {
			for x := 0; x < cols; x++ {
				s := img.atReflect(y, x-1) + img.atReflect(y, x+1) +
					img.atReflect(y-1, x) + img.atReflect(y+1, x)
				img.set(y, x, (s-div.at(y, x))/4)
			}
		}
	}

	// Normalise the reconstructed log field to [0,1] display luminance.
	minV, maxV := math.Inf(1), math.Inf(-1)
	for _, v := range img.data {
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
	ld := newPlane(rows, cols)
	for i := range ld.data {
		ld.data[i] = (img.data[i] - minV) / rng
	}
	sat := t.Saturation
	if sat <= 0 {
		sat = 1
	}
	return applyLuminanceMap(r, lum, ld, sat, gammaOrOne(t.Gamma))
}
