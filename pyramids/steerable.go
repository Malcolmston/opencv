package pyramids

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SteerableBasisG1 computes the two separable first-derivative-of-Gaussian
// basis responses of f at scale sigma: gx is the derivative along x (smoothed
// along y) and gy is the derivative along y (smoothed along x). Because the
// first derivative of a Gaussian is steerable, the oriented response at any
// angle theta is cos(theta)*gx + sin(theta)*gy (see [SteerG1]). It panics if
// sigma is not positive.
func SteerableBasisG1(f *cv.FloatMat, sigma float64) (gx, gy *cv.FloatMat) {
	pyramidsRequire(f, "SteerableBasisG1")
	smooth := GaussianKernel(sigma)
	deriv := GaussianDerivativeKernel(sigma)
	gx = ConvolveSeparable(f, deriv, smooth)
	gy = ConvolveSeparable(f, smooth, deriv)
	return gx, gy
}

// SteerG1 evaluates the first-derivative steerable filter at orientation theta
// (radians, measured from the +x axis) from the basis responses produced by
// [SteerableBasisG1]: out = cos(theta)*gx + sin(theta)*gy. The result is the
// directional derivative of the smoothed image along theta. gx and gy must
// share dimensions.
func SteerG1(gx, gy *cv.FloatMat, theta float64) *cv.FloatMat {
	pyramidsRequire(gx, "SteerG1")
	pyramidsRequire(gy, "SteerG1")
	pyramidsSameSize(gx, gy, "SteerG1")
	c, s := math.Cos(theta), math.Sin(theta)
	out := cv.NewFloatMat(gx.Rows, gx.Cols)
	for i := range out.Data {
		out.Data[i] = c*gx.Data[i] + s*gy.Data[i]
	}
	return out
}

// SteerableBasisG2 computes the three separable second-derivative-of-Gaussian
// basis responses of f at scale sigma: gxx (second derivative along x), gxy
// (mixed partial), and gyy (second derivative along y). The second derivative
// of a Gaussian is steerable from these three bases (Freeman & Adelson); use
// [SteerG2] to synthesise an arbitrary orientation. It panics if sigma is not
// positive.
func SteerableBasisG2(f *cv.FloatMat, sigma float64) (gxx, gxy, gyy *cv.FloatMat) {
	pyramidsRequire(f, "SteerableBasisG2")
	smooth := GaussianKernel(sigma)
	deriv := GaussianDerivativeKernel(sigma)
	deriv2 := GaussianSecondDerivativeKernel(sigma)
	gxx = ConvolveSeparable(f, deriv2, smooth)
	gyy = ConvolveSeparable(f, smooth, deriv2)
	gxy = ConvolveSeparable(f, deriv, deriv)
	return gxx, gxy, gyy
}

// SteerG2 evaluates the second-derivative steerable filter at orientation theta
// (radians) from the basis responses of [SteerableBasisG2] using the
// interpolation weights cos^2(theta), -2*cos(theta)*sin(theta) and
// sin^2(theta): out = cos^2 * gxx - 2*cos*sin * gxy + sin^2 * gyy. All three
// bases must share dimensions.
func SteerG2(gxx, gxy, gyy *cv.FloatMat, theta float64) *cv.FloatMat {
	pyramidsRequire(gxx, "SteerG2")
	pyramidsRequire(gxy, "SteerG2")
	pyramidsRequire(gyy, "SteerG2")
	pyramidsSameSize(gxx, gxy, "SteerG2")
	pyramidsSameSize(gxx, gyy, "SteerG2")
	c, s := math.Cos(theta), math.Sin(theta)
	ka, kb, kc := c*c, -2*c*s, s*s
	out := cv.NewFloatMat(gxx.Rows, gxx.Cols)
	for i := range out.Data {
		out.Data[i] = ka*gxx.Data[i] + kb*gxy.Data[i] + kc*gyy.Data[i]
	}
	return out
}

// OrientationEnergy returns the gradient magnitude sqrt(gx^2 + gy^2) of the two
// first-derivative basis responses, a rotation-invariant measure of local edge
// strength. gx and gy must share dimensions.
func OrientationEnergy(gx, gy *cv.FloatMat) *cv.FloatMat {
	pyramidsRequire(gx, "OrientationEnergy")
	pyramidsRequire(gy, "OrientationEnergy")
	pyramidsSameSize(gx, gy, "OrientationEnergy")
	out := cv.NewFloatMat(gx.Rows, gx.Cols)
	for i := range out.Data {
		out.Data[i] = math.Hypot(gx.Data[i], gy.Data[i])
	}
	return out
}

// OrientationMap returns the per-pixel gradient orientation atan2(gy, gx) in
// radians (range (-pi, pi]) from the two first-derivative basis responses. gx
// and gy must share dimensions.
func OrientationMap(gx, gy *cv.FloatMat) *cv.FloatMat {
	pyramidsRequire(gx, "OrientationMap")
	pyramidsRequire(gy, "OrientationMap")
	pyramidsSameSize(gx, gy, "OrientationMap")
	out := cv.NewFloatMat(gx.Rows, gx.Cols)
	for i := range out.Data {
		out.Data[i] = math.Atan2(gy.Data[i], gx.Data[i])
	}
	return out
}

// SteerableLevel holds the oriented band-pass responses at one octave of a
// steerable pyramid: one image per analysis orientation, plus the orientations
// themselves in radians.
type SteerableLevel struct {
	// Bands holds one oriented response image per orientation.
	Bands []*cv.FloatMat
	// Orientations holds the analysis angle (radians) of each band.
	Orientations []float64
}

// SteerablePyramid is a multi-scale, multi-orientation decomposition. Each
// level is a band-pass image (a difference of Gaussians between octaves)
// analysed into a fixed set of orientations by steering the first-derivative
// basis. Residual holds the coarsest low-pass image left over after the last
// octave. This is an analysis-oriented (energy) pyramid; it is not designed for
// exact self-inverting reconstruction.
type SteerablePyramid struct {
	// Levels holds the oriented band-pass responses from finest to coarsest.
	Levels []SteerableLevel
	// Residual is the coarsest low-pass image (at the smallest resolution).
	Residual *cv.FloatMat
}

// BuildSteerablePyramid decomposes f into levels octaves, each analysed into
// the given number of orientations equally spaced over [0, pi). At each octave
// a band-pass image is formed as the current low-pass minus a further-blurred
// copy, its first-derivative steerable basis is computed, and the oriented
// responses are steered at angles k*pi/orientations. The low-pass is then
// reduced (halved) for the next octave. It panics if levels < 1 or
// orientations < 1.
func BuildSteerablePyramid(f *cv.FloatMat, levels, orientations int) *SteerablePyramid {
	pyramidsRequire(f, "BuildSteerablePyramid")
	if levels < 1 {
		panic("pyramids: BuildSteerablePyramid: levels must be >= 1")
	}
	if orientations < 1 {
		panic("pyramids: BuildSteerablePyramid: orientations must be >= 1")
	}
	sp := &SteerablePyramid{Levels: make([]SteerableLevel, 0, levels)}
	lowpass := CloneFloat(f)
	for l := 0; l < levels; l++ {
		blurred := GaussianBlurFloat(lowpass, 1.0)
		band := SubtractFloat(lowpass, blurred)
		gx, gy := SteerableBasisG1(band, 1.0)
		lvl := SteerableLevel{
			Bands:        make([]*cv.FloatMat, orientations),
			Orientations: make([]float64, orientations),
		}
		for o := 0; o < orientations; o++ {
			theta := math.Pi * float64(o) / float64(orientations)
			lvl.Bands[o] = SteerG1(gx, gy, theta)
			lvl.Orientations[o] = theta
		}
		sp.Levels = append(sp.Levels, lvl)
		if blurred.Rows <= 1 && blurred.Cols <= 1 {
			lowpass = blurred
			break
		}
		lowpass = PyrDownFloat(blurred)
	}
	sp.Residual = lowpass
	return sp
}

// NumLevels returns the number of octaves in the steerable pyramid.
func (sp *SteerablePyramid) NumLevels() int { return len(sp.Levels) }

// Orientations returns the number of orientation bands per level (0 if the
// pyramid is empty).
func (sp *SteerablePyramid) Orientations() int {
	if len(sp.Levels) == 0 {
		return 0
	}
	return len(sp.Levels[0].Bands)
}
