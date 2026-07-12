package dnn_superres

import (
	"fmt"
	"math"

	cv "github.com/malcolmston/opencv"
)

// validateAnyScale checks a source image and an arbitrary integer scale factor
// used by the extended upsamplers in this package. Unlike [validate], which
// mirrors the trained-model constraint of scales 2, 3 and 4, this accepts any
// integer scale of 2 or more, so the classical resamplers here can enlarge by
// ×5, ×8, ×16 and beyond.
func validateAnyScale(src *cv.Mat, scale int) error {
	if src == nil || src.Empty() {
		return fmt.Errorf("dnn_superres: source image is empty")
	}
	if scale < 2 {
		return fmt.Errorf("dnn_superres: unsupported scale %d (want >= 2)", scale)
	}
	return nil
}

// mitchellNetravali is the Mitchell–Netravali cubic reconstruction kernel with
// the classic B = C = 1/3 parameters (support radius 2). It is the balanced
// compromise between the blur of a pure cubic B-spline and the ringing of
// Catmull-Rom, and is a popular default for photographic enlargement.
func mitchellNetravali(t float64) float64 {
	const b, c = 1.0 / 3.0, 1.0 / 3.0
	t = math.Abs(t)
	t2 := t * t
	t3 := t2 * t
	switch {
	case t < 1:
		return ((12-9*b-6*c)*t3 + (-18+12*b+6*c)*t2 + (6 - 2*b)) / 6
	case t < 2:
		return ((-b-6*c)*t3 + (6*b+30*c)*t2 + (-12*b-48*c)*t + (8*b + 24*c)) / 6
	default:
		return 0
	}
}

// cubicBSpline is the cubic B-spline kernel (support radius 2). As an
// approximating (not interpolating) kernel it never overshoots, giving the
// smoothest, ring-free enlargement of the cubic family at the cost of some
// softness. It is the B = 1, C = 0 corner of the Mitchell–Netravali family.
func cubicBSpline(t float64) float64 {
	t = math.Abs(t)
	switch {
	case t < 1:
		return (4 - 6*t*t + 3*t*t*t) / 6
	case t < 2:
		u := 2 - t
		return u * u * u / 6
	default:
		return 0
	}
}

// hermite is the cubic Hermite / smoothstep reconstruction kernel (support
// radius 1). It is the a = 0 member of the Keys cubic family: a fast two-tap
// cubic that is slightly crisper than bilinear with no overshoot.
func hermite(t float64) float64 {
	t = math.Abs(t)
	if t < 1 {
		return (2*t-3)*t*t + 1
	}
	return 0
}

// lanczos3 is the Lanczos windowed-sinc kernel with parameter a = 3 (support
// radius 3), a 6-tap resampler. It is a middle ground between the extra
// sharpness (and stronger ringing) of Lanczos-4 and the softness of bicubic.
func lanczos3(t float64) float64 {
	const a = 3.0
	if t <= -a || t >= a {
		return 0
	}
	return sinc(t) * sinc(t/a)
}

// gaussianKernel is a truncated Gaussian reconstruction kernel with sigma
// chosen so the support radius 2 captures essentially all of its mass. It gives
// a very smooth, low-alias enlargement with no ringing whatsoever, useful as an
// anti-alias-friendly baseline.
func gaussianKernel(t float64) float64 {
	const sigma = 0.9
	if t <= -2 || t >= 2 {
		return 0
	}
	return math.Exp(-(t * t) / (2 * sigma * sigma))
}

// UpsampleMitchell enlarges src by an arbitrary integer scale (>= 2) using the
// Mitchell–Netravali cubic (B = C = 1/3), a balanced photographic enlargement
// kernel that trades a little of Catmull-Rom's ringing for smoother tones. It
// returns an error for an empty image or a scale below 2.
func UpsampleMitchell(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validateAnyScale(src, scale); err != nil {
		return nil, err
	}
	return resampleSeparable(src, src.Cols*scale, src.Rows*scale, mitchellNetravali, 2), nil
}

// UpsampleBSpline enlarges src by an arbitrary integer scale (>= 2) using the
// cubic B-spline kernel, the smoothest, overshoot-free member of the cubic
// family. It is ideal when ringing must be avoided entirely, at the cost of some
// softness. It returns an error for an empty image or a scale below 2.
func UpsampleBSpline(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validateAnyScale(src, scale); err != nil {
		return nil, err
	}
	return resampleSeparable(src, src.Cols*scale, src.Rows*scale, cubicBSpline, 2), nil
}

// UpsampleHermite enlarges src by an arbitrary integer scale (>= 2) using the
// two-tap cubic Hermite kernel, a fast smoothstep resample slightly crisper than
// bilinear with no overshoot. It returns an error for an empty image or a scale
// below 2.
func UpsampleHermite(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validateAnyScale(src, scale); err != nil {
		return nil, err
	}
	return resampleSeparable(src, src.Cols*scale, src.Rows*scale, hermite, 1), nil
}

// UpsampleLanczos3 enlarges src by an arbitrary integer scale (>= 2) using
// Lanczos-3 windowed-sinc interpolation (6-tap separable), a sharpness/ringing
// compromise between bicubic and Lanczos-4. It returns an error for an empty
// image or a scale below 2.
func UpsampleLanczos3(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validateAnyScale(src, scale); err != nil {
		return nil, err
	}
	return resampleSeparable(src, src.Cols*scale, src.Rows*scale, lanczos3, 3), nil
}

// UpsampleGaussian enlarges src by an arbitrary integer scale (>= 2) using a
// truncated Gaussian reconstruction kernel. It produces the smoothest,
// ring-free, low-alias enlargement of the methods here and is a good baseline
// when subsequent processing prefers a clean, band-limited input. It returns an
// error for an empty image or a scale below 2.
func UpsampleGaussian(src *cv.Mat, scale int) (*cv.Mat, error) {
	if err := validateAnyScale(src, scale); err != nil {
		return nil, err
	}
	return resampleSeparable(src, src.Cols*scale, src.Rows*scale, gaussianKernel, 2), nil
}
