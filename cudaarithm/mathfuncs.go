package cudaarithm

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Pow raises each sample of src to power and returns the rounded, saturated
// result, mirroring cv::cuda::pow. The computation is on the raw 0..255 sample
// value, so with power > 1 all but the smallest samples saturate to 255; with a
// fractional power such as 0.5 the result stays in range. A negative base with a
// non-integer power yields 0.
func Pow(src *GpuMat, power float64, _ ...*Stream) *GpuMat {
	requireNonEmpty(src, "Pow")
	out := cv.NewMat(src.mat.Rows, src.mat.Cols, src.mat.Channels)
	for i, s := range src.mat.Data {
		v := math.Pow(float64(s), power)
		if math.IsNaN(v) || math.IsInf(v, 0) {
			v = 0
		}
		out.Data[i] = roundToUint8(v)
	}
	return wrap(out)
}

// Exp returns e raised to each sample, rounded and saturated, mirroring
// cv::cuda::exp. Because exp grows quickly, samples above ~5 saturate to 255.
func Exp(src *GpuMat, _ ...*Stream) *GpuMat {
	requireNonEmpty(src, "Exp")
	out := cv.NewMat(src.mat.Rows, src.mat.Cols, src.mat.Channels)
	for i, s := range src.mat.Data {
		out.Data[i] = roundToUint8(math.Exp(float64(s)))
	}
	return wrap(out)
}

// Log returns the natural logarithm of each sample, rounded and saturated,
// mirroring cv::cuda::log. A zero sample, whose true log is -inf, maps to 0.
func Log(src *GpuMat, _ ...*Stream) *GpuMat {
	requireNonEmpty(src, "Log")
	out := cv.NewMat(src.mat.Rows, src.mat.Cols, src.mat.Channels)
	for i, s := range src.mat.Data {
		if s == 0 {
			out.Data[i] = 0
			continue
		}
		out.Data[i] = roundToUint8(math.Log(float64(s)))
	}
	return wrap(out)
}

// Sqrt returns the square root of each sample, rounded and saturated, mirroring
// cv::cuda::sqrt. Results always fit in a byte (sqrt(255) < 16).
func Sqrt(src *GpuMat, _ ...*Stream) *GpuMat {
	requireNonEmpty(src, "Sqrt")
	out := cv.NewMat(src.mat.Rows, src.mat.Cols, src.mat.Channels)
	for i, s := range src.mat.Data {
		out.Data[i] = roundToUint8(math.Sqrt(float64(s)))
	}
	return wrap(out)
}

// Magnitude returns the per-sample vector magnitude sqrt(x^2 + y^2), rounded and
// saturated, mirroring cv::cuda::magnitude. Inputs must share a shape. Because
// the result is a byte, magnitudes above 255 saturate.
func Magnitude(x, y *GpuMat, _ ...*Stream) *GpuMat {
	requireSameShape(x, y, "Magnitude")
	out := cv.NewMat(x.mat.Rows, x.mat.Cols, x.mat.Channels)
	for i := range x.mat.Data {
		xv := float64(x.mat.Data[i])
		yv := float64(y.mat.Data[i])
		out.Data[i] = roundToUint8(math.Hypot(xv, yv))
	}
	return wrap(out)
}

// Phase returns the per-sample orientation atan2(y, x) of the vectors, mirroring
// cv::cuda::phase. When angleInDegrees is false the angle is in radians (0..2π,
// so it saturates to a byte above 6); when true it is in degrees (0..360, so it
// saturates above 255). The result is rounded and saturated. Inputs must share a
// shape.
func Phase(x, y *GpuMat, angleInDegrees bool, _ ...*Stream) *GpuMat {
	requireSameShape(x, y, "Phase")
	out := cv.NewMat(x.mat.Rows, x.mat.Cols, x.mat.Channels)
	for i := range x.mat.Data {
		out.Data[i] = roundToUint8(angleOf(float64(x.mat.Data[i]), float64(y.mat.Data[i]), angleInDegrees))
	}
	return wrap(out)
}

// CartToPolar converts the Cartesian vectors (x, y) to polar form, returning the
// magnitude and angle GpuMats, mirroring cv::cuda::cartToPolar. See [Magnitude]
// and [Phase] for the saturation behaviour of each output. Inputs must share a
// shape.
func CartToPolar(x, y *GpuMat, angleInDegrees bool, _ ...*Stream) (magnitude, angle *GpuMat) {
	requireSameShape(x, y, "CartToPolar")
	mag := cv.NewMat(x.mat.Rows, x.mat.Cols, x.mat.Channels)
	ang := cv.NewMat(x.mat.Rows, x.mat.Cols, x.mat.Channels)
	for i := range x.mat.Data {
		xv := float64(x.mat.Data[i])
		yv := float64(y.mat.Data[i])
		mag.Data[i] = roundToUint8(math.Hypot(xv, yv))
		ang.Data[i] = roundToUint8(angleOf(xv, yv, angleInDegrees))
	}
	return wrap(mag), wrap(ang)
}

// PolarToCart converts the polar vectors (magnitude, angle) back to Cartesian
// form, returning the x and y GpuMats, mirroring cv::cuda::polarToCart. angle is
// interpreted in degrees when angleInDegrees is true, otherwise radians. Outputs
// are rounded and saturated into [0,255], so components of vectors pointing into
// negative quadrants clamp to 0. Inputs must share a shape.
func PolarToCart(magnitude, angle *GpuMat, angleInDegrees bool, _ ...*Stream) (x, y *GpuMat) {
	requireSameShape(magnitude, angle, "PolarToCart")
	xm := cv.NewMat(magnitude.mat.Rows, magnitude.mat.Cols, magnitude.mat.Channels)
	ym := cv.NewMat(magnitude.mat.Rows, magnitude.mat.Cols, magnitude.mat.Channels)
	for i := range magnitude.mat.Data {
		m := float64(magnitude.mat.Data[i])
		a := float64(angle.mat.Data[i])
		if angleInDegrees {
			a *= math.Pi / 180
		}
		xm.Data[i] = roundToUint8(m * math.Cos(a))
		ym.Data[i] = roundToUint8(m * math.Sin(a))
	}
	return wrap(xm), wrap(ym)
}

// angleOf returns atan2(y, x) mapped to [0, 2π) radians, or the equivalent in
// degrees when deg is true.
func angleOf(x, y float64, deg bool) float64 {
	a := math.Atan2(y, x)
	if a < 0 {
		a += 2 * math.Pi
	}
	if deg {
		a *= 180 / math.Pi
	}
	return a
}
