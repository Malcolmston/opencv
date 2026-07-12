package cv

import (
	"fmt"
	"math"
)

// Integral computes the summed-area table (integral image) of a single-channel
// Mat. The result has one extra row and column of leading zeros, so element
// (y+1, x+1) equals the sum of every source sample in the rectangle
// [0,y]×[0,x]; the sum over any axis-aligned rectangle can then be found in
// constant time. It panics if src is not single-channel. This mirrors OpenCV's
// cv::integral.
func Integral(src *Mat) *FloatMat {
	requireChannels(src, 1, "Integral")
	rows, cols := src.Rows, src.Cols
	out := NewFloatMat(rows+1, cols+1)
	w := cols + 1
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := float64(src.Data[y*cols+x])
			out.Data[(y+1)*w+(x+1)] = v +
				out.Data[y*w+(x+1)] +
				out.Data[(y+1)*w+x] -
				out.Data[y*w+x]
		}
	}
	return out
}

// IntegralSquared computes the summed-area table of the squared sample values
// of a single-channel Mat, laid out like [Integral] with a leading zero row and
// column. Together with [Integral] it enables constant-time variance queries. It
// panics if src is not single-channel.
func IntegralSquared(src *Mat) *FloatMat {
	requireChannels(src, 1, "IntegralSquared")
	rows, cols := src.Rows, src.Cols
	out := NewFloatMat(rows+1, cols+1)
	w := cols + 1
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			v := float64(src.Data[y*cols+x])
			out.Data[(y+1)*w+(x+1)] = v*v +
				out.Data[y*w+(x+1)] +
				out.Data[(y+1)*w+x] -
				out.Data[y*w+x]
		}
	}
	return out
}

// Accumulate adds a single-channel image to an accumulator FloatMat in place
// (dst += src). It panics on a size mismatch or a multi-channel source. This is
// OpenCV's cv::accumulate.
func Accumulate(src *Mat, dst *FloatMat) {
	requireChannels(src, 1, "Accumulate")
	if src.Rows != dst.Rows || src.Cols != dst.Cols {
		panic("cv: Accumulate size mismatch")
	}
	for i := range dst.Data {
		dst.Data[i] += float64(src.Data[i])
	}
}

// AccumulateSquare adds the squared samples of a single-channel image to an
// accumulator FloatMat in place (dst += src²). It panics on a size mismatch or a
// multi-channel source. This is OpenCV's cv::accumulateSquare.
func AccumulateSquare(src *Mat, dst *FloatMat) {
	requireChannels(src, 1, "AccumulateSquare")
	if src.Rows != dst.Rows || src.Cols != dst.Cols {
		panic("cv: AccumulateSquare size mismatch")
	}
	for i := range dst.Data {
		v := float64(src.Data[i])
		dst.Data[i] += v * v
	}
}

// AccumulateWeighted updates a running average accumulator in place:
// dst = (1-alpha)*dst + alpha*src. Larger alpha weights recent frames more,
// making this a simple exponential background model. It panics on a size
// mismatch or a multi-channel source. This is OpenCV's cv::accumulateWeighted.
func AccumulateWeighted(src *Mat, dst *FloatMat, alpha float64) {
	requireChannels(src, 1, "AccumulateWeighted")
	if src.Rows != dst.Rows || src.Cols != dst.Cols {
		panic("cv: AccumulateWeighted size mismatch")
	}
	inv := 1 - alpha
	for i := range dst.Data {
		dst.Data[i] = inv*dst.Data[i] + alpha*float64(src.Data[i])
	}
}

// GetGaborKernel builds a Gabor filter kernel of size ksize×ksize. sigma is the
// standard deviation of the Gaussian envelope, theta the orientation of the
// normal to the parallel stripes in radians, lambda the wavelength of the
// sinusoidal factor, gamma the spatial aspect ratio and psi the phase offset.
// It mirrors OpenCV's cv::getGaborKernel. It panics unless ksize is positive.
func GetGaborKernel(ksize int, sigma, theta, lambda, gamma, psi float64) *FloatMat {
	if ksize <= 0 {
		panic(fmt.Sprintf("cv: GetGaborKernel requires positive ksize, got %d", ksize))
	}
	half := ksize / 2
	sigmaX := sigma
	sigmaY := sigma / gamma
	c, s := math.Cos(theta), math.Sin(theta)
	out := NewFloatMat(ksize, ksize)
	for y := -half; y <= half && y-(-half) < ksize; y++ {
		for x := -half; x <= half && x-(-half) < ksize; x++ {
			xr := float64(x)*c + float64(y)*s
			yr := -float64(x)*s + float64(y)*c
			env := math.Exp(-0.5 * (xr*xr/(sigmaX*sigmaX) + yr*yr/(sigmaY*sigmaY)))
			carrier := math.Cos(2*math.Pi*xr/lambda + psi)
			out.Data[(y+half)*ksize+(x+half)] = env * carrier
		}
	}
	return out
}

// SpatialGradient computes the first-order x and y derivatives of a
// single-channel image with the 3×3 Sobel operator, returning them as FloatMats
// (unclamped, so they carry sign). It mirrors OpenCV's cv::spatialGradient. It
// panics if src is not single-channel.
func SpatialGradient(src *Mat) (dx, dy *FloatMat) {
	requireChannels(src, 1, "SpatialGradient")
	dx = sobelChannel(src, 1, 0)
	dy = sobelChannel(src, 0, 1)
	return dx, dy
}

// sobelChannel runs SobelFloat on a single-channel image and reshapes its first
// channel plane into a FloatMat.
func sobelChannel(src *Mat, dx, dy int) *FloatMat {
	g := SobelFloat(src, dx, dy, 3)
	out := NewFloatMat(src.Rows, src.Cols)
	copy(out.Data, g[0])
	return out
}

// PreCornerDetect computes the corner-detection function
// Dx²·Dyy + Dy²·Dxx − 2·Dx·Dy·Dxy from the first and second derivatives of a
// single-channel image, matching OpenCV's cv::preCornerDetect. Peaks of the
// result indicate corners. It panics if src is not single-channel.
func PreCornerDetect(src *Mat) *FloatMat {
	requireChannels(src, 1, "PreCornerDetect")
	dx := sobelChannel(src, 1, 0)
	dy := sobelChannel(src, 0, 1)
	dxx := sobelChannel(src, 2, 0)
	dyy := sobelChannel(src, 0, 2)
	dxy := sobelChannel(src, 1, 1)
	out := NewFloatMat(src.Rows, src.Cols)
	for i := range out.Data {
		ax, ay := dx.Data[i], dy.Data[i]
		out.Data[i] = ax*ax*dyy.Data[i] + ay*ay*dxx.Data[i] - 2*ax*ay*dxy.Data[i]
	}
	return out
}

// ColormapType selects a false-colour palette for [ApplyColorMap].
type ColormapType int

const (
	// ColormapGray maps intensity to a neutral grayscale ramp.
	ColormapGray ColormapType = iota
	// ColormapJet maps intensity through the classic blue-cyan-yellow-red ramp.
	ColormapJet
	// ColormapHot maps intensity through a black-red-yellow-white ramp.
	ColormapHot
	// ColormapBone maps intensity through a blue-tinted grayscale ramp.
	ColormapBone
)

// ApplyColorMap converts a single-channel intensity image to a three-channel
// RGB image using the given colormap, mirroring OpenCV's cv::applyColorMap
// (though colours are emitted in RGB order to match this package's convention).
// It panics if src is not single-channel.
func ApplyColorMap(src *Mat, colormap ColormapType) *Mat {
	requireChannels(src, 1, "ApplyColorMap")
	out := NewMat(src.Rows, src.Cols, 3)
	for p := 0; p < src.Total(); p++ {
		t := float64(src.Data[p]) / 255
		r, g, b := colormapSample(colormap, t)
		out.Data[p*3+0] = clampToUint8(r*255 + 0.5)
		out.Data[p*3+1] = clampToUint8(g*255 + 0.5)
		out.Data[p*3+2] = clampToUint8(b*255 + 0.5)
	}
	return out
}

// colormapSample returns the RGB triple (each in [0,1]) for normalised
// intensity t in [0,1] under the given colormap.
func colormapSample(cm ColormapType, t float64) (r, g, b float64) {
	if t < 0 {
		t = 0
	} else if t > 1 {
		t = 1
	}
	switch cm {
	case ColormapJet:
		return clamp01(1.5 - math.Abs(4*t-3)), clamp01(1.5 - math.Abs(4*t-2)), clamp01(1.5 - math.Abs(4*t-1))
	case ColormapHot:
		return clamp01(3 * t), clamp01(3*t - 1), clamp01(3*t - 2)
	case ColormapBone:
		// Blue-tinted grayscale: a 7:1 blend of a gray ramp with a
		// reversed "hot"-style ramp, which lifts the blue channel in the
		// shadows and the red channel in the highlights.
		r = (7*t + clamp01(3*t-2)) / 8
		g = (7*t + clamp01(3*t-1)) / 8
		b = (7*t + clamp01(3*t)) / 8
		return r, g, b
	default:
		return t, t, t
	}
}

// clamp01 clamps v to the closed interval [0,1].
func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
