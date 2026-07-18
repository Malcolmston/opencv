package colorspaces2

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// ChannelMeans returns the mean sample value of each of the first three
// channels of src, on the 8-bit [0,255] scale. It panics if src is not a
// three-channel RGB Mat.
func ChannelMeans(src *cv.Mat) [3]float64 {
	colorspaces2RequireRGB(src, "ChannelMeans")
	var sum [3]float64
	n := src.Rows * src.Cols
	for i := 0; i < n; i++ {
		base := i * 3
		sum[0] += float64(src.Data[base])
		sum[1] += float64(src.Data[base+1])
		sum[2] += float64(src.Data[base+2])
	}
	return [3]float64{sum[0] / float64(n), sum[1] / float64(n), sum[2] / float64(n)}
}

// ChannelMaxima returns the maximum sample value of each of the first three
// channels of src, on the 8-bit [0,255] scale.
func ChannelMaxima(src *cv.Mat) [3]uint8 {
	colorspaces2RequireRGB(src, "ChannelMaxima")
	var max [3]uint8
	n := src.Rows * src.Cols
	for i := 0; i < n; i++ {
		base := i * 3
		for c := 0; c < 3; c++ {
			if v := src.Data[base+c]; v > max[c] {
				max[c] = v
			}
		}
	}
	return max
}

// ApplyChannelGains returns a new Mat in which channel c of every pixel of src
// is multiplied by gains[c] and clamped to [0,255]. It panics if src is not a
// three-channel RGB Mat.
func ApplyChannelGains(src *cv.Mat, gains [3]float64) *cv.Mat {
	colorspaces2RequireRGB(src, "ApplyChannelGains")
	dst := cv.NewMat(src.Rows, src.Cols, 3)
	n := src.Rows * src.Cols
	for i := 0; i < n; i++ {
		base := i * 3
		for c := 0; c < 3; c++ {
			v := math.Round(float64(src.Data[base+c]) * gains[c])
			if v < 0 {
				v = 0
			} else if v > 255 {
				v = 255
			}
			dst.Data[base+c] = uint8(v)
		}
	}
	return dst
}

// GrayWorldGains returns the per-channel gains implied by the gray-world
// assumption: each channel is scaled so its mean equals the average of the
// three channel means. A channel whose mean is zero receives a gain of 1.
func GrayWorldGains(src *cv.Mat) [3]float64 {
	means := ChannelMeans(src)
	avg := (means[0] + means[1] + means[2]) / 3
	var gains [3]float64
	for c := 0; c < 3; c++ {
		gains[c] = safeRatio(avg, means[c])
	}
	return gains
}

// GrayWorldWhiteBalance returns a white-balanced copy of src under the
// gray-world assumption, scaling each channel so its mean matches the overall
// mean.
func GrayWorldWhiteBalance(src *cv.Mat) *cv.Mat {
	return ApplyChannelGains(src, GrayWorldGains(src))
}

// WhitePatchGains returns the per-channel gains implied by the white-patch
// (max-RGB) assumption at the given percentile in [0,1]: each channel is scaled
// so that its value at that percentile maps to 255. A percentile of 1 uses the
// per-channel maximum. It panics if percentile is outside [0,1].
func WhitePatchGains(src *cv.Mat, percentile float64) [3]float64 {
	colorspaces2RequireRGB(src, "WhitePatchGains")
	if percentile < 0 || percentile > 1 {
		panic("colorspaces2: WhitePatchGains: percentile must be in [0,1]")
	}
	n := src.Rows * src.Cols
	var gains [3]float64
	for c := 0; c < 3; c++ {
		vals := make([]uint8, n)
		for i := 0; i < n; i++ {
			vals[i] = src.Data[i*3+c]
		}
		sort.Slice(vals, func(a, b int) bool { return vals[a] < vals[b] })
		idx := int(math.Round(percentile * float64(n-1)))
		ref := float64(vals[idx])
		gains[c] = safeRatio(255, ref)
	}
	return gains
}

// WhitePatchWhiteBalance returns a white-balanced copy of src using the
// white-patch assumption, scaling each channel so its maximum maps to 255.
func WhitePatchWhiteBalance(src *cv.Mat) *cv.Mat {
	return WhitePatchWhiteBalancePercentile(src, 1.0)
}

// WhitePatchWhiteBalancePercentile returns a white-balanced copy of src using
// the white-patch assumption evaluated at the given percentile in [0,1]. Using
// a percentile slightly below 1 (for example 0.99) is more robust to specular
// highlights and hot pixels than the strict maximum.
func WhitePatchWhiteBalancePercentile(src *cv.Mat, percentile float64) *cv.Mat {
	return ApplyChannelGains(src, WhitePatchGains(src, percentile))
}
