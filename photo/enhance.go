package photo

import (
	"math"
	"sort"

	cv "github.com/malcolmston/opencv"
)

// GammaCorrection applies a power-law (gamma) tone curve to img: each sample s
// is remapped to 255*(s/255)^gamma. gamma below one brightens mid-tones and
// lifts shadows (useful for dark images); gamma above one darkens them. The
// mapping is built once into a 256-entry lookup table, so the cost is
// independent of gamma. img may have any channel count; the output has the same
// shape and the original is not modified. gamma must be positive.
func GammaCorrection(img *cv.Mat, gamma float64) *cv.Mat {
	if img == nil || img.Empty() {
		panic("photo: GammaCorrection given an empty image")
	}
	if gamma <= 0 {
		panic("photo: GammaCorrection requires positive gamma")
	}
	var lut [256]uint8
	for i := 0; i < 256; i++ {
		lut[i] = clampU8(255 * math.Pow(float64(i)/255, gamma))
	}
	out := img.Clone()
	for i, v := range out.Data {
		out.Data[i] = lut[v]
	}
	return out
}

// UnsharpMask sharpens img with the classic unsharp-masking technique: it forms
// a Gaussian-blurred copy (the "unsharp" mask), takes the high-frequency detail
// as img minus that blur, and adds an amount-scaled copy of the detail back to
// img. Edges and texture gain local contrast while flat regions are untouched.
//
// ksize is the Gaussian kernel size (forced odd, default 5) and sigma its
// standard deviation (0 lets [cv.GaussianBlur] derive it from ksize). amount is
// the sharpening strength (e.g. 1.0 for a moderate effect); larger amounts
// sharpen more and can overshoot. img may have any channel count; the output has
// the same shape and the original is not modified.
func UnsharpMask(img *cv.Mat, ksize int, sigma, amount float64) *cv.Mat {
	if img == nil || img.Empty() {
		panic("photo: UnsharpMask given an empty image")
	}
	ksize = oddAtLeast(ksize, 5)
	if amount <= 0 {
		amount = 1.0
	}
	blur := cv.GaussianBlur(img, ksize, sigma)
	out := cv.NewMat(img.Rows, img.Cols, img.Channels)
	for i := range out.Data {
		o := float64(img.Data[i])
		b := float64(blur.Data[i])
		out.Data[i] = clampU8(o + amount*(o-b))
	}
	return out
}

// HistogramStretch performs per-channel linear contrast stretching
// (normalisation): for every channel the darkest sample is mapped to 0 and the
// brightest to 255, with a linear ramp in between, so the image uses the full
// tonal range. Channels are stretched independently. A channel whose samples are
// all equal is left unchanged. img may have any channel count; the output has
// the same shape and the original is not modified.
func HistogramStretch(img *cv.Mat) *cv.Mat {
	if img == nil || img.Empty() {
		panic("photo: HistogramStretch given an empty image")
	}
	ch := img.Channels
	lo := make([]int, ch)
	hi := make([]int, ch)
	for c := 0; c < ch; c++ {
		lo[c] = 255
		hi[c] = 0
	}
	for i := 0; i < img.Total(); i++ {
		for c := 0; c < ch; c++ {
			v := int(img.Data[i*ch+c])
			if v < lo[c] {
				lo[c] = v
			}
			if v > hi[c] {
				hi[c] = v
			}
		}
	}
	out := cv.NewMat(img.Rows, img.Cols, ch)
	for i := 0; i < img.Total(); i++ {
		for c := 0; c < ch; c++ {
			v := float64(img.Data[i*ch+c])
			if hi[c] > lo[c] {
				v = (v - float64(lo[c])) * 255 / float64(hi[c]-lo[c])
			}
			out.Data[i*ch+c] = clampU8(v)
		}
	}
	return out
}

// GrayWorldWhiteBalance corrects a colour cast under the gray-world assumption:
// the average colour of a scene should be neutral gray. It computes each
// channel's mean, then scales the channels so all three means equal their common
// average, neutralising a dominant tint. img must be three-channel; the output
// has the same shape and the original is not modified.
func GrayWorldWhiteBalance(img *cv.Mat) *cv.Mat {
	requireChannels(img, 3, "GrayWorldWhiteBalance")
	var mean [3]float64
	n := float64(img.Total())
	for i := 0; i < img.Total(); i++ {
		for c := 0; c < 3; c++ {
			mean[c] += float64(img.Data[i*3+c])
		}
	}
	for c := 0; c < 3; c++ {
		mean[c] /= n
	}
	gray := (mean[0] + mean[1] + mean[2]) / 3
	var scale [3]float64
	for c := 0; c < 3; c++ {
		if mean[c] > 0 {
			scale[c] = gray / mean[c]
		} else {
			scale[c] = 1
		}
	}
	out := cv.NewMat(img.Rows, img.Cols, 3)
	for i := 0; i < img.Total(); i++ {
		for c := 0; c < 3; c++ {
			out.Data[i*3+c] = clampU8(float64(img.Data[i*3+c]) * scale[c])
		}
	}
	return out
}

// SimpleWhiteBalance white-balances img by stretching each channel between
// robust percentiles, the method of OpenCV's SimpleWB. For every channel it
// discards the darkest pLow fraction and brightest pHigh fraction of samples as
// outliers, then linearly maps the remaining range to [0,255]. Clipping the
// extremes makes the balance robust to a few very dark or very bright pixels.
//
// pLow and pHigh are fractions in [0,0.5); each defaults to 0.02 when
// non-positive. img may have any channel count; the output has the same shape
// and the original is not modified.
func SimpleWhiteBalance(img *cv.Mat, pLow, pHigh float64) *cv.Mat {
	if img == nil || img.Empty() {
		panic("photo: SimpleWhiteBalance given an empty image")
	}
	if pLow <= 0 {
		pLow = 0.02
	}
	if pHigh <= 0 {
		pHigh = 0.02
	}
	ch := img.Channels
	n := img.Total()
	out := cv.NewMat(img.Rows, img.Cols, ch)
	buf := make([]int, n)
	for c := 0; c < ch; c++ {
		for i := 0; i < n; i++ {
			buf[i] = int(img.Data[i*ch+c])
		}
		sort.Ints(buf)
		loIdx := int(float64(n) * pLow)
		hiIdx := n - 1 - int(float64(n)*pHigh)
		if hiIdx <= loIdx {
			loIdx, hiIdx = 0, n-1
		}
		lo := float64(buf[loIdx])
		hi := float64(buf[hiIdx])
		span := hi - lo
		for i := 0; i < n; i++ {
			v := float64(img.Data[i*ch+c])
			if span > 0 {
				v = (v - lo) * 255 / span
			}
			out.Data[i*ch+c] = clampU8(v)
		}
	}
	return out
}
