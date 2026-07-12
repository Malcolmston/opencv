package intensity

import (
	"fmt"

	cv "github.com/malcolmston/opencv"
)

// stretchLUT builds the linear-stretch table that maps [lo,hi] onto [0,255],
// clamping outside that band. lo must be < hi.
func stretchLUT(lo, hi int) []uint8 {
	span := float64(hi - lo)
	lut := make([]uint8, 256)
	for i := 0; i < 256; i++ {
		lut[i] = clampToUint8((float64(i)-float64(lo))/span*255 + 0.5)
	}
	return lut
}

// AutoContrast performs a per-channel percentile contrast stretch, the operation
// PIL calls autocontrast. For each channel independently it discards the darkest
// and brightest clipPercent percent of samples and linearly maps the surviving
// range onto the full [0,255] span, so faint images gain contrast and a slight
// clip immunises the stretch against outliers. clipPercent is given in percent
// (e.g. 1 clips 1% at each end) and must lie in [0,49); it panics otherwise.
// Because the channels are stretched independently this also neutralises colour
// casts. A channel with no spread is left unchanged.
func AutoContrast(img *cv.Mat, clipPercent float64) *cv.Mat {
	requireImage(img, "AutoContrast")
	if !(clipPercent >= 0 && clipPercent < 49) {
		panic(fmt.Sprintf("intensity: AutoContrast requires clipPercent in [0,49), got %v", clipPercent))
	}
	frac := clipPercent / 100
	ch := img.Channels
	dst := cv.NewMat(img.Rows, img.Cols, ch)
	for c := 0; c < ch; c++ {
		hist := histFloat(channelFloat(img, c))
		lo, hi := clipBounds(hist, img.Total(), frac, frac)
		if hi <= lo {
			for p := 0; p < img.Total(); p++ {
				dst.Data[p*ch+c] = img.Data[p*ch+c]
			}
			continue
		}
		lut := stretchLUT(lo, hi)
		for p := 0; p < img.Total(); p++ {
			dst.Data[p*ch+c] = lut[img.Data[p*ch+c]]
		}
	}
	return dst
}

// AutoLevels performs a luminance-driven levels adjustment. Unlike
// [AutoContrast], it derives a single black point and white point from the image
// luminance — discarding the lowest lowPercent and highest highPercent percent
// of luminance — and applies that one mapping to every channel, which stretches
// contrast while preserving the colour balance and relative channel ratios.
// lowPercent and highPercent are in percent and must each lie in [0,49); it
// panics otherwise. A luminance range with no spread returns a copy.
func AutoLevels(img *cv.Mat, lowPercent, highPercent float64) *cv.Mat {
	requireImage(img, "AutoLevels")
	if !(lowPercent >= 0 && lowPercent < 49) || !(highPercent >= 0 && highPercent < 49) {
		panic(fmt.Sprintf("intensity: AutoLevels requires percentages in [0,49), got low=%v high=%v",
			lowPercent, highPercent))
	}
	hist := histFloat(lumaFloat(img))
	lo, hi := clipBounds(hist, img.Total(), lowPercent/100, highPercent/100)
	if hi <= lo {
		return img.Clone()
	}
	return applyLUT(img, stretchLUT(lo, hi))
}

// ContrastLimitedStretch is a percentile stretch whose amplification is capped.
// Like [AutoLevels] it finds a luminance black/white point after clipping
// clipPercent at each end, but it limits the stretch slope (gain) to maxGain, so
// a genuinely low-contrast or nearly flat image is not blown up by an enormous
// factor into amplified noise. The gain-limited mapping is centred on the
// midpoint of the surviving range and applied to every channel. clipPercent must
// lie in [0,49) and maxGain must be ≥ 1; it panics otherwise.
func ContrastLimitedStretch(img *cv.Mat, clipPercent, maxGain float64) *cv.Mat {
	requireImage(img, "ContrastLimitedStretch")
	if !(clipPercent >= 0 && clipPercent < 49) {
		panic(fmt.Sprintf("intensity: ContrastLimitedStretch requires clipPercent in [0,49), got %v", clipPercent))
	}
	if !(maxGain >= 1) {
		panic(fmt.Sprintf("intensity: ContrastLimitedStretch requires maxGain >= 1, got %v", maxGain))
	}
	frac := clipPercent / 100
	hist := histFloat(lumaFloat(img))
	lo, hi := clipBounds(hist, img.Total(), frac, frac)
	if hi <= lo {
		return img.Clone()
	}
	// Unclamped gain that would map [lo,hi] onto the full range, then limit it.
	gain := 255 / float64(hi-lo)
	if gain > maxGain {
		gain = maxGain
	}
	center := float64(lo+hi) / 2
	lut := make([]uint8, 256)
	for i := 0; i < 256; i++ {
		v := (float64(i)-center)*gain + 127.5
		lut[i] = clampToUint8(v + 0.5)
	}
	return applyLUT(img, lut)
}
