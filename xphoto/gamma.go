package xphoto

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// GammaCorrection applies a power-law (gamma) transfer function to every sample
// of src, porting cv::xphoto::gammaCorrection. Each 8-bit sample s in [0,255] is
// mapped to 255*(s/255)**(1/gamma) via a precomputed 256-entry lookup table, so
// the operation is exact and O(pixels) regardless of channel count.
//
// gamma follows the usual photographic convention: gamma > 1 brightens the
// mid-tones (a decoding / display gamma), gamma < 1 darkens them, and gamma == 1
// is the identity. A non-positive gamma is treated as 1. src may have any number
// of channels; the same curve is applied to every channel independently. The
// input is never modified.
func GammaCorrection(src *cv.Mat, gamma float64) *cv.Mat {
	requireNonEmpty(src, "GammaCorrection")
	if gamma <= 0 {
		gamma = 1
	}
	lut := GammaLUT(gamma)
	dst := cv.NewMat(src.Rows, src.Cols, src.Channels)
	for i, v := range src.Data {
		dst.Data[i] = lut[v]
	}
	return dst
}

// GammaLUT returns the 256-entry lookup table used by [GammaCorrection] for the
// given gamma. Entry i equals round(255*(i/255)**(1/gamma)) clamped to [0,255].
// It is exported so callers can reuse a single table across many images or
// compose it with other per-sample maps. A non-positive gamma yields the
// identity table.
func GammaLUT(gamma float64) [256]uint8 {
	var lut [256]uint8
	if gamma <= 0 {
		for i := range lut {
			lut[i] = uint8(i)
		}
		return lut
	}
	inv := 1.0 / gamma
	for i := 0; i < 256; i++ {
		lut[i] = clampU8(255.0 * math.Pow(float64(i)/255.0, inv))
	}
	return lut
}
