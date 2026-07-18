package colorspaces2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// SRGBToLinear converts a single gamma-encoded sRGB component in [0,1] to its
// linear-light value using the piecewise sRGB electro-optical transfer
// function.
func SRGBToLinear(c float64) float64 {
	if c <= 0.04045 {
		return c / 12.92
	}
	return math.Pow((c+0.055)/1.055, 2.4)
}

// LinearToSRGB converts a single linear-light component in [0,1] to its
// gamma-encoded sRGB value, inverting [SRGBToLinear].
func LinearToSRGB(c float64) float64 {
	if c <= 0.0031308 {
		return c * 12.92
	}
	return 1.055*math.Pow(c, 1/2.4) - 0.055
}

// ApplyGamma applies a simple power-law gamma curve to each channel of c,
// returning out = in^(1/gamma). A gamma greater than 1 brightens mid-tones; a
// gamma of 1 is the identity. It panics if gamma is not positive.
func ApplyGamma(c RGB, gamma float64) RGB {
	if gamma <= 0 {
		panic("colorspaces2: ApplyGamma requires gamma > 0")
	}
	inv := 1 / gamma
	return RGB{
		R: math.Pow(clamp01(c.R), inv),
		G: math.Pow(clamp01(c.G), inv),
		B: math.Pow(clamp01(c.B), inv),
	}
}

// BuildGammaLUT returns a 256-entry lookup table mapping an 8-bit input to the
// power-law gamma-corrected 8-bit output out = round(255*(in/255)^(1/gamma)).
// It panics if gamma is not positive.
func BuildGammaLUT(gamma float64) [256]uint8 {
	if gamma <= 0 {
		panic("colorspaces2: BuildGammaLUT requires gamma > 0")
	}
	inv := 1 / gamma
	var lut [256]uint8
	for i := 0; i < 256; i++ {
		lut[i] = uint8(math.Round(255 * math.Pow(float64(i)/255, inv)))
	}
	return lut
}

// ApplyLUTMat returns a copy of src with the per-channel lookup table lut
// applied to every sample. It works for any channel count.
func ApplyLUTMat(src *cv.Mat, lut [256]uint8) *cv.Mat {
	if src == nil || src.Empty() {
		panic("colorspaces2: ApplyLUTMat: empty Mat")
	}
	dst := cv.NewMat(src.Rows, src.Cols, src.Channels)
	for i, v := range src.Data {
		dst.Data[i] = lut[v]
	}
	return dst
}

// GammaCorrectMat returns a new Mat with power-law gamma correction applied to
// every sample of src (out = 255*(in/255)^(1/gamma)). It panics if gamma is not
// positive.
func GammaCorrectMat(src *cv.Mat, gamma float64) *cv.Mat {
	return ApplyLUTMat(src, BuildGammaLUT(gamma))
}

// LinearizeMat returns a new Mat in which each sample of the sRGB-encoded src
// has been converted to linear light and rescaled to [0,255]. It is the Mat
// analogue of [SRGBToLinear].
func LinearizeMat(src *cv.Mat) *cv.Mat {
	var lut [256]uint8
	for i := 0; i < 256; i++ {
		lut[i] = uint8(math.Round(255 * SRGBToLinear(float64(i)/255)))
	}
	return ApplyLUTMat(src, lut)
}

// DelinearizeMat returns a new Mat in which each linear-light sample of src has
// been converted back to sRGB encoding, inverting [LinearizeMat].
func DelinearizeMat(src *cv.Mat) *cv.Mat {
	var lut [256]uint8
	for i := 0; i < 256; i++ {
		lut[i] = uint8(math.Round(255 * LinearToSRGB(float64(i)/255)))
	}
	return ApplyLUTMat(src, lut)
}
