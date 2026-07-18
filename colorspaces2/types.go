package colorspaces2

import (
	"image/color"
	"math"

	cv "github.com/malcolmston/opencv"
)

// RGB is an sRGB colour with gamma-encoded components normalised to [0,1].
// Values outside that range are permitted intermediate results; use
// [RGB.Clamp] to bring them back into gamut.
type RGB struct {
	// R is the red component in [0,1].
	R float64
	// G is the green component in [0,1].
	G float64
	// B is the blue component in [0,1].
	B float64
}

// HSV is a colour in the hue/saturation/value cylinder. H is in degrees [0,360),
// while S and V are in [0,1].
type HSV struct {
	// H is the hue angle in degrees, [0,360).
	H float64
	// S is the saturation in [0,1].
	S float64
	// V is the value (brightness) in [0,1].
	V float64
}

// HSL is a colour in the hue/saturation/lightness cylinder. H is in degrees
// [0,360), while S and L are in [0,1].
type HSL struct {
	// H is the hue angle in degrees, [0,360).
	H float64
	// S is the saturation in [0,1].
	S float64
	// L is the lightness in [0,1].
	L float64
}

// XYZ is a colour in the CIE 1931 XYZ space using the sRGB primaries. For the
// sRGB white point the reference white is (0.95047, 1.0, 1.08883).
type XYZ struct {
	// X is the X tristimulus value.
	X float64
	// Y is the Y tristimulus value (luminance).
	Y float64
	// Z is the Z tristimulus value.
	Z float64
}

// Lab is a colour in CIE L*a*b* space. L is in [0,100]; a and b are unbounded
// but typically lie in roughly [-128,127].
type Lab struct {
	// L is the lightness in [0,100].
	L float64
	// A is the green–red opponent axis.
	A float64
	// B is the blue–yellow opponent axis.
	B float64
}

// Luv is a colour in CIE L*u*v* space. L is in [0,100]; u and v are unbounded.
type Luv struct {
	// L is the lightness in [0,100].
	L float64
	// U is the u* chromaticity coordinate.
	U float64
	// V is the v* chromaticity coordinate.
	V float64
}

// YCbCr is a luma/chroma colour using the ITU-R BT.601 full-range (JPEG)
// coefficients. Y is in [0,1]; Cb and Cr are in [-0.5,0.5].
type YCbCr struct {
	// Y is the luma in [0,1].
	Y float64
	// Cb is the blue-difference chroma in [-0.5,0.5].
	Cb float64
	// Cr is the red-difference chroma in [-0.5,0.5].
	Cr float64
}

// YUV is an analog luma/chroma colour using the ITU-R BT.601 weights. Y is in
// [0,1]; U is in [-0.436,0.436] and V is in [-0.615,0.615].
type YUV struct {
	// Y is the luma in [0,1].
	Y float64
	// U is the blue-difference chroma in [-0.436,0.436].
	U float64
	// V is the red-difference chroma in [-0.615,0.615].
	V float64
}

// CMYK is a subtractive cyan/magenta/yellow/black colour with all four
// components in [0,1].
type CMYK struct {
	// C is the cyan component in [0,1].
	C float64
	// M is the magenta component in [0,1].
	M float64
	// Y is the yellow component in [0,1].
	Y float64
	// K is the key (black) component in [0,1].
	K float64
}

// NewRGBFromUint8 builds an [RGB] from three 8-bit samples, dividing each by
// 255 so that 0 maps to 0.0 and 255 maps to 1.0.
func NewRGBFromUint8(r, g, b uint8) RGB {
	return RGB{R: float64(r) / 255, G: float64(g) / 255, B: float64(b) / 255}
}

// Clamp returns a copy of c with every component limited to the [0,1] range.
func (c RGB) Clamp() RGB {
	return RGB{R: clamp01(c.R), G: clamp01(c.G), B: clamp01(c.B)}
}

// ToUint8 clamps c to [0,1] and returns the three components scaled to 8-bit
// integers with round-half-away-from-zero rounding.
func (c RGB) ToUint8() (r, g, b uint8) {
	return to8(c.R), to8(c.G), to8(c.B)
}

// ToRGBA converts c to a fully opaque [color.RGBA] from the standard library.
func (c RGB) ToRGBA() color.RGBA {
	r, g, b := c.ToUint8()
	return color.RGBA{R: r, G: g, B: b, A: 255}
}

// Luma returns the ITU-R BT.601 luma of c (0.299R + 0.587G + 0.114B), computed
// on the gamma-encoded components.
func (c RGB) Luma() float64 {
	return 0.299*c.R + 0.587*c.G + 0.114*c.B
}

// clamp01 limits v to the closed interval [0,1].
func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// to8 clamps v to [0,1] and scales it to an 8-bit value with rounding.
func to8(v float64) uint8 {
	return uint8(math.Round(clamp01(v) * 255))
}

// colorspaces2ReadRGB reads pixel (y,x) of a three-channel Mat as a normalised
// [RGB] value.
func colorspaces2ReadRGB(m *cv.Mat, y, x int) RGB {
	i := (y*m.Cols + x) * m.Channels
	return RGB{
		R: float64(m.Data[i]) / 255,
		G: float64(m.Data[i+1]) / 255,
		B: float64(m.Data[i+2]) / 255,
	}
}

// colorspaces2WriteRGB writes c into pixel (y,x) of a three-channel Mat,
// clamping and rounding to 8-bit.
func colorspaces2WriteRGB(m *cv.Mat, y, x int, c RGB) {
	i := (y*m.Cols + x) * m.Channels
	m.Data[i] = to8(c.R)
	m.Data[i+1] = to8(c.G)
	m.Data[i+2] = to8(c.B)
}

// colorspaces2RequireRGB panics if m is not a usable three-channel Mat.
func colorspaces2RequireRGB(m *cv.Mat, fn string) {
	if m == nil || m.Empty() {
		panic("colorspaces2: " + fn + ": empty Mat")
	}
	if m.Channels != 3 {
		panic("colorspaces2: " + fn + ": requires a 3-channel RGB Mat")
	}
}
