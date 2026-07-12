package mcc

import "math"

// D65 tristimulus white point (2° observer), used by every XYZ<->Lab
// conversion in this package.
const (
	whiteX = 0.95047
	whiteY = 1.00000
	whiteZ = 1.08883
)

// SRGBToLinear expands a single sRGB component in the range [0,1] to
// linear-light, inverting the sRGB transfer function (a near-2.2 gamma with a
// small linear toe). Values outside [0,1] are handled by the same formula.
func SRGBToLinear(c float64) float64 {
	if c <= 0.04045 {
		return c / 12.92
	}
	return math.Pow((c+0.055)/1.055, 2.4)
}

// LinearToSRGB compresses a single linear-light component in [0,1] back to sRGB,
// the inverse of [SRGBToLinear].
func LinearToSRGB(c float64) float64 {
	if c <= 0.0031308 {
		return c * 12.92
	}
	return 1.055*math.Pow(c, 1.0/2.4) - 0.055
}

// linearRGBToXYZ converts a linear-light RGB triple (each in [0,1], sRGB
// primaries) to CIE XYZ under the D65 white point.
func linearRGBToXYZ(r, g, b float64) (x, y, z float64) {
	x = 0.4124564*r + 0.3575761*g + 0.1804375*b
	y = 0.2126729*r + 0.7151522*g + 0.0721750*b
	z = 0.0193339*r + 0.1191920*g + 0.9503041*b
	return
}

// labF is the CIE nonlinearity used by the XYZ->Lab transform.
func labF(t float64) float64 {
	const eps = 216.0 / 24389.0 // 0.008856
	if t > eps {
		return math.Cbrt(t)
	}
	const kappa = 24389.0 / 27.0 // 903.3
	return (kappa*t + 16) / 116
}

// xyzToLab converts CIE XYZ (D65) to CIE L*a*b*.
func xyzToLab(x, y, z float64) (l, a, bb float64) {
	fx := labF(x / whiteX)
	fy := labF(y / whiteY)
	fz := labF(z / whiteZ)
	l = 116*fy - 16
	a = 500 * (fx - fy)
	bb = 200 * (fy - fz)
	return
}

// rgbToLabF converts an sRGB triple given as float64 components in the 0..255
// range to CIE L*a*b* (D65). It is the float entry point used internally where
// measurements are kept as floats.
func rgbToLabF(r, g, b float64) [3]float64 {
	lr := SRGBToLinear(clamp01(r / 255))
	lg := SRGBToLinear(clamp01(g / 255))
	lb := SRGBToLinear(clamp01(b / 255))
	x, y, z := linearRGBToXYZ(lr, lg, lb)
	l, a, bb := xyzToLab(x, y, z)
	return [3]float64{l, a, bb}
}

// RGBToLab converts an 8-bit sRGB color (D65) to CIE L*a*b*. L is in [0,100];
// a and b are unbounded but typically within roughly [-128,127].
func RGBToLab(r, g, b uint8) [3]float64 {
	return rgbToLabF(float64(r), float64(g), float64(b))
}

// RGBToXYZ converts an 8-bit sRGB color to CIE XYZ (D65), with Y normalised so
// that a diffuse white maps to Y=1.
func RGBToXYZ(r, g, b uint8) [3]float64 {
	lr := SRGBToLinear(float64(r) / 255)
	lg := SRGBToLinear(float64(g) / 255)
	lb := SRGBToLinear(float64(b) / 255)
	x, y, z := linearRGBToXYZ(lr, lg, lb)
	return [3]float64{x, y, z}
}

// DeltaE76 returns the CIE76 color difference between two CIE L*a*b* colors: the
// plain Euclidean distance sqrt(dL^2 + da^2 + db^2). Values under about 1 are
// imperceptible; under about 2.3 are barely perceptible.
func DeltaE76(lab1, lab2 [3]float64) float64 {
	dl := lab1[0] - lab2[0]
	da := lab1[1] - lab2[1]
	db := lab1[2] - lab2[2]
	return math.Sqrt(dl*dl + da*da + db*db)
}

// DeltaERGB returns the CIE76 Delta E between two 8-bit sRGB colors by
// converting both to Lab first. It is a convenience wrapper over [RGBToLab] and
// [DeltaE76].
func DeltaERGB(a, b [3]uint8) float64 {
	return DeltaE76(RGBToLab(a[0], a[1], a[2]), RGBToLab(b[0], b[1], b[2]))
}

// clamp01 clamps a value to the closed interval [0,1].
func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// clampToUint8 rounds and saturates a float64 to the 0..255 uint8 range.
func clampToUint8(v float64) uint8 {
	v = math.Round(v)
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}
