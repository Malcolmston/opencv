package colorspaces2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// KelvinToRGB converts a correlated colour temperature in kelvin to an
// approximate display [RGB] using the widely used Tanner Helland approximation.
// It is accurate for roughly 1000K–40000K and the result is normalised to
// [0,1]. The brightest channel is not necessarily 1.0.
func KelvinToRGB(kelvin float64) RGB {
	t := kelvin / 100
	var r, g, b float64
	// Red.
	if t <= 66 {
		r = 255
	} else {
		r = 329.698727446 * math.Pow(t-60, -0.1332047592)
	}
	// Green.
	if t <= 66 {
		g = 99.4708025861*math.Log(t) - 161.1195681661
	} else {
		g = 288.1221695283 * math.Pow(t-60, -0.0755148492)
	}
	// Blue.
	switch {
	case t >= 66:
		b = 255
	case t <= 19:
		b = 0
	default:
		b = 138.5177312231*math.Log(t-10) - 305.0447927307
	}
	return RGB{
		R: clamp01(r / 255),
		G: clamp01(g / 255),
		B: clamp01(b / 255),
	}
}

// PlanckianXYZ returns the CIE [XYZ] chromaticity (with Y normalised to 1.0) of
// a black-body radiator at the given temperature in kelvin, using Kim et al.'s
// cubic-spline approximation of the Planckian locus. It is valid for roughly
// 1667K–25000K.
func PlanckianXYZ(kelvin float64) XYZ {
	t := kelvin
	var x float64
	if t <= 4000 {
		x = -0.2661239e9/(t*t*t) - 0.2343589e6/(t*t) + 0.8776956e3/t + 0.179910
	} else {
		x = -3.0258469e9/(t*t*t) + 2.1070379e6/(t*t) + 0.2226347e3/t + 0.240390
	}
	var y float64
	switch {
	case t <= 2222:
		y = -1.1063814*x*x*x - 1.34811020*x*x + 2.18555832*x - 0.20219683
	case t <= 4000:
		y = -0.9549476*x*x*x - 1.37418593*x*x + 2.09137015*x - 0.16748867
	default:
		y = 3.0817580*x*x*x - 5.87338670*x*x + 3.75112997*x - 0.37001483
	}
	if y == 0 {
		return XYZ{X: 0, Y: 1, Z: 0}
	}
	return XYZ{X: x / y, Y: 1, Z: (1 - x - y) / y}
}

// CorrelatedColorTemperature estimates the correlated colour temperature (in
// kelvin) of an [XYZ] colour using McCamy's cubic approximation. It returns 0
// for the degenerate black colour.
func CorrelatedColorTemperature(c XYZ) float64 {
	sum := c.X + c.Y + c.Z
	if sum == 0 {
		return 0
	}
	x := c.X / sum
	y := c.Y / sum
	den := 0.1858 - y
	if den == 0 {
		return 0
	}
	n := (x - 0.3320) / den
	return 449*n*n*n + 3525*n*n + 6823.3*n + 5520.33
}

// AdjustColorTemperatureMat returns a new Mat that re-balances src as though the
// scene white were shifted from a neutral reference to the target colour
// temperature in kelvin. Temperatures below the 6500K neutral warm the image;
// higher temperatures cool it. The per-channel gains are normalised so the
// green channel is unchanged.
func AdjustColorTemperatureMat(src *cv.Mat, kelvin float64) *cv.Mat {
	colorspaces2RequireRGB(src, "AdjustColorTemperatureMat")
	target := KelvinToRGB(kelvin)
	neutral := KelvinToRGB(6500)
	gains := [3]float64{
		safeRatio(target.R, neutral.R),
		safeRatio(target.G, neutral.G),
		safeRatio(target.B, neutral.B),
	}
	if gains[1] != 0 {
		gains[0] /= gains[1]
		gains[2] /= gains[1]
		gains[1] = 1
	}
	return ApplyChannelGains(src, gains)
}
