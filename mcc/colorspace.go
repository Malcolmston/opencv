package mcc

import "math"

// labFInv inverts [labF], the CIE nonlinearity, mapping f(t) back to t.
func labFInv(ft float64) float64 {
	const eps = 216.0 / 24389.0
	const kappa = 24389.0 / 27.0
	t3 := ft * ft * ft
	if t3 > eps {
		return t3
	}
	return (116*ft - 16) / kappa
}

// XYZToLab converts a CIE XYZ color to CIE L*a*b* under an explicit reference
// white point (for example [WhiteD65], [WhiteD50] or [WhiteA]). L* is in
// [0,100]; a* and b* are unbounded but typically within roughly [-128,127].
// Unlike the package's internal D65-only helper, this accepts any white point,
// which is what non-D65 workflows (print/D50, tungsten/A) require.
func XYZToLab(xyz, white [3]float64) [3]float64 {
	fx := labF(xyz[0] / white[0])
	fy := labF(xyz[1] / white[1])
	fz := labF(xyz[2] / white[2])
	return [3]float64{116*fy - 16, 500 * (fx - fy), 200 * (fy - fz)}
}

// LabToXYZ converts a CIE L*a*b* color back to CIE XYZ under the given reference
// white point. It is the exact inverse of [XYZToLab].
func LabToXYZ(lab, white [3]float64) [3]float64 {
	fy := (lab[0] + 16) / 116
	fx := fy + lab[1]/500
	fz := fy - lab[2]/200
	return [3]float64{
		labFInv(fx) * white[0],
		labFInv(fy) * white[1],
		labFInv(fz) * white[2],
	}
}

// LabToLCh converts CIE L*a*b* to cylindrical CIE LCh(ab): lightness L*, chroma
// C* = sqrt(a*^2 + b*^2) and hue angle h in degrees within [0,360). Chroma and
// hue are often more intuitive than the Cartesian a*/b* axes.
func LabToLCh(lab [3]float64) [3]float64 {
	c := math.Hypot(lab[1], lab[2])
	h := math.Atan2(lab[2], lab[1]) * 180 / math.Pi
	if h < 0 {
		h += 360
	}
	return [3]float64{lab[0], c, h}
}

// LChToLab converts cylindrical CIE LCh(ab) (L*, C*, hue in degrees) back to
// Cartesian CIE L*a*b*. It is the exact inverse of [LabToLCh].
func LChToLab(lch [3]float64) [3]float64 {
	hr := lch[2] * math.Pi / 180
	return [3]float64{lch[0], lch[1] * math.Cos(hr), lch[1] * math.Sin(hr)}
}

// XYZToxyY converts CIE XYZ to CIE xyY: the (x,y) chromaticity coordinates plus
// the luminance Y. For a black input (X=Y=Z=0) the chromaticity is reported as
// (0,0).
func XYZToxyY(xyz [3]float64) [3]float64 {
	sum := xyz[0] + xyz[1] + xyz[2]
	if sum == 0 {
		return [3]float64{0, 0, 0}
	}
	return [3]float64{xyz[0] / sum, xyz[1] / sum, xyz[1]}
}

// XYYToXYZ converts CIE xyY (chromaticity x,y and luminance Y) back to CIE XYZ.
// A y of 0 is degenerate and yields a black (0,0,0) result.
func XYYToXYZ(xyY [3]float64) [3]float64 {
	x, y, yy := xyY[0], xyY[1], xyY[2]
	if y == 0 {
		return [3]float64{0, 0, 0}
	}
	return [3]float64{x * yy / y, yy, (1 - x - y) * yy / y}
}

// xyzToLinearRGB converts a CIE XYZ triple (D65) to linear-light sRGB primaries;
// it is the inverse of the internal linearRGBToXYZ.
func xyzToLinearRGB(x, y, z float64) (r, g, b float64) {
	r = 3.2404542*x - 1.5371385*y - 0.4985314*z
	g = -0.9692660*x + 1.8760108*y + 0.0415560*z
	b = 0.0556434*x - 0.2040259*y + 1.0572252*z
	return
}

// LinearRGBToXYZ converts a linear-light sRGB triple (each component in [0,1],
// sRGB primaries, D65) to CIE XYZ with Y normalised so a diffuse white maps to
// Y=1. This exposes the matrix that underlies [RGBToXYZ] for callers working in
// linear light.
func LinearRGBToXYZ(r, g, b float64) [3]float64 {
	x, y, z := linearRGBToXYZ(r, g, b)
	return [3]float64{x, y, z}
}

// XYZToRGB converts a CIE XYZ color (D65) to an 8-bit sRGB triple, applying the
// sRGB companding curve and saturating out-of-gamut results to [0,255].
func XYZToRGB(xyz [3]float64) [3]uint8 {
	lr, lg, lb := xyzToLinearRGB(xyz[0], xyz[1], xyz[2])
	return [3]uint8{
		clampToUint8(LinearToSRGB(clamp01(lr)) * 255),
		clampToUint8(LinearToSRGB(clamp01(lg)) * 255),
		clampToUint8(LinearToSRGB(clamp01(lb)) * 255),
	}
}

// LabToRGB converts a CIE L*a*b* color under the given white point to an 8-bit
// sRGB triple. When white is not D65 the color is chromatically adapted to D65
// (via [Bradford]) before encoding, so print/D50 or tungsten/A Lab values render
// correctly on an sRGB display. Out-of-gamut colors are saturated to [0,255].
func LabToRGB(lab, white [3]float64) [3]uint8 {
	xyz := LabToXYZ(lab, white)
	if white != WhiteD65 {
		xyz = ChromaticAdaptation(xyz, white, WhiteD65, Bradford)
	}
	return XYZToRGB(xyz)
}

// GammaExpand applies a pure power-law decoding: it raises a nonnegative
// component to the given gamma (mapping a gamma-encoded value to linear light),
// preserving sign for negatives. Use it as a simpler alternative to the sRGB
// curve when a display is characterised by a single exponent such as 2.2.
func GammaExpand(c, gamma float64) float64 {
	if c < 0 {
		return -math.Pow(-c, gamma)
	}
	return math.Pow(c, gamma)
}

// GammaCompress applies a pure power-law encoding (component^(1/gamma)),
// the inverse of [GammaExpand].
func GammaCompress(c, gamma float64) float64 {
	if gamma == 0 {
		return c
	}
	if c < 0 {
		return -math.Pow(-c, 1/gamma)
	}
	return math.Pow(c, 1/gamma)
}
