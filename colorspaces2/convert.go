package colorspaces2

import "math"

// RGBToHSV converts a gamma-encoded [RGB] colour to [HSV]. The hue is undefined
// for achromatic colours and is reported as 0 in that case.
func RGBToHSV(c RGB) HSV {
	max := math.Max(c.R, math.Max(c.G, c.B))
	min := math.Min(c.R, math.Min(c.G, c.B))
	delta := max - min
	h := hueOf(c, max, delta)
	var s float64
	if max > 0 {
		s = delta / max
	}
	return HSV{H: h, S: s, V: max}
}

// HSVToRGB converts an [HSV] colour back to gamma-encoded [RGB].
func HSVToRGB(c HSV) RGB {
	h := math.Mod(math.Mod(c.H, 360)+360, 360)
	cc := c.V * c.S
	x := cc * (1 - math.Abs(math.Mod(h/60, 2)-1))
	m := c.V - cc
	r, g, b := hueSector(h, cc, x)
	return RGB{R: r + m, G: g + m, B: b + m}
}

// RGBToHSL converts a gamma-encoded [RGB] colour to [HSL].
func RGBToHSL(c RGB) HSL {
	max := math.Max(c.R, math.Max(c.G, c.B))
	min := math.Min(c.R, math.Min(c.G, c.B))
	delta := max - min
	l := (max + min) / 2
	h := hueOf(c, max, delta)
	var s float64
	if delta != 0 {
		s = delta / (1 - math.Abs(2*l-1))
	}
	return HSL{H: h, S: s, L: l}
}

// HSLToRGB converts an [HSL] colour back to gamma-encoded [RGB].
func HSLToRGB(c HSL) RGB {
	h := math.Mod(math.Mod(c.H, 360)+360, 360)
	cc := (1 - math.Abs(2*c.L-1)) * c.S
	x := cc * (1 - math.Abs(math.Mod(h/60, 2)-1))
	m := c.L - cc/2
	r, g, b := hueSector(h, cc, x)
	return RGB{R: r + m, G: g + m, B: b + m}
}

// hueOf computes the hue angle in degrees given the channel maximum and the
// max-min spread of an RGB colour.
func hueOf(c RGB, max, delta float64) float64 {
	if delta == 0 {
		return 0
	}
	var h float64
	switch max {
	case c.R:
		h = math.Mod((c.G-c.B)/delta, 6)
	case c.G:
		h = (c.B-c.R)/delta + 2
	default:
		h = (c.R-c.G)/delta + 4
	}
	h *= 60
	if h < 0 {
		h += 360
	}
	return h
}

// hueSector maps a hue angle and the chroma/second-largest values to the raw
// RGB triple used by the HSV/HSL reconstruction formulae.
func hueSector(h, cc, x float64) (r, g, b float64) {
	switch {
	case h < 60:
		return cc, x, 0
	case h < 120:
		return x, cc, 0
	case h < 180:
		return 0, cc, x
	case h < 240:
		return 0, x, cc
	case h < 300:
		return x, 0, cc
	default:
		return cc, 0, x
	}
}

// sRGB <-> linear-RGB <-> XYZ matrices for the sRGB primaries and D65 white.
var (
	rgb2xyzMat = Matrix3{
		{0.4124564, 0.3575761, 0.1804375},
		{0.2126729, 0.7151522, 0.0721750},
		{0.0193339, 0.1191920, 0.9503041},
	}
	xyz2rgbMat = Matrix3{
		{3.2404542, -1.5371385, -0.4985314},
		{-0.9692660, 1.8760108, 0.0415560},
		{0.0556434, -0.2040259, 1.0572252},
	}
)

// RGBToXYZ converts a gamma-encoded sRGB colour to CIE [XYZ], linearising the
// sRGB transfer function first.
func RGBToXYZ(c RGB) XYZ {
	r := SRGBToLinear(c.R)
	g := SRGBToLinear(c.G)
	b := SRGBToLinear(c.B)
	v := rgb2xyzMat.MulVec(XYZ{r, g, b})
	return v
}

// XYZToRGB converts a CIE [XYZ] colour to gamma-encoded sRGB, applying the sRGB
// transfer function after the linear transform.
func XYZToRGB(c XYZ) RGB {
	v := xyz2rgbMat.MulVec(c)
	return RGB{
		R: LinearToSRGB(v.X),
		G: LinearToSRGB(v.Y),
		B: LinearToSRGB(v.Z),
	}
}

// labDelta is the CIE break-point 6/29 used by the L*a*b*/L*u*v* transfer
// functions.
const labDelta = 6.0 / 29.0

// labF is the forward CIELAB non-linearity.
func labF(t float64) float64 {
	if t > labDelta*labDelta*labDelta {
		return math.Cbrt(t)
	}
	return t/(3*labDelta*labDelta) + 4.0/29.0
}

// labFInv inverts [labF].
func labFInv(t float64) float64 {
	if t > labDelta {
		return t * t * t
	}
	return 3 * labDelta * labDelta * (t - 4.0/29.0)
}

// XYZToLab converts a CIE [XYZ] colour to [Lab] using the D65 reference white.
func XYZToLab(c XYZ) Lab {
	w := WhitePointD65
	fx := labF(c.X / w.X)
	fy := labF(c.Y / w.Y)
	fz := labF(c.Z / w.Z)
	return Lab{
		L: 116*fy - 16,
		A: 500 * (fx - fy),
		B: 200 * (fy - fz),
	}
}

// LabToXYZ converts a [Lab] colour to CIE [XYZ] using the D65 reference white.
func LabToXYZ(c Lab) XYZ {
	w := WhitePointD65
	fy := (c.L + 16) / 116
	fx := fy + c.A/500
	fz := fy - c.B/200
	return XYZ{
		X: w.X * labFInv(fx),
		Y: w.Y * labFInv(fy),
		Z: w.Z * labFInv(fz),
	}
}

// RGBToLab converts a gamma-encoded sRGB colour to [Lab] (via [XYZ]).
func RGBToLab(c RGB) Lab { return XYZToLab(RGBToXYZ(c)) }

// LabToRGB converts a [Lab] colour to gamma-encoded sRGB (via [XYZ]).
func LabToRGB(c Lab) RGB { return XYZToRGB(LabToXYZ(c)) }

// uvPrime returns the CIE 1976 u',v' chromaticity coordinates of an XYZ colour.
// For the black point (0,0,0) it returns (0,0).
func uvPrime(c XYZ) (up, vp float64) {
	d := c.X + 15*c.Y + 3*c.Z
	if d == 0 {
		return 0, 0
	}
	return 4 * c.X / d, 9 * c.Y / d
}

// XYZToLuv converts a CIE [XYZ] colour to [Luv] using the D65 reference white.
func XYZToLuv(c XYZ) Luv {
	w := WhitePointD65
	up, vp := uvPrime(c)
	unp, vnp := uvPrime(w)
	yr := c.Y / w.Y
	var l float64
	if yr > labDelta*labDelta*labDelta {
		l = 116*math.Cbrt(yr) - 16
	} else {
		l = (29.0 / 3.0) * (29.0 / 3.0) * (29.0 / 3.0) * yr
	}
	return Luv{
		L: l,
		U: 13 * l * (up - unp),
		V: 13 * l * (vp - vnp),
	}
}

// LuvToXYZ converts a [Luv] colour to CIE [XYZ] using the D65 reference white.
func LuvToXYZ(c Luv) XYZ {
	if c.L == 0 {
		return XYZ{}
	}
	w := WhitePointD65
	unp, vnp := uvPrime(w)
	up := c.U/(13*c.L) + unp
	vp := c.V/(13*c.L) + vnp
	var y float64
	if c.L > 8 {
		t := (c.L + 16) / 116
		y = w.Y * t * t * t
	} else {
		y = w.Y * c.L * (3.0 / 29.0) * (3.0 / 29.0) * (3.0 / 29.0)
	}
	if vp == 0 {
		return XYZ{X: 0, Y: y, Z: 0}
	}
	x := y * 9 * up / (4 * vp)
	z := y * (12 - 3*up - 20*vp) / (4 * vp)
	return XYZ{X: x, Y: y, Z: z}
}

// RGBToLuv converts a gamma-encoded sRGB colour to [Luv] (via [XYZ]).
func RGBToLuv(c RGB) Luv { return XYZToLuv(RGBToXYZ(c)) }

// LuvToRGB converts a [Luv] colour to gamma-encoded sRGB (via [XYZ]).
func LuvToRGB(c Luv) RGB { return XYZToRGB(LuvToXYZ(c)) }

// RGBToYCbCr converts a gamma-encoded [RGB] colour to full-range (JPEG) [YCbCr]
// using the ITU-R BT.601 luma coefficients.
func RGBToYCbCr(c RGB) YCbCr {
	y := 0.299*c.R + 0.587*c.G + 0.114*c.B
	return YCbCr{
		Y:  y,
		Cb: -0.168736*c.R - 0.331264*c.G + 0.5*c.B,
		Cr: 0.5*c.R - 0.418688*c.G - 0.081312*c.B,
	}
}

// YCbCrToRGB converts a full-range (JPEG) [YCbCr] colour back to [RGB].
func YCbCrToRGB(c YCbCr) RGB {
	return RGB{
		R: c.Y + 1.402*c.Cr,
		G: c.Y - 0.344136*c.Cb - 0.714136*c.Cr,
		B: c.Y + 1.772*c.Cb,
	}
}

// RGBToYUV converts a gamma-encoded [RGB] colour to analog [YUV] using the
// ITU-R BT.601 weights.
func RGBToYUV(c RGB) YUV {
	y := 0.299*c.R + 0.587*c.G + 0.114*c.B
	return YUV{
		Y: y,
		U: 0.492 * (c.B - y),
		V: 0.877 * (c.R - y),
	}
}

// YUVToRGB converts an analog [YUV] colour back to [RGB].
func YUVToRGB(c YUV) RGB {
	return RGB{
		R: c.Y + 1.13983*c.V,
		G: c.Y - 0.39465*c.U - 0.58060*c.V,
		B: c.Y + 2.03211*c.U,
	}
}

// RGBToCMYK converts a gamma-encoded [RGB] colour to [CMYK]. Pure black maps to
// K=1 with the chromatic channels set to 0.
func RGBToCMYK(c RGB) CMYK {
	max := math.Max(c.R, math.Max(c.G, c.B))
	k := 1 - max
	if k >= 1 {
		return CMYK{K: 1}
	}
	inv := 1 - k
	return CMYK{
		C: (1 - c.R - k) / inv,
		M: (1 - c.G - k) / inv,
		Y: (1 - c.B - k) / inv,
		K: k,
	}
}

// CMYKToRGB converts a [CMYK] colour back to gamma-encoded [RGB].
func CMYKToRGB(c CMYK) RGB {
	inv := 1 - c.K
	return RGB{
		R: (1 - c.C) * inv,
		G: (1 - c.M) * inv,
		B: (1 - c.Y) * inv,
	}
}
