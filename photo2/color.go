package photo2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// photo2SRGBToLinear removes the sRGB display gamma from a value in [0,1].
func photo2SRGBToLinear(c float64) float64 {
	if c <= 0.04045 {
		return c / 12.92
	}
	return math.Pow((c+0.055)/1.055, 2.4)
}

// photo2LinearToSRGB applies the sRGB display gamma to a linear value in [0,1].
func photo2LinearToSRGB(c float64) float64 {
	if c <= 0.0031308 {
		return 12.92 * c
	}
	return 1.055*math.Pow(c, 1/2.4) - 0.055
}

// D65 reference white for CIELab.
const (
	photo2Xn = 0.95047
	photo2Yn = 1.0
	photo2Zn = 1.08883
)

// RGBToXYZ converts an 8-bit sRGB image to linear CIE XYZ tristimulus planes
// (D65). The input is gamma-decoded before the linear transform. The three
// returned planes hold X, Y and Z, each nominally in [0,1] for in-gamut colours.
// The input must be three-channel.
func RGBToXYZ(img *cv.Mat) []*cv.FloatMat {
	photo2RequireRGB(img, "RGBToXYZ")
	total := img.Rows * img.Cols
	X := cv.NewFloatMat(img.Rows, img.Cols)
	Y := cv.NewFloatMat(img.Rows, img.Cols)
	Z := cv.NewFloatMat(img.Rows, img.Cols)
	for i := 0; i < total; i++ {
		r := photo2SRGBToLinear(float64(img.Data[i*3+0]) / 255)
		g := photo2SRGBToLinear(float64(img.Data[i*3+1]) / 255)
		b := photo2SRGBToLinear(float64(img.Data[i*3+2]) / 255)
		X.Data[i] = 0.4124*r + 0.3576*g + 0.1805*b
		Y.Data[i] = 0.2126*r + 0.7152*g + 0.0722*b
		Z.Data[i] = 0.0193*r + 0.1192*g + 0.9505*b
	}
	return []*cv.FloatMat{X, Y, Z}
}

// XYZToRGB converts linear CIE XYZ planes (D65) back to an 8-bit sRGB image,
// applying the inverse linear transform and the sRGB gamma. channels must hold
// exactly three planes (X, Y, Z) of equal size.
func XYZToRGB(channels []*cv.FloatMat) *cv.Mat {
	rows, cols := photo2RequireChannels(channels, "XYZToRGB")
	if len(channels) != 3 {
		panic("photo2: XYZToRGB requires 3 planes")
	}
	X, Y, Z := channels[0], channels[1], channels[2]
	out := cv.NewMat(rows, cols, 3)
	total := rows * cols
	for i := 0; i < total; i++ {
		r := 3.2406*X.Data[i] - 1.5372*Y.Data[i] - 0.4986*Z.Data[i]
		g := -0.9689*X.Data[i] + 1.8758*Y.Data[i] + 0.0415*Z.Data[i]
		b := 0.0557*X.Data[i] - 0.2040*Y.Data[i] + 1.0570*Z.Data[i]
		out.Data[i*3+0] = photo2Clamp8(photo2LinearToSRGB(photo2Clamp01(r)) * 255)
		out.Data[i*3+1] = photo2Clamp8(photo2LinearToSRGB(photo2Clamp01(g)) * 255)
		out.Data[i*3+2] = photo2Clamp8(photo2LinearToSRGB(photo2Clamp01(b)) * 255)
	}
	return out
}

// photo2LabF is the CIELab nonlinearity.
func photo2LabF(t float64) float64 {
	const eps = 216.0 / 24389.0 // 0.008856
	if t > eps {
		return math.Cbrt(t)
	}
	return (24389.0/27.0*t + 16) / 116
}

// photo2LabFInv is the inverse of photo2LabF.
func photo2LabFInv(t float64) float64 {
	const eps = 6.0 / 29.0
	if t > eps {
		return t * t * t
	}
	return 3 * eps * eps * (t - 4.0/29.0)
}

// RGBToLab converts an 8-bit sRGB image to CIELab (D65). The three returned
// planes hold L in [0,100] and a, b roughly in [-128,127]. The input must be
// three-channel.
func RGBToLab(img *cv.Mat) []*cv.FloatMat {
	photo2RequireRGB(img, "RGBToLab")
	xyz := RGBToXYZ(img)
	total := img.Rows * img.Cols
	L := cv.NewFloatMat(img.Rows, img.Cols)
	A := cv.NewFloatMat(img.Rows, img.Cols)
	B := cv.NewFloatMat(img.Rows, img.Cols)
	for i := 0; i < total; i++ {
		fx := photo2LabF(xyz[0].Data[i] / photo2Xn)
		fy := photo2LabF(xyz[1].Data[i] / photo2Yn)
		fz := photo2LabF(xyz[2].Data[i] / photo2Zn)
		L.Data[i] = 116*fy - 16
		A.Data[i] = 500 * (fx - fy)
		B.Data[i] = 200 * (fy - fz)
	}
	return []*cv.FloatMat{L, A, B}
}

// LabToRGB converts CIELab planes (D65) back to an 8-bit sRGB image. channels
// must hold exactly three planes (L, a, b) of equal size.
func LabToRGB(channels []*cv.FloatMat) *cv.Mat {
	rows, cols := photo2RequireChannels(channels, "LabToRGB")
	if len(channels) != 3 {
		panic("photo2: LabToRGB requires 3 planes")
	}
	L, A, B := channels[0], channels[1], channels[2]
	X := cv.NewFloatMat(rows, cols)
	Y := cv.NewFloatMat(rows, cols)
	Z := cv.NewFloatMat(rows, cols)
	total := rows * cols
	for i := 0; i < total; i++ {
		fy := (L.Data[i] + 16) / 116
		fx := fy + A.Data[i]/500
		fz := fy - B.Data[i]/200
		X.Data[i] = photo2Xn * photo2LabFInv(fx)
		Y.Data[i] = photo2Yn * photo2LabFInv(fy)
		Z.Data[i] = photo2Zn * photo2LabFInv(fz)
	}
	return XYZToRGB([]*cv.FloatMat{X, Y, Z})
}

// photo2RGBToLAB converts a raw RGB triple in [0,1] to Ruderman's decorrelated
// lαβ space (via LMS and a base-10 logarithm), the space used by Reinhard's
// colour transfer.
func photo2RGBToLAB(r, g, b float64) (l, alpha, beta float64) {
	L := 0.3811*r + 0.5783*g + 0.0402*b
	M := 0.1967*r + 0.7244*g + 0.0782*b
	S := 0.0241*r + 0.1288*g + 0.8444*b
	const floor = 1e-4
	if L < floor {
		L = floor
	}
	if M < floor {
		M = floor
	}
	if S < floor {
		S = floor
	}
	L = math.Log10(L)
	M = math.Log10(M)
	S = math.Log10(S)
	l = (L + M + S) / math.Sqrt(3)
	alpha = (L + M - 2*S) / math.Sqrt(6)
	beta = (L - M) / math.Sqrt(2)
	return l, alpha, beta
}

// photo2LABToRGB inverts photo2RGBToLAB, returning an RGB triple in [0,1].
func photo2LABToRGB(l, alpha, beta float64) (r, g, b float64) {
	L := l/math.Sqrt(3) + alpha/math.Sqrt(6) + beta/math.Sqrt(2)
	M := l/math.Sqrt(3) + alpha/math.Sqrt(6) - beta/math.Sqrt(2)
	S := l/math.Sqrt(3) - 2*alpha/math.Sqrt(6)
	L = math.Pow(10, L)
	M = math.Pow(10, M)
	S = math.Pow(10, S)
	r = 4.4679*L - 3.5873*M + 0.1193*S
	g = -1.2186*L + 2.3809*M - 0.1624*S
	b = 0.0497*L - 0.2439*M + 1.2045*S
	return r, g, b
}

// ColorTransferReinhard transfers the colour statistics of target onto src using
// Reinhard et al.'s (2001) method. Both images are converted to the decorrelated
// lαβ space; each channel of src is shifted and scaled so its mean and standard
// deviation match target's, then converted back to RGB. The result keeps src's
// content but takes on target's overall palette and tone. Both inputs must be
// three-channel; they may differ in size.
func ColorTransferReinhard(src, target *cv.Mat) *cv.Mat {
	photo2RequireRGB(src, "ColorTransferReinhard")
	photo2RequireRGB(target, "ColorTransferReinhard")
	sMean, sStd := photo2LABStats(src)
	tMean, tStd := photo2LABStats(target)
	total := src.Rows * src.Cols
	out := cv.NewMat(src.Rows, src.Cols, 3)
	for i := 0; i < total; i++ {
		r := float64(src.Data[i*3+0]) / 255
		g := float64(src.Data[i*3+1]) / 255
		b := float64(src.Data[i*3+2]) / 255
		l, a, be := photo2RGBToLAB(r, g, b)
		vals := [3]float64{l, a, be}
		for c := 0; c < 3; c++ {
			scale := 1.0
			if sStd[c] > 1e-9 {
				scale = tStd[c] / sStd[c]
			}
			vals[c] = (vals[c]-sMean[c])*scale + tMean[c]
		}
		nr, ng, nb := photo2LABToRGB(vals[0], vals[1], vals[2])
		out.Data[i*3+0] = photo2Clamp8(photo2Clamp01(nr) * 255)
		out.Data[i*3+1] = photo2Clamp8(photo2Clamp01(ng) * 255)
		out.Data[i*3+2] = photo2Clamp8(photo2Clamp01(nb) * 255)
	}
	return out
}

// photo2LABStats returns the per-channel mean and standard deviation of an image
// in lαβ space.
func photo2LABStats(img *cv.Mat) (mean, std [3]float64) {
	total := img.Rows * img.Cols
	for i := 0; i < total; i++ {
		r := float64(img.Data[i*3+0]) / 255
		g := float64(img.Data[i*3+1]) / 255
		b := float64(img.Data[i*3+2]) / 255
		l, a, be := photo2RGBToLAB(r, g, b)
		mean[0] += l
		mean[1] += a
		mean[2] += be
	}
	n := float64(total)
	for c := 0; c < 3; c++ {
		mean[c] /= n
	}
	for i := 0; i < total; i++ {
		r := float64(img.Data[i*3+0]) / 255
		g := float64(img.Data[i*3+1]) / 255
		b := float64(img.Data[i*3+2]) / 255
		l, a, be := photo2RGBToLAB(r, g, b)
		vals := [3]float64{l, a, be}
		for c := 0; c < 3; c++ {
			d := vals[c] - mean[c]
			std[c] += d * d
		}
	}
	for c := 0; c < 3; c++ {
		std[c] = math.Sqrt(std[c] / n)
	}
	return mean, std
}

// GrayWorldWhiteBalance applies the grey-world assumption: it scales each colour
// channel so their means become equal to the overall mean, neutralising a global
// colour cast. The input must be three-channel.
func GrayWorldWhiteBalance(img *cv.Mat) *cv.Mat {
	photo2RequireRGB(img, "GrayWorldWhiteBalance")
	total := img.Rows * img.Cols
	var sum [3]float64
	for i := 0; i < total; i++ {
		for c := 0; c < 3; c++ {
			sum[c] += float64(img.Data[i*3+c])
		}
	}
	var mean [3]float64
	gray := 0.0
	for c := 0; c < 3; c++ {
		mean[c] = sum[c] / float64(total)
		gray += mean[c]
	}
	gray /= 3
	var gain [3]float64
	for c := 0; c < 3; c++ {
		if mean[c] > 1e-9 {
			gain[c] = gray / mean[c]
		} else {
			gain[c] = 1
		}
	}
	out := cv.NewMat(img.Rows, img.Cols, 3)
	for i := 0; i < total; i++ {
		for c := 0; c < 3; c++ {
			out.Data[i*3+c] = photo2Clamp8(float64(img.Data[i*3+c]) * gain[c])
		}
	}
	return out
}

// SimpleWhiteBalance stretches each channel between the pLow and pHigh
// percentiles of its histogram, clipping outliers, which both corrects colour
// casts and expands contrast. pLow and pHigh are fractions in [0,1) with
// pLow < pHigh (e.g. 0.02 and 0.98). The input must be three-channel.
func SimpleWhiteBalance(img *cv.Mat, pLow, pHigh float64) *cv.Mat {
	photo2RequireRGB(img, "SimpleWhiteBalance")
	if pLow < 0 {
		pLow = 0
	}
	if pHigh > 1 {
		pHigh = 1
	}
	if pLow >= pHigh {
		pLow, pHigh = 0.02, 0.98
	}
	total := img.Rows * img.Cols
	out := cv.NewMat(img.Rows, img.Cols, 3)
	for c := 0; c < 3; c++ {
		var hist [256]int
		for i := 0; i < total; i++ {
			hist[img.Data[i*3+c]]++
		}
		loCount := int(pLow * float64(total))
		hiCount := int(pHigh * float64(total))
		lo, hi := 0, 255
		cum := 0
		for v := 0; v < 256; v++ {
			cum += hist[v]
			if cum > loCount {
				lo = v
				break
			}
		}
		cum = 0
		for v := 0; v < 256; v++ {
			cum += hist[v]
			if cum >= hiCount {
				hi = v
				break
			}
		}
		span := hi - lo
		if span <= 0 {
			span = 1
		}
		for i := 0; i < total; i++ {
			v := float64(int(img.Data[i*3+c])-lo) / float64(span) * 255
			out.Data[i*3+c] = photo2Clamp8(v)
		}
	}
	return out
}
