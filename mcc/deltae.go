package mcc

import "math"

// DeltaE94 returns the CIE94 color difference between two CIE L*a*b* colors
// using the graphic-arts weighting (kL=kC=kH=1, K1=0.045, K2=0.015). CIE94
// improves on CIE76 by scaling the chroma and hue differences by the sample
// chroma, which better matches perceived difference for saturated colors. Use
// [DeltaE94Textiles] for the textile weighting.
func DeltaE94(lab1, lab2 [3]float64) float64 {
	return deltaE94(lab1, lab2, 1, 0.045, 0.015)
}

// DeltaE94Textiles returns the CIE94 color difference under the textile
// application weighting (kL=2, K1=0.048, K2=0.014).
func DeltaE94Textiles(lab1, lab2 [3]float64) float64 {
	return deltaE94(lab1, lab2, 2, 0.048, 0.014)
}

// deltaE94 implements CIE94 with explicit lightness weight kL and the K1/K2
// constants that set the chroma and hue scaling. kC and kH are fixed at 1, as
// in both standard applications.
func deltaE94(lab1, lab2 [3]float64, kL, k1, k2 float64) float64 {
	dl := lab1[0] - lab2[0]
	c1 := math.Hypot(lab1[1], lab1[2])
	c2 := math.Hypot(lab2[1], lab2[2])
	dc := c1 - c2
	da := lab1[1] - lab2[1]
	db := lab1[2] - lab2[2]
	// dH^2 = da^2 + db^2 - dC^2, clamped at 0 against round-off.
	dh2 := da*da + db*db - dc*dc
	if dh2 < 0 {
		dh2 = 0
	}
	sl := 1.0
	sc := 1 + k1*c1
	sh := 1 + k2*c1
	tl := dl / (kL * sl)
	tc := dc / sc
	th := math.Sqrt(dh2) / sh
	return math.Sqrt(tl*tl + tc*tc + th*th)
}

// DeltaE2000 returns the CIEDE2000 color difference between two CIE L*a*b*
// colors with all parametric weights kL=kC=kH=1. CIEDE2000 is the current CIE
// recommendation and adds lightness-, chroma- and hue-dependent weighting plus a
// blue-region rotation term to CIE94. The implementation follows Sharma,
// Wu & Dalal (2005) and matches their published test data. Use
// [DeltaE2000Weighted] to supply non-unit parametric factors.
func DeltaE2000(lab1, lab2 [3]float64) float64 {
	return DeltaE2000Weighted(lab1, lab2, 1, 1, 1)
}

// DeltaE2000Weighted returns the CIEDE2000 color difference with explicit
// parametric weighting factors kL, kC and kH (all 1 for the default metric).
func DeltaE2000Weighted(lab1, lab2 [3]float64, kL, kC, kH float64) float64 {
	const deg = math.Pi / 180
	l1, a1, b1 := lab1[0], lab1[1], lab1[2]
	l2, a2, b2 := lab2[0], lab2[1], lab2[2]

	c1 := math.Hypot(a1, b1)
	c2 := math.Hypot(a2, b2)
	cBar := (c1 + c2) / 2

	cBar7 := math.Pow(cBar, 7)
	g := 0.5 * (1 - math.Sqrt(cBar7/(cBar7+math.Pow(25, 7))))

	a1p := (1 + g) * a1
	a2p := (1 + g) * a2
	c1p := math.Hypot(a1p, b1)
	c2p := math.Hypot(a2p, b2)

	h1p := atan2Deg(b1, a1p)
	h2p := atan2Deg(b2, a2p)

	dLp := l2 - l1
	dCp := c2p - c1p

	var dhp float64
	switch {
	case c1p*c2p == 0:
		dhp = 0
	case math.Abs(h2p-h1p) <= 180:
		dhp = h2p - h1p
	case h2p-h1p > 180:
		dhp = h2p - h1p - 360
	default:
		dhp = h2p - h1p + 360
	}
	dHp := 2 * math.Sqrt(c1p*c2p) * math.Sin(dhp/2*deg)

	lBarp := (l1 + l2) / 2
	cBarp := (c1p + c2p) / 2

	var hBarp float64
	switch {
	case c1p*c2p == 0:
		hBarp = h1p + h2p
	case math.Abs(h1p-h2p) <= 180:
		hBarp = (h1p + h2p) / 2
	case h1p+h2p < 360:
		hBarp = (h1p + h2p + 360) / 2
	default:
		hBarp = (h1p + h2p - 360) / 2
	}

	t := 1 -
		0.17*math.Cos((hBarp-30)*deg) +
		0.24*math.Cos((2*hBarp)*deg) +
		0.32*math.Cos((3*hBarp+6)*deg) -
		0.20*math.Cos((4*hBarp-63)*deg)

	hExp := (hBarp - 275) / 25
	dTheta := 30 * math.Exp(-(hExp * hExp))
	cBarp7 := math.Pow(cBarp, 7)
	rc := 2 * math.Sqrt(cBarp7/(cBarp7+math.Pow(25, 7)))
	rt := -rc * math.Sin(2*dTheta*deg)

	lBarp50 := (lBarp - 50) * (lBarp - 50)
	sl := 1 + (0.015*lBarp50)/math.Sqrt(20+lBarp50)
	sc := 1 + 0.045*cBarp
	sh := 1 + 0.015*cBarp*t

	termL := dLp / (kL * sl)
	termC := dCp / (kC * sc)
	termH := dHp / (kH * sh)
	return math.Sqrt(termL*termL + termC*termC + termH*termH + rt*termC*termH)
}

// atan2Deg returns atan2(y,x) in degrees within [0,360).
func atan2Deg(y, x float64) float64 {
	if y == 0 && x == 0 {
		return 0
	}
	d := math.Atan2(y, x) * 180 / math.Pi
	if d < 0 {
		d += 360
	}
	return d
}

// DeltaECMC returns the CMC l:c color difference between two CIE L*a*b* colors.
// The lightness and chroma weights l and c set the tolerance ellipsoid: CMC(2:1)
// (l=2, c=1) is the common acceptability threshold and CMC(1:1) the
// perceptibility threshold. The metric is asymmetric — lab1 is the reference and
// lab2 the sample. See [DeltaECMCAcceptability] and [DeltaECMCPerceptibility]
// for the two standard settings.
func DeltaECMC(lab1, lab2 [3]float64, l, c float64) float64 {
	const deg = math.Pi / 180
	l1, a1, b1 := lab1[0], lab1[1], lab1[2]
	dl := l1 - lab2[0]

	c1 := math.Hypot(a1, b1)
	c2 := math.Hypot(lab2[1], lab2[2])
	dc := c1 - c2

	da := a1 - lab2[1]
	db := b1 - lab2[2]
	dh2 := da*da + db*db - dc*dc
	if dh2 < 0 {
		dh2 = 0
	}

	var sl float64
	if l1 < 16 {
		sl = 0.511
	} else {
		sl = 0.040975 * l1 / (1 + 0.01765*l1)
	}
	sc := 0.0638*c1/(1+0.0131*c1) + 0.638

	h1 := atan2Deg(b1, a1)
	var ft float64
	if h1 >= 164 && h1 <= 345 {
		ft = 0.56 + math.Abs(0.2*math.Cos((h1+168)*deg))
	} else {
		ft = 0.36 + math.Abs(0.4*math.Cos((h1+35)*deg))
	}
	c1p4 := math.Pow(c1, 4)
	f := math.Sqrt(c1p4 / (c1p4 + 1900))
	sh := sc * (f*ft + 1 - f)

	tl := dl / (l * sl)
	tc := dc / (c * sc)
	th := math.Sqrt(dh2) / sh
	return math.Sqrt(tl*tl + tc*tc + th*th)
}

// DeltaECMCAcceptability returns the CMC(2:1) color difference, the common
// pass/fail acceptability tolerance used in industry.
func DeltaECMCAcceptability(lab1, lab2 [3]float64) float64 {
	return DeltaECMC(lab1, lab2, 2, 1)
}

// DeltaECMCPerceptibility returns the CMC(1:1) color difference, the threshold
// of perceptible difference.
func DeltaECMCPerceptibility(lab1, lab2 [3]float64) float64 {
	return DeltaECMC(lab1, lab2, 1, 1)
}

// DeltaE2000RGB returns the CIEDE2000 difference between two 8-bit sRGB colors,
// converting both to Lab (D65) first — the perceptual analogue of [DeltaERGB].
func DeltaE2000RGB(a, b [3]uint8) float64 {
	return DeltaE2000(RGBToLab(a[0], a[1], a[2]), RGBToLab(b[0], b[1], b[2]))
}
