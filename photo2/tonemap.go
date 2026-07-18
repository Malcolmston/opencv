package photo2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// photo2LogAvgLuminance returns the log-average (geometric mean) luminance of a
// linear HDR image and its maximum. eps guards against log(0).
func photo2LogAvgLuminance(lum *cv.FloatMat) (logAvg, maxL float64) {
	const eps = 1e-6
	var sumLog float64
	maxL = eps
	for _, v := range lum.Data {
		if v < 0 {
			v = 0
		}
		sumLog += math.Log(v + eps)
		if v > maxL {
			maxL = v
		}
	}
	logAvg = math.Exp(sumLog / float64(len(lum.Data)))
	return logAvg, maxL
}

// photo2Recolor rebuilds an 8-bit image from a new per-pixel display luminance
// ld given the original linear channels and their luminance lum. Colour is
// carried through as (channel/lum)^saturation, then gamma encoded.
func photo2Recolor(hdr []*cv.FloatMat, lum, ld *cv.FloatMat, saturation, gamma float64) *cv.Mat {
	rows, cols := lum.Rows, lum.Cols
	nch := len(hdr)
	invGamma := 1.0
	if gamma > 0 {
		invGamma = 1.0 / gamma
	}
	out := cv.NewMat(rows, cols, nch)
	total := rows * cols
	for i := 0; i < total; i++ {
		l := lum.Data[i]
		d := ld.Data[i]
		if d < 0 {
			d = 0
		}
		if nch == 1 {
			out.Data[i] = photo2Clamp8(math.Pow(photo2Clamp01(d), invGamma) * 255)
			continue
		}
		for c := 0; c < nch; c++ {
			var ratio float64
			if l > 1e-9 {
				ratio = hdr[c].Data[i] / l
			}
			v := math.Pow(ratio, saturation) * d
			out.Data[i*nch+c] = photo2Clamp8(math.Pow(photo2Clamp01(v), invGamma) * 255)
		}
	}
	return out
}

// GammaToneMap maps a linear HDR image to a display image by applying a simple
// gamma curve, out = clamp(in^(1/gamma))*255, independently per channel. It is
// the most naive tonemapper and a useful baseline. gamma must be positive; a
// value of 1 gives a plain linear clamp. channels holds one linear plane per
// colour channel.
func GammaToneMap(channels []*cv.FloatMat, gamma float64) *cv.Mat {
	photo2RequireChannels(channels, "GammaToneMap")
	if gamma <= 0 {
		gamma = 1
	}
	inv := 1.0 / gamma
	out := make([]*cv.FloatMat, len(channels))
	for c := range channels {
		p := cv.NewFloatMat(channels[c].Rows, channels[c].Cols)
		for i, v := range channels[c].Data {
			if v < 0 {
				v = 0
			}
			p.Data[i] = math.Pow(v, inv)
		}
		out[c] = p
	}
	return FromFloat(out)
}

// LogToneMap compresses a linear HDR image with a logarithmic luminance curve,
// Ld = log(1 + bias*Lw) / log(1 + bias*Lmax), then re-introduces colour and
// applies a display gamma of 2.2. Larger bias brightens the midtones. bias must
// be positive. This is a fast global operator with no local adaptation.
func LogToneMap(channels []*cv.FloatMat, bias float64) *cv.Mat {
	photo2RequireChannels(channels, "LogToneMap")
	if bias <= 0 {
		bias = 1
	}
	lum := LuminanceChannels(channels)
	_, maxL := photo2LogAvgLuminance(lum)
	denom := math.Log1p(bias * maxL)
	if denom <= 0 {
		denom = 1
	}
	ld := cv.NewFloatMat(lum.Rows, lum.Cols)
	for i, v := range lum.Data {
		if v < 0 {
			v = 0
		}
		ld.Data[i] = math.Log1p(bias*v) / denom
	}
	return photo2Recolor(channels, lum, ld, 1.0, 2.2)
}

// ReinhardParams configures [ReinhardToneMap].
type ReinhardParams struct {
	// Key (a) sets the target middle-grey; the log-average luminance is mapped
	// to roughly this value. Typical range 0.05–0.9, default 0.18.
	Key float64
	// WhitePoint is the smallest luminance (after keying) that is mapped to pure
	// white; a non-positive value uses the scaled scene maximum, giving the
	// pure Ld = L/(1+L) curve.
	WhitePoint float64
	// Saturation controls how strongly colour is preserved; 1 keeps the
	// original chroma, values below 1 desaturate.
	Saturation float64
	// Gamma is the display gamma applied at the end; 2.2 is standard.
	Gamma float64
}

// DefaultReinhardParams returns the recommended defaults for [ReinhardToneMap].
func DefaultReinhardParams() ReinhardParams {
	return ReinhardParams{Key: 0.18, WhitePoint: 0, Saturation: 1.0, Gamma: 2.2}
}

// ReinhardToneMap applies Reinhard et al.'s (2002) photographic global tone
// reproduction operator. Scene luminance is scaled so its log-average maps to
// the key value, then compressed with the extended operator
// Ld = L*(1 + L/Lwhite^2) / (1 + L); colour is restored and a display gamma
// applied. channels holds one linear plane per colour channel.
func ReinhardToneMap(channels []*cv.FloatMat, params ReinhardParams) *cv.Mat {
	photo2RequireChannels(channels, "ReinhardToneMap")
	if params.Key <= 0 {
		params.Key = 0.18
	}
	if params.Gamma <= 0 {
		params.Gamma = 2.2
	}
	if params.Saturation <= 0 {
		params.Saturation = 1.0
	}
	lum := LuminanceChannels(channels)
	logAvg, maxL := photo2LogAvgLuminance(lum)
	scale := params.Key / logAvg
	lwhite := params.WhitePoint
	if lwhite <= 0 {
		lwhite = scale * maxL
	} else {
		lwhite = scale * lwhite
	}
	lw2 := lwhite * lwhite
	ld := cv.NewFloatMat(lum.Rows, lum.Cols)
	for i, v := range lum.Data {
		if v < 0 {
			v = 0
		}
		l := scale * v
		ld.Data[i] = l * (1 + l/lw2) / (1 + l)
	}
	return photo2Recolor(channels, lum, ld, params.Saturation, params.Gamma)
}

// DragoParams configures [DragoToneMap].
type DragoParams struct {
	// Bias shapes the logarithmic base across the luminance range; it must lie
	// in (0,1]. Lower values brighten dark regions. Default 0.85.
	Bias float64
	// Saturation controls colour preservation; 1 keeps the original chroma.
	Saturation float64
	// Gamma is the display gamma applied at the end; 2.2 is standard.
	Gamma float64
}

// DefaultDragoParams returns the recommended defaults for [DragoToneMap].
func DefaultDragoParams() DragoParams {
	return DragoParams{Bias: 0.85, Saturation: 1.0, Gamma: 2.2}
}

// DragoToneMap applies Drago et al.'s (2003) adaptive logarithmic tone mapping.
// The luminance is compressed with a base that varies between log2 and log10
// according to the bias parameter, giving good contrast across a wide dynamic
// range; colour is then restored and a display gamma applied. channels holds one
// linear plane per colour channel.
func DragoToneMap(channels []*cv.FloatMat, params DragoParams) *cv.Mat {
	photo2RequireChannels(channels, "DragoToneMap")
	if params.Bias <= 0 || params.Bias > 1 {
		params.Bias = 0.85
	}
	if params.Gamma <= 0 {
		params.Gamma = 2.2
	}
	if params.Saturation <= 0 {
		params.Saturation = 1.0
	}
	lum := LuminanceChannels(channels)
	logAvg, maxL := photo2LogAvgLuminance(lum)
	// Normalise by the log-average so the key is stable, as in the paper.
	lwMax := maxL / logAvg
	if lwMax <= 0 {
		lwMax = 1
	}
	logExp := math.Log(params.Bias) / math.Log(0.5)
	denom := math.Log10(1 + lwMax)
	if denom <= 0 {
		denom = 1
	}
	ld := cv.NewFloatMat(lum.Rows, lum.Cols)
	for i, v := range lum.Data {
		if v < 0 {
			v = 0
		}
		lw := v / logAvg
		interp := math.Pow(lw/lwMax, logExp)
		ld.Data[i] = math.Log1p(lw) / (math.Log(2+8*interp) * denom)
	}
	return photo2Recolor(channels, lum, ld, params.Saturation, params.Gamma)
}

// DurandParams configures [DurandToneMap].
type DurandParams struct {
	// Contrast is the target base-layer contrast in log10 units; the base
	// dynamic range is compressed to this span. Default 4.0.
	Contrast float64
	// SigmaSpace is the spatial standard deviation of the bilateral filter used
	// to extract the base layer. Default 2.0.
	SigmaSpace float64
	// SigmaColor is the range (intensity) standard deviation of the bilateral
	// filter, in log10 luminance units. Default 0.4.
	SigmaColor float64
	// Saturation controls colour preservation; 1 keeps the original chroma.
	Saturation float64
	// Gamma is the display gamma applied at the end; 2.2 is standard.
	Gamma float64
}

// DefaultDurandParams returns the recommended defaults for [DurandToneMap].
func DefaultDurandParams() DurandParams {
	return DurandParams{Contrast: 4.0, SigmaSpace: 2.0, SigmaColor: 0.4, Saturation: 1.0, Gamma: 2.2}
}

// DurandToneMap applies Durand and Dorsey's (2002) fast bilateral-filtering tone
// mapping. Log-luminance is split by an edge-preserving bilateral filter into a
// large-scale base layer and a detail layer; the base is contrast-compressed
// while the detail is preserved, so local contrast and edges survive the global
// range reduction. channels holds one linear plane per colour channel.
func DurandToneMap(channels []*cv.FloatMat, params DurandParams) *cv.Mat {
	photo2RequireChannels(channels, "DurandToneMap")
	if params.Contrast <= 0 {
		params.Contrast = 4.0
	}
	if params.SigmaSpace <= 0 {
		params.SigmaSpace = 2.0
	}
	if params.SigmaColor <= 0 {
		params.SigmaColor = 0.4
	}
	if params.Gamma <= 0 {
		params.Gamma = 2.2
	}
	if params.Saturation <= 0 {
		params.Saturation = 1.0
	}
	lum := LuminanceChannels(channels)
	const eps = 1e-6
	logLum := cv.NewFloatMat(lum.Rows, lum.Cols)
	for i, v := range lum.Data {
		if v < 0 {
			v = 0
		}
		logLum.Data[i] = math.Log10(v + eps)
	}
	base := photo2BilateralFloat(logLum, params.SigmaSpace, params.SigmaColor)
	// Detail = logLum - base.
	minB := math.Inf(1)
	maxB := math.Inf(-1)
	for _, v := range base.Data {
		if v < minB {
			minB = v
		}
		if v > maxB {
			maxB = v
		}
	}
	span := maxB - minB
	if span <= 0 {
		span = 1
	}
	compression := params.Contrast / span
	ld := cv.NewFloatMat(lum.Rows, lum.Cols)
	for i := range ld.Data {
		detail := logLum.Data[i] - base.Data[i]
		// Compress the base, keep detail, offset so the brightest base maps to 1.
		logOut := base.Data[i]*compression + detail - maxB*compression
		ld.Data[i] = math.Pow(10, logOut)
	}
	return photo2Recolor(channels, lum, ld, params.Saturation, params.Gamma)
}
