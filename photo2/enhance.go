package photo2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// GammaCorrection applies a power-law transfer curve out = 255*(in/255)^gamma,
// independently per channel. gamma < 1 brightens, gamma > 1 darkens. gamma must
// be positive.
func GammaCorrection(img *cv.Mat, gamma float64) *cv.Mat {
	photo2RequireImage(img, "GammaCorrection")
	if gamma <= 0 {
		gamma = 1
	}
	var lut [256]uint8
	for i := 0; i < 256; i++ {
		lut[i] = photo2Clamp8(math.Pow(float64(i)/255, gamma) * 255)
	}
	return ApplyLUT(img, lut)
}

// LogTransform applies a logarithmic transfer curve
// out = 255*log(1+in)/log(256), which strongly brightens shadows and compresses
// highlights. It operates independently per channel.
func LogTransform(img *cv.Mat) *cv.Mat {
	photo2RequireImage(img, "LogTransform")
	scale := 255 / math.Log(256)
	var lut [256]uint8
	for i := 0; i < 256; i++ {
		lut[i] = photo2Clamp8(scale * math.Log1p(float64(i)))
	}
	return ApplyLUT(img, lut)
}

// ApplyLUT maps every sample of an image through the 256-entry lookup table,
// independently per channel. It is the building block for the tone-curve
// operators in this package.
func ApplyLUT(img *cv.Mat, lut [256]uint8) *cv.Mat {
	photo2RequireImage(img, "ApplyLUT")
	out := cv.NewMat(img.Rows, img.Cols, img.Channels)
	for i, v := range img.Data {
		out.Data[i] = lut[v]
	}
	return out
}

// UnsharpMask sharpens an image by subtracting a Gaussian-blurred copy: out =
// in + amount*(in - blur). ksize is retained for API familiarity but the blur
// is defined by sigma; amount scales the added high-frequency detail (typical
// 0.5–2). It operates per channel.
func UnsharpMask(img *cv.Mat, ksize int, sigma, amount float64) *cv.Mat {
	photo2RequireImage(img, "UnsharpMask")
	if sigma <= 0 {
		if ksize > 0 {
			sigma = float64(ksize) / 6
		} else {
			sigma = 1
		}
	}
	blur := GaussianBlur(img, sigma)
	out := cv.NewMat(img.Rows, img.Cols, img.Channels)
	for i := range img.Data {
		v := float64(img.Data[i])
		b := float64(blur.Data[i])
		out.Data[i] = photo2Clamp8(v + amount*(v-b))
	}
	return out
}

// Sharpen applies a fixed 3x3 sharpening kernel whose strength is set by amount
// (0 leaves the image unchanged). It operates per channel with reflected
// borders.
func Sharpen(img *cv.Mat, amount float64) *cv.Mat {
	photo2RequireImage(img, "Sharpen")
	rows, cols, nch := img.Rows, img.Cols, img.Channels
	out := cv.NewMat(rows, cols, nch)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			for c := 0; c < nch; c++ {
				center := float64(img.Data[(y*cols+x)*nch+c])
				up := float64(img.Data[(photo2Reflect(y-1, rows)*cols+x)*nch+c])
				dn := float64(img.Data[(photo2Reflect(y+1, rows)*cols+x)*nch+c])
				lf := float64(img.Data[(y*cols+photo2Reflect(x-1, cols))*nch+c])
				rt := float64(img.Data[(y*cols+photo2Reflect(x+1, cols))*nch+c])
				lap := 4*center - up - dn - lf - rt
				out.Data[(y*cols+x)*nch+c] = photo2Clamp8(center + amount*lap)
			}
		}
	}
	return out
}

// HistogramStretch performs per-channel linear contrast stretching, remapping
// each channel's observed [min,max] to the full [0,255] range. A constant
// channel is left unchanged.
func HistogramStretch(img *cv.Mat) *cv.Mat {
	photo2RequireImage(img, "HistogramStretch")
	nch := img.Channels
	total := img.Rows * img.Cols
	out := cv.NewMat(img.Rows, img.Cols, nch)
	for c := 0; c < nch; c++ {
		lo := 255
		hi := 0
		for i := 0; i < total; i++ {
			v := int(img.Data[i*nch+c])
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		span := hi - lo
		for i := 0; i < total; i++ {
			v := int(img.Data[i*nch+c])
			if span <= 0 {
				out.Data[i*nch+c] = uint8(v)
			} else {
				out.Data[i*nch+c] = photo2Clamp8(float64(v-lo) / float64(span) * 255)
			}
		}
	}
	return out
}

// HistogramEqualize performs global histogram equalisation. For a grayscale
// image the intensity CDF is used directly; for a colour image the operation is
// applied to luminance and the colour ratios are preserved, avoiding hue
// shifts.
func HistogramEqualize(img *cv.Mat) *cv.Mat {
	photo2RequireImage(img, "HistogramEqualize")
	if img.Channels == 1 {
		return photo2EqualizePlane8(img)
	}
	photo2RequireRGB(img, "HistogramEqualize")
	// Equalise luminance, scale channels by the ratio new/old.
	total := img.Rows * img.Cols
	lum := make([]uint8, total)
	for i := 0; i < total; i++ {
		r := float64(img.Data[i*3+0])
		g := float64(img.Data[i*3+1])
		b := float64(img.Data[i*3+2])
		lum[i] = photo2Clamp8(photo2Luma(r, g, b))
	}
	lutY := photo2EqualizeLUT(lum)
	out := cv.NewMat(img.Rows, img.Cols, 3)
	for i := 0; i < total; i++ {
		oldY := float64(lum[i])
		newY := float64(lutY[lum[i]])
		var ratio float64 = 1
		if oldY > 0 {
			ratio = newY / oldY
		}
		for c := 0; c < 3; c++ {
			out.Data[i*3+c] = photo2Clamp8(float64(img.Data[i*3+c]) * ratio)
		}
	}
	return out
}

// photo2EqualizeLUT builds the equalisation lookup table from an 8-bit sample
// slice.
func photo2EqualizeLUT(data []uint8) [256]uint8 {
	var hist [256]int
	for _, v := range data {
		hist[v]++
	}
	var lut [256]uint8
	total := len(data)
	// Find the first non-zero bin for the classic cdf_min normalisation.
	cdf := 0
	cdfMin := 0
	for i := 0; i < 256; i++ {
		if hist[i] != 0 {
			cdfMin = i
			break
		}
	}
	cdfMinVal := hist[cdfMin]
	denom := total - cdfMinVal
	if denom <= 0 {
		denom = 1
	}
	for i := 0; i < 256; i++ {
		cdf += hist[i]
		lut[i] = photo2Clamp8(float64(cdf-cdfMinVal) / float64(denom) * 255)
	}
	return lut
}

// photo2EqualizePlane8 equalises a single-channel 8-bit image.
func photo2EqualizePlane8(img *cv.Mat) *cv.Mat {
	lut := photo2EqualizeLUT(img.Data)
	out := cv.NewMat(img.Rows, img.Cols, 1)
	for i, v := range img.Data {
		out.Data[i] = lut[v]
	}
	return out
}

// AdjustBrightness adds delta (in 8-bit units, may be negative) to every sample,
// clamping to [0,255].
func AdjustBrightness(img *cv.Mat, delta float64) *cv.Mat {
	photo2RequireImage(img, "AdjustBrightness")
	out := cv.NewMat(img.Rows, img.Cols, img.Channels)
	for i, v := range img.Data {
		out.Data[i] = photo2Clamp8(float64(v) + delta)
	}
	return out
}

// AdjustContrast scales every sample around mid-grey (128): out = 128 +
// factor*(in-128). factor > 1 increases contrast, factor < 1 reduces it.
func AdjustContrast(img *cv.Mat, factor float64) *cv.Mat {
	photo2RequireImage(img, "AdjustContrast")
	out := cv.NewMat(img.Rows, img.Cols, img.Channels)
	for i, v := range img.Data {
		out.Data[i] = photo2Clamp8(128 + factor*(float64(v)-128))
	}
	return out
}

// AdjustSaturation scales colour saturation around the per-pixel luminance:
// out_c = luma + factor*(in_c - luma). factor > 1 boosts saturation, 0 gives a
// grayscale image, values above 1 intensify colour. The input must be
// three-channel.
func AdjustSaturation(img *cv.Mat, factor float64) *cv.Mat {
	photo2RequireRGB(img, "AdjustSaturation")
	out := cv.NewMat(img.Rows, img.Cols, 3)
	total := img.Rows * img.Cols
	for i := 0; i < total; i++ {
		r := float64(img.Data[i*3+0])
		g := float64(img.Data[i*3+1])
		b := float64(img.Data[i*3+2])
		y := photo2Luma(r, g, b)
		out.Data[i*3+0] = photo2Clamp8(y + factor*(r-y))
		out.Data[i*3+1] = photo2Clamp8(y + factor*(g-y))
		out.Data[i*3+2] = photo2Clamp8(y + factor*(b-y))
	}
	return out
}

// ExposureCompensate multiplies the linear scene value by 2^stops, simulating an
// exposure adjustment. Positive stops brighten, negative darken. The operation
// is performed in a linearised (gamma 2.2 removed) space and re-encoded, so it
// behaves like a real exposure change rather than a simple gain.
func ExposureCompensate(img *cv.Mat, stops float64) *cv.Mat {
	photo2RequireImage(img, "ExposureCompensate")
	gain := math.Pow(2, stops)
	var lut [256]uint8
	for i := 0; i < 256; i++ {
		lin := math.Pow(float64(i)/255, 2.2) * gain
		lut[i] = photo2Clamp8(math.Pow(photo2Clamp01(lin), 1/2.2) * 255)
	}
	return ApplyLUT(img, lut)
}

// Blend returns the per-sample convex combination alpha*a + (1-alpha)*b. The two
// images must share dimensions and channel count; alpha is clamped to [0,1].
func Blend(a, b *cv.Mat, alpha float64) *cv.Mat {
	photo2RequireImage(a, "Blend")
	photo2RequireImage(b, "Blend")
	if a.Rows != b.Rows || a.Cols != b.Cols || a.Channels != b.Channels {
		panic("photo2: Blend images must share dimensions and channels")
	}
	alpha = photo2Clamp01(alpha)
	out := cv.NewMat(a.Rows, a.Cols, a.Channels)
	for i := range a.Data {
		out.Data[i] = photo2Clamp8(alpha*float64(a.Data[i]) + (1-alpha)*float64(b.Data[i]))
	}
	return out
}

// LocalContrast enhances mid-scale local contrast (a "clarity" control) by
// amplifying the difference between the image and a Gaussian-blurred version at
// the given sigma. amount scales the effect; it operates per channel.
func LocalContrast(img *cv.Mat, sigma, amount float64) *cv.Mat {
	photo2RequireImage(img, "LocalContrast")
	blur := GaussianBlur(img, sigma)
	out := cv.NewMat(img.Rows, img.Cols, img.Channels)
	for i := range img.Data {
		v := float64(img.Data[i])
		b := float64(blur.Data[i])
		out.Data[i] = photo2Clamp8(v + amount*(v-b))
	}
	return out
}

// Vignette darkens the image toward the corners with a smooth radial falloff,
// simulating a lens vignette. strength in [0,1] sets how dark the corners
// become (0 is no effect, 1 fully black corners). It operates per channel.
func Vignette(img *cv.Mat, strength float64) *cv.Mat {
	photo2RequireImage(img, "Vignette")
	strength = photo2Clamp01(strength)
	rows, cols, nch := img.Rows, img.Cols, img.Channels
	cy := float64(rows-1) / 2
	cx := float64(cols-1) / 2
	maxD := math.Hypot(cx, cy)
	if maxD <= 0 {
		maxD = 1
	}
	out := cv.NewMat(rows, cols, nch)
	for y := 0; y < rows; y++ {
		for x := 0; x < cols; x++ {
			d := math.Hypot(float64(x)-cx, float64(y)-cy) / maxD
			factor := 1 - strength*d*d
			if factor < 0 {
				factor = 0
			}
			for c := 0; c < nch; c++ {
				i := (y*cols + x) * nch
				out.Data[i+c] = photo2Clamp8(float64(img.Data[i+c]) * factor)
			}
		}
	}
	return out
}
