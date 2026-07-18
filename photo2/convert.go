package photo2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// ToFloat converts an 8-bit image to a slice of per-channel [cv.FloatMat]
// planes with samples scaled into [0,1]. The result has one plane per channel,
// all sharing the image dimensions. It is the standard entry point for the
// floating-point pipeline used by tonemapping and colour grading.
func ToFloat(img *cv.Mat) []*cv.FloatMat {
	photo2RequireImage(img, "ToFloat")
	planes := make([]*cv.FloatMat, img.Channels)
	total := img.Rows * img.Cols
	for c := 0; c < img.Channels; c++ {
		p := cv.NewFloatMat(img.Rows, img.Cols)
		for i := 0; i < total; i++ {
			p.Data[i] = float64(img.Data[i*img.Channels+c]) / 255
		}
		planes[c] = p
	}
	return planes
}

// FromFloat merges per-channel float planes back into an 8-bit image. Each
// value is scaled by 255, rounded and clamped into [0,255], so channels holding
// values outside [0,1] are saturated. All planes must share dimensions.
func FromFloat(channels []*cv.FloatMat) *cv.Mat {
	rows, cols := photo2RequireChannels(channels, "FromFloat")
	nch := len(channels)
	out := cv.NewMat(rows, cols, nch)
	total := rows * cols
	for c := 0; c < nch; c++ {
		p := channels[c]
		for i := 0; i < total; i++ {
			out.Data[i*nch+c] = photo2Clamp8(p.Data[i] * 255)
		}
	}
	return out
}

// Grayscale converts an image to a single-channel 8-bit image using the Rec.709
// luma weights. A single-channel input is returned as a clone.
func Grayscale(img *cv.Mat) *cv.Mat {
	photo2RequireImage(img, "Grayscale")
	if img.Channels == 1 {
		return img.Clone()
	}
	if img.Channels != 3 {
		photo2RequireRGB(img, "Grayscale")
	}
	out := cv.NewMat(img.Rows, img.Cols, 1)
	total := img.Rows * img.Cols
	for i := 0; i < total; i++ {
		r := float64(img.Data[i*3+0])
		g := float64(img.Data[i*3+1])
		b := float64(img.Data[i*3+2])
		out.Data[i] = photo2Clamp8(photo2Luma(r, g, b))
	}
	return out
}

// Luminance returns the Rec.709 relative luminance of an image as a
// single-channel [cv.FloatMat] scaled into [0,1]. A single-channel input is
// interpreted directly as luminance. The image is treated as display-referred
// (no gamma linearisation is applied).
func Luminance(img *cv.Mat) *cv.FloatMat {
	photo2RequireImage(img, "Luminance")
	out := cv.NewFloatMat(img.Rows, img.Cols)
	total := img.Rows * img.Cols
	if img.Channels == 1 {
		for i := 0; i < total; i++ {
			out.Data[i] = float64(img.Data[i]) / 255
		}
		return out
	}
	for i := 0; i < total; i++ {
		r := float64(img.Data[i*img.Channels+0]) / 255
		g := float64(img.Data[i*img.Channels+1]) / 255
		b := float64(img.Data[i*img.Channels+2]) / 255
		out.Data[i] = photo2Luma(r, g, b)
	}
	return out
}

// LuminanceChannels returns the Rec.709 luminance of a set of float RGB planes.
// A single plane is returned as a clone. channels must have one or three planes.
func LuminanceChannels(channels []*cv.FloatMat) *cv.FloatMat {
	rows, cols := photo2RequireChannels(channels, "LuminanceChannels")
	if len(channels) == 1 {
		return photo2CloneFloat(channels[0])
	}
	out := cv.NewFloatMat(rows, cols)
	total := rows * cols
	for i := 0; i < total; i++ {
		out.Data[i] = photo2Luma(channels[0].Data[i], channels[1].Data[i], channels[2].Data[i])
	}
	return out
}

// Clamp01 returns a copy of f with every sample clamped into [0,1].
func Clamp01(f *cv.FloatMat) *cv.FloatMat {
	photo2RequireFloat(f, "Clamp01")
	out := cv.NewFloatMat(f.Rows, f.Cols)
	for i, v := range f.Data {
		out.Data[i] = photo2Clamp01(v)
	}
	return out
}

// Normalize returns a copy of f linearly rescaled so its minimum maps to 0 and
// its maximum to 1. A constant matrix is returned as all zeros.
func Normalize(f *cv.FloatMat) *cv.FloatMat {
	photo2RequireFloat(f, "Normalize")
	lo := math.Inf(1)
	hi := math.Inf(-1)
	for _, v := range f.Data {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	out := cv.NewFloatMat(f.Rows, f.Cols)
	span := hi - lo
	if span <= 0 {
		return out
	}
	for i, v := range f.Data {
		out.Data[i] = (v - lo) / span
	}
	return out
}
