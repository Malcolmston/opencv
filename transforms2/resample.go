package transforms2

import (
	"math"

	cv "github.com/malcolmston/opencv"
)

// Interpolation selects the resampling kernel used when a transform samples the
// source image at fractional coordinates.
type Interpolation int

const (
	// InterpNearest selects the nearest source sample (fast, blocky).
	InterpNearest Interpolation = iota
	// InterpBilinear performs bilinear interpolation of the four nearest
	// samples.
	InterpBilinear
	// InterpBicubic performs bicubic (Keys, a=-0.75) interpolation of the
	// sixteen nearest samples, giving smoother results than bilinear.
	InterpBicubic
)

// BorderMode selects how out-of-range coordinates are handled while sampling.
type BorderMode int

const (
	// BorderConstant substitutes a constant fill value for out-of-range
	// samples.
	BorderConstant BorderMode = iota
	// BorderReplicate repeats the nearest edge sample (aaaaaa|abcdefgh|hhhhhh).
	BorderReplicate
	// BorderReflect mirrors samples about the edge without repeating it
	// (gfedcb|abcdefgh|gfedcba), matching OpenCV's BORDER_REFLECT_101.
	BorderReflect
	// BorderWrap tiles the image periodically (abcdefgh|abcdefgh|abcdefgh).
	BorderWrap
)

// transforms2borderIndex maps a possibly out-of-range index i into [0, n)
// according to mode. The boolean is false only for BorderConstant when i is
// out of range, signalling that the constant fill value must be used instead.
func transforms2borderIndex(i, n int, mode BorderMode) (int, bool) {
	if i >= 0 && i < n {
		return i, true
	}
	if n == 1 {
		if mode == BorderConstant {
			return 0, false
		}
		return 0, true
	}
	switch mode {
	case BorderReplicate:
		if i < 0 {
			return 0, true
		}
		return n - 1, true
	case BorderReflect:
		period := 2 * (n - 1)
		p := ((i % period) + period) % period
		if p >= n {
			p = period - p
		}
		return p, true
	case BorderWrap:
		return ((i % n) + n) % n, true
	default: // BorderConstant
		return 0, false
	}
}

// transforms2fetch returns channel c of the source sample at integer pixel
// (xi, yi), applying the border policy. Out-of-range constant borders return
// fill.
func transforms2fetch(src *cv.Mat, xi, yi, c int, border BorderMode, fill float64) float64 {
	ix, okx := transforms2borderIndex(xi, src.Cols, border)
	if !okx {
		return fill
	}
	iy, oky := transforms2borderIndex(yi, src.Rows, border)
	if !oky {
		return fill
	}
	return float64(src.Data[(iy*src.Cols+ix)*src.Channels+c])
}

// transforms2cubic evaluates the Keys cubic convolution kernel (a=-0.75) at
// distance t; its four taps partition unity, so a constant image is preserved.
func transforms2cubic(t float64) float64 {
	const a = -0.75
	t = math.Abs(t)
	switch {
	case t <= 1:
		return ((a+2)*t-(a+3))*t*t + 1
	case t < 2:
		return (((t-5)*t+8)*t - 4) * a
	default:
		return 0
	}
}

// SampleChannel returns the interpolated value of channel c of src at the
// fractional coordinate (x, y) using the given interpolation and border mode.
// The result is a float in the sample range [0, 255] and is not clamped to an
// integer. It panics if c is out of range.
func SampleChannel(src *cv.Mat, x, y float64, c int, interp Interpolation, border BorderMode, fill float64) float64 {
	if c < 0 || c >= src.Channels {
		panic("transforms2: SampleChannel channel out of range")
	}
	switch interp {
	case InterpNearest:
		return transforms2fetch(src, int(math.Round(x)), int(math.Round(y)), c, border, fill)
	case InterpBilinear:
		x0 := int(math.Floor(x))
		y0 := int(math.Floor(y))
		dx := x - float64(x0)
		dy := y - float64(y0)
		v00 := transforms2fetch(src, x0, y0, c, border, fill)
		v01 := transforms2fetch(src, x0+1, y0, c, border, fill)
		v10 := transforms2fetch(src, x0, y0+1, c, border, fill)
		v11 := transforms2fetch(src, x0+1, y0+1, c, border, fill)
		top := v00*(1-dx) + v01*dx
		bot := v10*(1-dx) + v11*dx
		return top*(1-dy) + bot*dy
	case InterpBicubic:
		x0 := int(math.Floor(x))
		y0 := int(math.Floor(y))
		dx := x - float64(x0)
		dy := y - float64(y0)
		var wx, wy [4]float64
		for k := 0; k < 4; k++ {
			wx[k] = transforms2cubic(dx - float64(k-1))
			wy[k] = transforms2cubic(dy - float64(k-1))
		}
		var acc float64
		for j := 0; j < 4; j++ {
			var row float64
			for i := 0; i < 4; i++ {
				row += wx[i] * transforms2fetch(src, x0+i-1, y0+j-1, c, border, fill)
			}
			acc += wy[j] * row
		}
		return acc
	default:
		panic("transforms2: unknown interpolation")
	}
}

// SamplePixel returns every channel of src sampled at the fractional coordinate
// (x, y) as freshly allocated, clamped 8-bit values.
func SamplePixel(src *cv.Mat, x, y float64, interp Interpolation, border BorderMode, fill float64) []uint8 {
	out := make([]uint8, src.Channels)
	for c := 0; c < src.Channels; c++ {
		out[c] = transforms2clampByte(SampleChannel(src, x, y, c, interp, border, fill))
	}
	return out
}

// transforms2clampByte rounds v and clamps it to the 8-bit range.
func transforms2clampByte(v float64) uint8 {
	v = math.Round(v)
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}

// transforms2warpInverse fills a new width x height Mat by inverse-mapping each
// destination pixel through inv (which returns the source coordinate) and
// sampling src.
func transforms2warpInverse(src *cv.Mat, width, height int, interp Interpolation, border BorderMode, fill float64, inv func(x, y float64) (float64, float64)) *cv.Mat {
	if width <= 0 || height <= 0 {
		panic("transforms2: warp requires positive width and height")
	}
	dst := cv.NewMat(height, width, src.Channels)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			sx, sy := inv(float64(x), float64(y))
			di := (y*width + x) * src.Channels
			for c := 0; c < src.Channels; c++ {
				dst.Data[di+c] = transforms2clampByte(SampleChannel(src, sx, sy, c, interp, border, fill))
			}
		}
	}
	return dst
}
