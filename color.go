package cv

import (
	"fmt"
	"math"
)

// ColorConversionCode selects a colour-space conversion for [CvtColor]. The
// constants mirror the equivalent cv2.COLOR_* codes.
type ColorConversionCode int

const (
	// ColorRGB2Gray converts three-channel RGB to single-channel grayscale
	// using the ITU-R BT.601 luma weights (0.299R + 0.587G + 0.114B).
	ColorRGB2Gray ColorConversionCode = iota
	// ColorGray2RGB replicates the single gray channel into three RGB channels.
	ColorGray2RGB
	// ColorRGB2BGR swaps the red and blue channels (and vice versa).
	ColorRGB2BGR
	// ColorBGR2RGB is identical to ColorRGB2BGR; swapping is its own inverse.
	ColorBGR2RGB
	// ColorBGR2Gray converts three-channel BGR to grayscale.
	ColorBGR2Gray
	// ColorRGB2HSV converts RGB to HSV. Hue is in [0,179] (degrees/2, matching
	// OpenCV's 8-bit convention), saturation and value are in [0,255].
	ColorRGB2HSV
	// ColorHSV2RGB converts HSV (see ColorRGB2HSV for ranges) back to RGB.
	ColorHSV2RGB
)

// CvtColor converts src between colour spaces according to code and returns a
// new Mat. It panics if the source channel count does not match what the code
// expects.
func CvtColor(src *Mat, code ColorConversionCode) *Mat {
	switch code {
	case ColorRGB2Gray, ColorBGR2Gray:
		return toGray(src, code == ColorBGR2Gray)
	case ColorGray2RGB:
		return grayToRGB(src)
	case ColorRGB2BGR, ColorBGR2RGB:
		return swapRB(src)
	case ColorRGB2HSV:
		return rgbToHSV(src)
	case ColorHSV2RGB:
		return hsvToRGB(src)
	default:
		panic(fmt.Sprintf("cv: CvtColor unknown code %d", code))
	}
}

func requireChannels(src *Mat, want int, name string) {
	if src.Channels != want {
		panic(fmt.Sprintf("cv: %s requires %d channels, got %d", name, want, src.Channels))
	}
}

func toGray(src *Mat, bgr bool) *Mat {
	requireChannels(src, 3, "CvtColor RGB/BGR->Gray")
	dst := NewMat(src.Rows, src.Cols, 1)
	ri, bi := 0, 2
	if bgr {
		ri, bi = 2, 0
	}
	for p := 0; p < src.Total(); p++ {
		base := p * 3
		r := float64(src.Data[base+ri])
		g := float64(src.Data[base+1])
		b := float64(src.Data[base+bi])
		y := 0.299*r + 0.587*g + 0.114*b
		dst.Data[p] = clampToUint8(y + 0.5)
	}
	return dst
}

func grayToRGB(src *Mat) *Mat {
	requireChannels(src, 1, "CvtColor Gray->RGB")
	dst := NewMat(src.Rows, src.Cols, 3)
	for p := 0; p < src.Total(); p++ {
		v := src.Data[p]
		dst.Data[p*3+0] = v
		dst.Data[p*3+1] = v
		dst.Data[p*3+2] = v
	}
	return dst
}

func swapRB(src *Mat) *Mat {
	requireChannels(src, 3, "CvtColor RGB<->BGR")
	dst := NewMat(src.Rows, src.Cols, 3)
	for p := 0; p < src.Total(); p++ {
		base := p * 3
		dst.Data[base+0] = src.Data[base+2]
		dst.Data[base+1] = src.Data[base+1]
		dst.Data[base+2] = src.Data[base+0]
	}
	return dst
}

func rgbToHSV(src *Mat) *Mat {
	requireChannels(src, 3, "CvtColor RGB->HSV")
	dst := NewMat(src.Rows, src.Cols, 3)
	for p := 0; p < src.Total(); p++ {
		base := p * 3
		r := float64(src.Data[base+0]) / 255.0
		g := float64(src.Data[base+1]) / 255.0
		b := float64(src.Data[base+2]) / 255.0
		max := math.Max(r, math.Max(g, b))
		min := math.Min(r, math.Min(g, b))
		delta := max - min

		var h float64
		switch {
		case delta == 0:
			h = 0
		case max == r:
			h = 60 * math.Mod((g-b)/delta, 6)
		case max == g:
			h = 60 * ((b-r)/delta + 2)
		default:
			h = 60 * ((r-g)/delta + 4)
		}
		if h < 0 {
			h += 360
		}
		var s float64
		if max > 0 {
			s = delta / max
		}
		v := max

		// OpenCV 8-bit HSV: H in [0,179], S and V in [0,255].
		dst.Data[base+0] = uint8(math.Round(h / 2))
		dst.Data[base+1] = clampToUint8(s*255 + 0.5)
		dst.Data[base+2] = clampToUint8(v*255 + 0.5)
	}
	return dst
}

func hsvToRGB(src *Mat) *Mat {
	requireChannels(src, 3, "CvtColor HSV->RGB")
	dst := NewMat(src.Rows, src.Cols, 3)
	for p := 0; p < src.Total(); p++ {
		base := p * 3
		h := float64(src.Data[base+0]) * 2 // back to [0,360)
		s := float64(src.Data[base+1]) / 255.0
		v := float64(src.Data[base+2]) / 255.0

		c := v * s
		x := c * (1 - math.Abs(math.Mod(h/60, 2)-1))
		mm := v - c
		var r, g, b float64
		switch {
		case h < 60:
			r, g, b = c, x, 0
		case h < 120:
			r, g, b = x, c, 0
		case h < 180:
			r, g, b = 0, c, x
		case h < 240:
			r, g, b = 0, x, c
		case h < 300:
			r, g, b = x, 0, c
		default:
			r, g, b = c, 0, x
		}
		dst.Data[base+0] = clampToUint8((r+mm)*255 + 0.5)
		dst.Data[base+1] = clampToUint8((g+mm)*255 + 0.5)
		dst.Data[base+2] = clampToUint8((b+mm)*255 + 0.5)
	}
	return dst
}

// clampToUint8 rounds toward zero after the caller has added any rounding bias
// and clamps the result into [0,255].
func clampToUint8(v float64) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 255 {
		return 255
	}
	return uint8(v)
}
