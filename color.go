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
	// ColorRGB2Lab converts RGB to CIE L*a*b* (D65 white point). In 8-bit form
	// L is scaled to [0,255] and a,b are stored with a +128 offset, matching
	// OpenCV's convention.
	ColorRGB2Lab
	// ColorLab2RGB converts CIE L*a*b* (see ColorRGB2Lab for ranges) back to RGB.
	ColorLab2RGB
	// ColorRGB2YCrCb converts RGB to Y'CrCb. Y is luma; Cr and Cb are the
	// red/blue chroma differences stored with a +128 offset.
	ColorRGB2YCrCb
	// ColorYCrCb2RGB converts Y'CrCb (see ColorRGB2YCrCb) back to RGB.
	ColorYCrCb2RGB
	// ColorRGB2HLS converts RGB to HLS. Hue is in [0,179] (degrees/2), while
	// lightness and saturation are in [0,255].
	ColorRGB2HLS
	// ColorHLS2RGB converts HLS (see ColorRGB2HLS for ranges) back to RGB.
	ColorHLS2RGB
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
	case ColorRGB2Lab:
		return rgbToLab(src)
	case ColorLab2RGB:
		return labToRGB(src)
	case ColorRGB2YCrCb:
		return rgbToYCrCb(src)
	case ColorYCrCb2RGB:
		return yCrCbToRGB(src)
	case ColorRGB2HLS:
		return rgbToHLS(src)
	case ColorHLS2RGB:
		return hlsToRGB(src)
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

// sRGB gamma helpers and D65 white point shared by the Lab conversions.
const (
	labXn = 0.950456
	labYn = 1.0
	labZn = 1.088754
)

func srgbToLinear(c float64) float64 {
	if c <= 0.04045 {
		return c / 12.92
	}
	return math.Pow((c+0.055)/1.055, 2.4)
}

func linearToSrgb(c float64) float64 {
	if c <= 0.0031308 {
		return 12.92 * c
	}
	return 1.055*math.Pow(c, 1/2.4) - 0.055
}

func labF(t float64) float64 {
	if t > 0.008856 {
		return math.Cbrt(t)
	}
	return 7.787*t + 16.0/116.0
}

func labFInv(t float64) float64 {
	t3 := t * t * t
	if t3 > 0.008856 {
		return t3
	}
	return (t - 16.0/116.0) / 7.787
}

func rgbToLab(src *Mat) *Mat {
	requireChannels(src, 3, "CvtColor RGB->Lab")
	dst := NewMat(src.Rows, src.Cols, 3)
	for p := 0; p < src.Total(); p++ {
		base := p * 3
		r := srgbToLinear(float64(src.Data[base+0]) / 255)
		g := srgbToLinear(float64(src.Data[base+1]) / 255)
		b := srgbToLinear(float64(src.Data[base+2]) / 255)
		x := (0.412453*r + 0.357580*g + 0.180423*b) / labXn
		y := (0.212671*r + 0.715160*g + 0.072169*b) / labYn
		z := (0.019334*r + 0.119193*g + 0.950227*b) / labZn
		fx, fy, fz := labF(x), labF(y), labF(z)
		l := 116*fy - 16
		a := 500 * (fx - fy)
		bb := 200 * (fy - fz)
		dst.Data[base+0] = clampToUint8(l*255/100 + 0.5)
		dst.Data[base+1] = clampToUint8(a + 128 + 0.5)
		dst.Data[base+2] = clampToUint8(bb + 128 + 0.5)
	}
	return dst
}

func labToRGB(src *Mat) *Mat {
	requireChannels(src, 3, "CvtColor Lab->RGB")
	dst := NewMat(src.Rows, src.Cols, 3)
	for p := 0; p < src.Total(); p++ {
		base := p * 3
		l := float64(src.Data[base+0]) * 100 / 255
		a := float64(src.Data[base+1]) - 128
		bb := float64(src.Data[base+2]) - 128
		fy := (l + 16) / 116
		fx := fy + a/500
		fz := fy - bb/200
		x := labXn * labFInv(fx)
		y := labYn * labFInv(fy)
		z := labZn * labFInv(fz)
		r := 3.240479*x - 1.537150*y - 0.498535*z
		g := -0.969256*x + 1.875992*y + 0.041556*z
		b := 0.055648*x - 0.204043*y + 1.057311*z
		dst.Data[base+0] = clampToUint8(linearToSrgb(r)*255 + 0.5)
		dst.Data[base+1] = clampToUint8(linearToSrgb(g)*255 + 0.5)
		dst.Data[base+2] = clampToUint8(linearToSrgb(b)*255 + 0.5)
	}
	return dst
}

func rgbToYCrCb(src *Mat) *Mat {
	requireChannels(src, 3, "CvtColor RGB->YCrCb")
	dst := NewMat(src.Rows, src.Cols, 3)
	for p := 0; p < src.Total(); p++ {
		base := p * 3
		r := float64(src.Data[base+0])
		g := float64(src.Data[base+1])
		b := float64(src.Data[base+2])
		y := 0.299*r + 0.587*g + 0.114*b
		cr := (r-y)*0.713 + 128
		cb := (b-y)*0.564 + 128
		dst.Data[base+0] = clampToUint8(y + 0.5)
		dst.Data[base+1] = clampToUint8(cr + 0.5)
		dst.Data[base+2] = clampToUint8(cb + 0.5)
	}
	return dst
}

func yCrCbToRGB(src *Mat) *Mat {
	requireChannels(src, 3, "CvtColor YCrCb->RGB")
	dst := NewMat(src.Rows, src.Cols, 3)
	for p := 0; p < src.Total(); p++ {
		base := p * 3
		y := float64(src.Data[base+0])
		cr := float64(src.Data[base+1]) - 128
		cb := float64(src.Data[base+2]) - 128
		r := y + 1.403*cr
		g := y - 0.714*cr - 0.344*cb
		b := y + 1.773*cb
		dst.Data[base+0] = clampToUint8(r + 0.5)
		dst.Data[base+1] = clampToUint8(g + 0.5)
		dst.Data[base+2] = clampToUint8(b + 0.5)
	}
	return dst
}

func rgbToHLS(src *Mat) *Mat {
	requireChannels(src, 3, "CvtColor RGB->HLS")
	dst := NewMat(src.Rows, src.Cols, 3)
	for p := 0; p < src.Total(); p++ {
		base := p * 3
		r := float64(src.Data[base+0]) / 255
		g := float64(src.Data[base+1]) / 255
		b := float64(src.Data[base+2]) / 255
		max := math.Max(r, math.Max(g, b))
		min := math.Min(r, math.Min(g, b))
		delta := max - min
		l := (max + min) / 2
		var h, s float64
		if delta != 0 {
			if l < 0.5 {
				s = delta / (max + min)
			} else {
				s = delta / (2 - max - min)
			}
			switch {
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
		}
		dst.Data[base+0] = uint8(math.Round(h / 2))
		dst.Data[base+1] = clampToUint8(l*255 + 0.5)
		dst.Data[base+2] = clampToUint8(s*255 + 0.5)
	}
	return dst
}

func hlsToRGB(src *Mat) *Mat {
	requireChannels(src, 3, "CvtColor HLS->RGB")
	dst := NewMat(src.Rows, src.Cols, 3)
	for p := 0; p < src.Total(); p++ {
		base := p * 3
		h := float64(src.Data[base+0]) * 2
		l := float64(src.Data[base+1]) / 255
		s := float64(src.Data[base+2]) / 255
		var c float64
		if l < 0.5 {
			c = 2 * l * s
		} else {
			c = (2 - 2*l) * s
		}
		x := c * (1 - math.Abs(math.Mod(h/60, 2)-1))
		m := l - c/2
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
		dst.Data[base+0] = clampToUint8((r+m)*255 + 0.5)
		dst.Data[base+1] = clampToUint8((g+m)*255 + 0.5)
		dst.Data[base+2] = clampToUint8((b+m)*255 + 0.5)
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
