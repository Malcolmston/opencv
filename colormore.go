package cv

import "math"

// requireRGB panics unless m has three channels.
func requireRGB(m *Mat, fn string) {
	if m.Channels != 3 {
		panic("cv: " + fn + " requires a 3-channel RGB image")
	}
}

// RGBToXYZ converts an 8-bit RGB image to CIE 1931 XYZ using OpenCV's linear
// transform matrix, returning a 3-channel image whose channels are X, Y and Z
// clamped to [0,255]. This mirrors cv2.cvtColor with COLOR_RGB2XYZ.
func RGBToXYZ(src *Mat) *Mat {
	requireRGB(src, "RGBToXYZ")
	dst := NewMat(src.Rows, src.Cols, 3)
	n := src.Rows * src.Cols
	for i := 0; i < n; i++ {
		b := i * 3
		r := float64(src.Data[b])
		g := float64(src.Data[b+1])
		bl := float64(src.Data[b+2])
		dst.Data[b] = clampToUint8(0.412453*r + 0.357580*g + 0.180423*bl + 0.5)
		dst.Data[b+1] = clampToUint8(0.212671*r + 0.715160*g + 0.072169*bl + 0.5)
		dst.Data[b+2] = clampToUint8(0.019334*r + 0.119193*g + 0.950227*bl + 0.5)
	}
	return dst
}

// XYZToRGB is the inverse of [RGBToXYZ], converting a 3-channel XYZ image back
// to 8-bit RGB. It mirrors cv2.cvtColor with COLOR_XYZ2RGB.
func XYZToRGB(src *Mat) *Mat {
	requireRGB(src, "XYZToRGB")
	dst := NewMat(src.Rows, src.Cols, 3)
	n := src.Rows * src.Cols
	for i := 0; i < n; i++ {
		b := i * 3
		x := float64(src.Data[b])
		y := float64(src.Data[b+1])
		z := float64(src.Data[b+2])
		dst.Data[b] = clampToUint8(3.240479*x - 1.537150*y - 0.498535*z + 0.5)
		dst.Data[b+1] = clampToUint8(-0.969256*x + 1.875991*y + 0.041556*z + 0.5)
		dst.Data[b+2] = clampToUint8(0.055648*x - 0.204043*y + 1.057311*z + 0.5)
	}
	return dst
}

// RGBToYUV converts an 8-bit RGB image to analogue BT.601 Y'UV, with U and V
// offset by 128 so they fit in [0,255]. This mirrors cv2.cvtColor with
// COLOR_RGB2YUV.
func RGBToYUV(src *Mat) *Mat {
	requireRGB(src, "RGBToYUV")
	dst := NewMat(src.Rows, src.Cols, 3)
	n := src.Rows * src.Cols
	for i := 0; i < n; i++ {
		b := i * 3
		r := float64(src.Data[b])
		g := float64(src.Data[b+1])
		bl := float64(src.Data[b+2])
		yv := 0.299*r + 0.587*g + 0.114*bl
		u := -0.14713*r - 0.28886*g + 0.436*bl + 128
		v := 0.615*r - 0.51499*g - 0.10001*bl + 128
		dst.Data[b] = clampToUint8(yv + 0.5)
		dst.Data[b+1] = clampToUint8(u + 0.5)
		dst.Data[b+2] = clampToUint8(v + 0.5)
	}
	return dst
}

// YUVToRGB is the inverse of [RGBToYUV], converting a 3-channel Y'UV image back
// to 8-bit RGB. It mirrors cv2.cvtColor with COLOR_YUV2RGB.
func YUVToRGB(src *Mat) *Mat {
	requireRGB(src, "YUVToRGB")
	dst := NewMat(src.Rows, src.Cols, 3)
	n := src.Rows * src.Cols
	for i := 0; i < n; i++ {
		b := i * 3
		yv := float64(src.Data[b])
		u := float64(src.Data[b+1]) - 128
		v := float64(src.Data[b+2]) - 128
		dst.Data[b] = clampToUint8(yv + 1.13983*v + 0.5)
		dst.Data[b+1] = clampToUint8(yv - 0.39465*u - 0.58060*v + 0.5)
		dst.Data[b+2] = clampToUint8(yv + 2.03211*u + 0.5)
	}
	return dst
}

// RGBToCMYK converts an 8-bit RGB image to CMYK and returns a 4-channel image
// whose channels are C, M, Y and K, each scaled to [0,255].
func RGBToCMYK(src *Mat) *Mat {
	requireRGB(src, "RGBToCMYK")
	dst := NewMat(src.Rows, src.Cols, 4)
	n := src.Rows * src.Cols
	for i := 0; i < n; i++ {
		s := i * 3
		d := i * 4
		r := float64(src.Data[s]) / 255
		g := float64(src.Data[s+1]) / 255
		bl := float64(src.Data[s+2]) / 255
		mx := math.Max(r, math.Max(g, bl))
		k := 1 - mx
		if k >= 1 {
			dst.Data[d] = 0
			dst.Data[d+1] = 0
			dst.Data[d+2] = 0
			dst.Data[d+3] = 255
			continue
		}
		inv := 1 / (1 - k)
		dst.Data[d] = clampToUint8((1-r-k)*inv*255 + 0.5)
		dst.Data[d+1] = clampToUint8((1-g-k)*inv*255 + 0.5)
		dst.Data[d+2] = clampToUint8((1-bl-k)*inv*255 + 0.5)
		dst.Data[d+3] = clampToUint8(k*255 + 0.5)
	}
	return dst
}

// CMYKToRGB is the inverse of [RGBToCMYK], converting a 4-channel CMYK image to
// 8-bit RGB. It panics unless src has four channels.
func CMYKToRGB(src *Mat) *Mat {
	if src.Channels != 4 {
		panic("cv: CMYKToRGB requires a 4-channel CMYK image")
	}
	dst := NewMat(src.Rows, src.Cols, 3)
	n := src.Rows * src.Cols
	for i := 0; i < n; i++ {
		s := i * 4
		d := i * 3
		c := float64(src.Data[s]) / 255
		m := float64(src.Data[s+1]) / 255
		y := float64(src.Data[s+2]) / 255
		k := float64(src.Data[s+3]) / 255
		dst.Data[d] = clampToUint8((1-c)*(1-k)*255 + 0.5)
		dst.Data[d+1] = clampToUint8((1-m)*(1-k)*255 + 0.5)
		dst.Data[d+2] = clampToUint8((1-y)*(1-k)*255 + 0.5)
	}
	return dst
}

// RGBToGray601 converts an 8-bit RGB image to single-channel grayscale using
// the ITU-R BT.601 luma weights (0.299, 0.587, 0.114), matching OpenCV's
// default COLOR_RGB2GRAY.
func RGBToGray601(src *Mat) *Mat {
	requireRGB(src, "RGBToGray601")
	dst := NewMat(src.Rows, src.Cols, 1)
	n := src.Rows * src.Cols
	for i := 0; i < n; i++ {
		b := i * 3
		dst.Data[i] = clampToUint8(0.299*float64(src.Data[b]) + 0.587*float64(src.Data[b+1]) + 0.114*float64(src.Data[b+2]) + 0.5)
	}
	return dst
}

// RGBToGray709 converts an 8-bit RGB image to single-channel grayscale using
// the ITU-R BT.709 luma weights (0.2126, 0.7152, 0.0722).
func RGBToGray709(src *Mat) *Mat {
	requireRGB(src, "RGBToGray709")
	dst := NewMat(src.Rows, src.Cols, 1)
	n := src.Rows * src.Cols
	for i := 0; i < n; i++ {
		b := i * 3
		dst.Data[i] = clampToUint8(0.2126*float64(src.Data[b]) + 0.7152*float64(src.Data[b+1]) + 0.0722*float64(src.Data[b+2]) + 0.5)
	}
	return dst
}
