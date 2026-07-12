package gapi

import (
	cv "github.com/malcolmston/opencv"
)

// Additional operation names for the imgproc-style operations that do not have a
// direct core-op counterpart.
const (
	OpRGB2Gray = "rgb2gray"
	OpBGR2Gray = "bgr2gray"
)

// RGB2Gray returns a lazy node converting a three-channel RGB image to
// single-channel grayscale.
func RGB2Gray(src GMat) GMat {
	return newOp(OpRGB2Gray, []GMat{src}, nil, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.CvtColor(ctx.Mats[0], cv.ColorRGB2Gray)
	})
}

// BGR2Gray returns a lazy node converting a three-channel BGR image to
// single-channel grayscale.
func BGR2Gray(src GMat) GMat {
	return newOp(OpBGR2Gray, []GMat{src}, nil, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.CvtColor(ctx.Mats[0], cv.ColorBGR2Gray)
	})
}

// CvtColor returns a lazy node converting src between colour spaces per code.
func CvtColor(src GMat, code ColorConversionCode) GMat {
	return newOp(OpCvtColor, []GMat{src}, nil, []int{int(code)}, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.CvtColor(ctx.Mats[0], ColorConversionCode(ctx.Ints[0]))
	})
}

// Blur returns a lazy node smoothing src with a normalised ksize×ksize box
// filter. ksize must be a positive odd integer.
func Blur(src GMat, ksize int) GMat {
	return newOp(OpBlur, []GMat{src}, nil, []int{ksize}, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.Blur(ctx.Mats[0], ctx.Ints[0])
	})
}

// GaussianBlur returns a lazy node convolving src with a separable Gaussian of
// size ksize×ksize and standard deviation sigma (sigma <= 0 derives it from
// ksize). ksize must be a positive odd integer.
func GaussianBlur(src GMat, ksize int, sigma float64) GMat {
	return newOp(OpGaussianBlur, []GMat{src}, nil, []int{ksize}, []float64{sigma}, nil, func(ctx KernelContext) *cv.Mat {
		return cv.GaussianBlur(ctx.Mats[0], ctx.Ints[0], ctx.Floats[0])
	})
}

// MedianBlur returns a lazy node replacing each sample with the median of its
// ksize×ksize neighbourhood. ksize must be a positive odd integer.
func MedianBlur(src GMat, ksize int) GMat {
	return newOp(OpMedianBlur, []GMat{src}, nil, []int{ksize}, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.MedianBlur(ctx.Mats[0], ctx.Ints[0])
	})
}

// Sobel returns a lazy node computing the Sobel derivative of src. dx and dy are
// derivative orders and ksize the aperture (1 or 3); the result is scaled by
// scale, offset by delta and clamped to [0,255].
func Sobel(src GMat, dx, dy, ksize int, scale, delta float64) GMat {
	return newOp(OpSobel, []GMat{src}, nil, []int{dx, dy, ksize}, []float64{scale, delta}, nil, func(ctx KernelContext) *cv.Mat {
		return cv.Sobel(ctx.Mats[0], ctx.Ints[0], ctx.Ints[1], ctx.Ints[2], ctx.Floats[0], ctx.Floats[1])
	})
}

// Laplacian returns a lazy node computing the Laplacian of src with the given
// aperture (1 or 3), scaled by scale, offset by delta and clamped to [0,255].
func Laplacian(src GMat, ksize int, scale, delta float64) GMat {
	return newOp(OpLaplacian, []GMat{src}, nil, []int{ksize}, []float64{scale, delta}, nil, func(ctx KernelContext) *cv.Mat {
		return cv.Laplacian(ctx.Mats[0], ctx.Ints[0], ctx.Floats[0], ctx.Floats[1])
	})
}

// Canny returns a lazy node running the Canny edge detector on a single-channel
// image with the given hysteresis thresholds.
func Canny(src GMat, lowThresh, highThresh float64) GMat {
	return newOp(OpCanny, []GMat{src}, nil, nil, []float64{lowThresh, highThresh}, nil, func(ctx KernelContext) *cv.Mat {
		return cv.Canny(ctx.Mats[0], ctx.Floats[0], ctx.Floats[1])
	})
}

// Dilate returns a lazy node growing bright regions of src with a structuring
// element of the given shape and size (ksize×ksize), repeated iterations times.
func Dilate(src GMat, shape MorphShape, ksize, iterations int) GMat {
	return newOp(OpDilate, []GMat{src}, nil, []int{int(shape), ksize, iterations}, nil, nil, func(ctx KernelContext) *cv.Mat {
		k := cv.GetStructuringElement(MorphShape(ctx.Ints[0]), ctx.Ints[1], ctx.Ints[1])
		return cv.Dilate(ctx.Mats[0], k, ctx.Ints[2])
	})
}

// Erode returns a lazy node shrinking bright regions of src with a structuring
// element of the given shape and size (ksize×ksize), repeated iterations times.
func Erode(src GMat, shape MorphShape, ksize, iterations int) GMat {
	return newOp(OpErode, []GMat{src}, nil, []int{int(shape), ksize, iterations}, nil, nil, func(ctx KernelContext) *cv.Mat {
		k := cv.GetStructuringElement(MorphShape(ctx.Ints[0]), ctx.Ints[1], ctx.Ints[1])
		return cv.Erode(ctx.Mats[0], k, ctx.Ints[2])
	})
}

// EqualizeHist returns a lazy node performing global histogram equalisation on a
// single-channel image.
func EqualizeHist(src GMat) GMat {
	return newOp(OpEqualizeHist, []GMat{src}, nil, nil, nil, nil, func(ctx KernelContext) *cv.Mat {
		return cv.EqualizeHist(ctx.Mats[0])
	})
}
