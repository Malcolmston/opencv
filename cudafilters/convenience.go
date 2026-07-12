package cudafilters

import (
	"image"

	cv "github.com/malcolmston/opencv"
)

// This file provides one-call convenience wrappers for the case where a filter
// is used exactly once. Each builds the corresponding [Filter] with the default
// centred anchor and border, applies it to src and returns the result. The
// optional stream is accepted for API compatibility and ignored.

// BoxFilter smooths src with a normalised box of the given square odd ksize.
// It is [CreateBoxFilter] applied once.
func BoxFilter(src *GpuMat, ksize image.Point, stream ...*Stream) *GpuMat {
	return CreateBoxFilter(ksize, AnchorCenter, BorderDefault).Apply(src, stream...)
}

// Blur smooths src with a normalised box of the given square odd ksize. It is a
// synonym for [BoxFilter].
func Blur(src *GpuMat, ksize image.Point, stream ...*Stream) *GpuMat {
	return CreateBlurFilter(ksize).Apply(src, stream...)
}

// Filter2D convolves src with an arbitrary kernel plus delta. It is
// [CreateLinearFilter] applied once.
func Filter2D(src *GpuMat, kernel cv.Kernel, delta float64, stream ...*Stream) *GpuMat {
	return CreateLinearFilter(kernel, delta, AnchorCenter, BorderDefault).Apply(src, stream...)
}

// SepFilter2D applies a separable linear filter (horizontal rowKernel then
// vertical columnKernel) plus delta. It is [CreateSeparableLinearFilter] applied
// once.
func SepFilter2D(src *GpuMat, rowKernel, columnKernel []float64, delta float64, stream ...*Stream) *GpuMat {
	return CreateSeparableLinearFilter(rowKernel, columnKernel, delta, AnchorCenter, BorderDefault).Apply(src, stream...)
}

// GaussianBlur blurs src with a separable Gaussian of the given ksize and
// standard deviations. It is [CreateGaussianFilter] applied once.
func GaussianBlur(src *GpuMat, ksize image.Point, sigma1, sigma2 float64, stream ...*Stream) *GpuMat {
	return CreateGaussianFilter(ksize, sigma1, sigma2, BorderDefault).Apply(src, stream...)
}

// Sobel computes the Sobel derivative of orders (dx, dy) with aperture ksize. It
// is [CreateSobelFilter] applied once.
func Sobel(src *GpuMat, dx, dy, ksize int, scale, delta float64, stream ...*Stream) *GpuMat {
	return CreateSobelFilter(dx, dy, ksize, scale, delta, BorderDefault).Apply(src, stream...)
}

// Scharr computes the 3×3 Scharr derivative of orders (dx, dy). It is
// [CreateScharrFilter] applied once.
func Scharr(src *GpuMat, dx, dy int, scale, delta float64, stream ...*Stream) *GpuMat {
	return CreateScharrFilter(dx, dy, scale, delta, BorderDefault).Apply(src, stream...)
}

// Laplacian computes the Laplacian with aperture ksize. It is
// [CreateLaplacianFilter] applied once.
func Laplacian(src *GpuMat, ksize int, scale, delta float64, stream ...*Stream) *GpuMat {
	return CreateLaplacianFilter(ksize, scale, delta, BorderDefault).Apply(src, stream...)
}

// MedianBlur replaces each sample with the median of its windowSize×windowSize
// neighbourhood. It is [CreateMedianFilter] applied once.
func MedianBlur(src *GpuMat, windowSize int, stream ...*Stream) *GpuMat {
	return CreateMedianFilter(windowSize).Apply(src, stream...)
}

// BoxMax replaces each sample with the maximum over a ksize rectangle. It is
// [CreateBoxMaxFilter] applied once.
func BoxMax(src *GpuMat, ksize image.Point, stream ...*Stream) *GpuMat {
	return CreateBoxMaxFilter(ksize, AnchorCenter, BorderDefault).Apply(src, stream...)
}

// BoxMin replaces each sample with the minimum over a ksize rectangle. It is
// [CreateBoxMinFilter] applied once.
func BoxMin(src *GpuMat, ksize image.Point, stream ...*Stream) *GpuMat {
	return CreateBoxMinFilter(ksize, AnchorCenter, BorderDefault).Apply(src, stream...)
}

// Erode applies a morphological erosion with the given structuring element. It
// is [CreateErodeFilter] applied once.
func Erode(src *GpuMat, kernel *cv.Mat, iterations int, stream ...*Stream) *GpuMat {
	return CreateErodeFilter(kernel, AnchorCenter, iterations).Apply(src, stream...)
}

// Dilate applies a morphological dilation with the given structuring element. It
// is [CreateDilateFilter] applied once.
func Dilate(src *GpuMat, kernel *cv.Mat, iterations int, stream ...*Stream) *GpuMat {
	return CreateDilateFilter(kernel, AnchorCenter, iterations).Apply(src, stream...)
}

// MorphologyEx applies the compound morphological operation op with the given
// structuring element. It is [CreateMorphologyFilter] applied once.
func MorphologyEx(src *GpuMat, op MorphOp, kernel *cv.Mat, iterations int, stream ...*Stream) *GpuMat {
	return CreateMorphologyFilter(op, kernel, AnchorCenter, iterations).Apply(src, stream...)
}

// RowSum computes the running horizontal sum over ksize neighbours. It is
// [CreateRowSumFilter] applied once. Sums saturate at 255 (see
// [CreateRowSumFilter]).
func RowSum(src *GpuMat, ksize int, stream ...*Stream) *GpuMat {
	return CreateRowSumFilter(ksize).Apply(src, stream...)
}

// ColumnSum computes the running vertical sum over ksize neighbours. It is
// [CreateColumnSumFilter] applied once. Sums saturate at 255 (see
// [CreateColumnSumFilter]).
func ColumnSum(src *GpuMat, ksize int, stream ...*Stream) *GpuMat {
	return CreateColumnSumFilter(ksize).Apply(src, stream...)
}
