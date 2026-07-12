package cudafilters

import (
	"fmt"
	"image"

	cv "github.com/malcolmston/opencv"
)

// Filter is the object interface OpenCV's cudafilters exposes: a reusable filter
// created by one of the Create* factories and applied to a [GpuMat] with Apply.
// The same Filter may be applied to many inputs.
type Filter interface {
	// Apply filters src and returns the result in a new [GpuMat]. Any [Stream]
	// arguments are accepted for API compatibility and ignored, because this
	// package computes synchronously on the CPU. Apply panics if src is empty.
	Apply(src *GpuMat, stream ...*Stream) *GpuMat
}

// filterFunc adapts a plain cv.Mat→cv.Mat transform into a [Filter]. The heavy
// lifting always lives in the root package; a factory just captures its
// parameters in one of these closures.
type filterFunc func(src *cv.Mat) *cv.Mat

// Apply implements [Filter].
func (f filterFunc) Apply(src *GpuMat, _ ...*Stream) *GpuMat {
	if src.Empty() {
		panic("cudafilters: Apply called with an empty GpuMat")
	}
	return &GpuMat{mat: f(src.mat)}
}

// BorderType names a pixel-extrapolation method, mirroring OpenCV's border
// constants for API compatibility.
//
// Honesty note: the root-package filter primitives always extrapolate by edge
// replication (BORDER_REPLICATE). A BorderType is therefore accepted by the
// factories but is advisory only — every value is treated as replication. The
// constant is retained so that call sites written against OpenCV compile
// unchanged.
type BorderType int

// Border extrapolation modes. All are accepted; all behave as
// [BorderReplicate] in this CPU backend (see [BorderType]).
const (
	BorderConstant   BorderType = iota // iiiiii|abcdefgh|iiiiiii with a constant i
	BorderReplicate                    // aaaaaa|abcdefgh|hhhhhhh (the mode actually used)
	BorderReflect                      // fedcba|abcdefgh|hgfedcb
	BorderWrap                         // cdefgh|abcdefgh|abcdefg
	BorderReflect101                   // gfedcb|abcdefgh|gfedcba
	// BorderDefault is OpenCV's default (BORDER_REFLECT_101).
	BorderDefault = BorderReflect101
)

// MorphOp selects the operation performed by [CreateMorphologyFilter], mirroring
// OpenCV's MorphTypes. Each value maps directly onto the root package's
// [cv.MorphType].
type MorphOp int

// Morphological operations understood by [CreateMorphologyFilter].
const (
	MorphErode    MorphOp = iota // erosion (local minimum)
	MorphDilate                  // dilation (local maximum)
	MorphOpen                    // erosion then dilation
	MorphClose                   // dilation then erosion
	MorphGradient                // dilation minus erosion
	MorphTophat                  // source minus its opening
	MorphBlackhat                // closing minus the source
)

// toCVMorphType translates a [MorphOp] into the root package's [cv.MorphType].
func (op MorphOp) toCVMorphType() cv.MorphType {
	switch op {
	case MorphErode:
		return cv.MorphErode
	case MorphDilate:
		return cv.MorphDilate
	case MorphOpen:
		return cv.MorphOpen
	case MorphClose:
		return cv.MorphClose
	case MorphGradient:
		return cv.MorphGradient
	case MorphTophat:
		return cv.MorphTophat
	case MorphBlackhat:
		return cv.MorphBlackhat
	default:
		panic(fmt.Sprintf("cudafilters: unknown MorphOp %d", op))
	}
}

// requireSquareOdd validates that a Size-style ksize is square with a positive
// odd extent, the only form the root scalar-ksize primitives accept. It returns
// that extent.
func requireSquareOdd(ksize image.Point, who string) int {
	if ksize.X != ksize.Y {
		panic(fmt.Sprintf("cudafilters: %s requires a square ksize, got %dx%d", who, ksize.X, ksize.Y))
	}
	k := ksize.X
	if k <= 0 || k%2 == 0 {
		panic(fmt.Sprintf("cudafilters: %s requires a positive odd ksize, got %d", who, k))
	}
	return k
}

// requirePositiveOdd validates a scalar positive odd extent.
func requirePositiveOdd(k int, who string) {
	if k <= 0 || k%2 == 0 {
		panic(fmt.Sprintf("cudafilters: %s requires a positive odd size, got %d", who, k))
	}
}

// requireCenterAnchor rejects a non-centred anchor, which the engine cannot
// honour.
func requireCenterAnchor(anchor image.Point, who string) {
	if !isCenterAnchor(anchor) {
		panic(fmt.Sprintf("cudafilters: %s only supports the centred anchor image.Pt(-1,-1), got %v", who, anchor))
	}
}

// ---------------------------------------------------------------------------
// Linear filtering
// ---------------------------------------------------------------------------

// CreateBoxFilter returns a [Filter] that smooths with a normalised (averaging)
// box of size ksize, mirroring cv::cuda::createBoxFilter. ksize must be square
// and odd; anchor must be [AnchorCenter]. borderType is advisory (see
// [BorderType]). It delegates to [cv.BoxFilter] with normalisation enabled.
func CreateBoxFilter(ksize image.Point, anchor image.Point, borderType BorderType) Filter {
	k := requireSquareOdd(ksize, "CreateBoxFilter")
	requireCenterAnchor(anchor, "CreateBoxFilter")
	_ = borderType
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.BoxFilter(src, k, true)
	})
}

// CreateBlurFilter is an alias for [CreateBoxFilter] with the centred anchor and
// default border, matching OpenCV's blur convenience. It returns a normalised
// box (mean) filter of the given square odd ksize.
func CreateBlurFilter(ksize image.Point) Filter {
	return CreateBoxFilter(ksize, AnchorCenter, BorderDefault)
}

// CreateLinearFilter returns a [Filter] that convolves with an arbitrary 2-D
// kernel, mirroring cv::cuda::createLinearFilter (the GPU analogue of
// cv::filter2D). delta is added to every filtered sample before rounding and
// clamping. anchor must be [AnchorCenter]; borderType is advisory. It delegates
// to [cv.Filter2D].
func CreateLinearFilter(kernel cv.Kernel, delta float64, anchor image.Point, borderType BorderType) Filter {
	requireCenterAnchor(anchor, "CreateLinearFilter")
	_ = borderType
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.Filter2D(src, kernel, delta)
	})
}

// CreateSeparableLinearFilter returns a [Filter] that applies a separable linear
// filter: convolution with the 1-D horizontal kernel rowKernel followed by the
// 1-D vertical kernel columnKernel, mirroring cv::cuda::
// createSeparableLinearFilter. delta is added before rounding and clamping.
// anchor must be [AnchorCenter]; borderType is advisory. It delegates to
// [cv.Filter2DSep].
func CreateSeparableLinearFilter(rowKernel, columnKernel []float64, delta float64, anchor image.Point, borderType BorderType) Filter {
	if len(rowKernel) == 0 || len(columnKernel) == 0 {
		panic("cudafilters: CreateSeparableLinearFilter requires non-empty kernels")
	}
	requireCenterAnchor(anchor, "CreateSeparableLinearFilter")
	_ = borderType
	rk := append([]float64(nil), rowKernel...)
	ck := append([]float64(nil), columnKernel...)
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.Filter2DSep(src, rk, ck, delta)
	})
}

// CreateGaussianFilter returns a [Filter] that blurs with a separable Gaussian
// of size ksize, mirroring cv::cuda::createGaussianFilter. sigma1 is the
// standard deviation along X; sigma2 along Y (when sigma2 <= 0 it defaults to
// sigma1). A sigma <= 0 is derived from the corresponding kernel extent as
// OpenCV does. ksize.X and ksize.Y must each be positive and odd. borderType is
// advisory. It delegates to [cv.GaussianKernel1D] and [cv.Filter2DSep].
func CreateGaussianFilter(ksize image.Point, sigma1, sigma2 float64, borderType BorderType) Filter {
	requirePositiveOdd(ksize.X, "CreateGaussianFilter (ksize.X)")
	requirePositiveOdd(ksize.Y, "CreateGaussianFilter (ksize.Y)")
	if sigma2 <= 0 {
		sigma2 = sigma1
	}
	_ = borderType
	kx := cv.GaussianKernel1D(ksize.X, sigma1)
	ky := cv.GaussianKernel1D(ksize.Y, sigma2)
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.Filter2DSep(src, kx, ky, 0)
	})
}

// ---------------------------------------------------------------------------
// Derivatives
// ---------------------------------------------------------------------------

// CreateSobelFilter returns a [Filter] computing the Sobel derivative of orders
// (dx, dy) with aperture ksize (1 or 3), mirroring cv::cuda::createSobelFilter.
// The result is scaled by scale, offset by delta and clamped to [0,255].
// borderType is advisory. It delegates to [cv.Sobel].
func CreateSobelFilter(dx, dy, ksize int, scale, delta float64, borderType BorderType) Filter {
	_ = borderType
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.Sobel(src, dx, dy, ksize, scale, delta)
	})
}

// CreateScharrFilter returns a [Filter] computing the 3×3 Scharr derivative,
// mirroring cv::cuda::createScharrFilter. Exactly one of dx, dy must be 1 and
// the other 0. The result is scaled, offset by delta and clamped. borderType is
// advisory. It delegates to [cv.Scharr].
func CreateScharrFilter(dx, dy int, scale, delta float64, borderType BorderType) Filter {
	_ = borderType
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.Scharr(src, dx, dy, scale, delta)
	})
}

// CreateDerivFilter returns a general derivative [Filter] of orders (dx, dy),
// mirroring cv::cuda::createDerivFilter. ksize selects the operator: a ksize of
// -1 (or the sentinel that OpenCV spells FILTER_SCHARR) uses the 3×3 Scharr
// kernel — valid only when dx+dy == 1 — while ksize of 1 or 3 uses Sobel. The
// result is scaled, offset by delta and clamped. borderType is advisory. It
// delegates to [cv.Scharr] or [cv.Sobel].
func CreateDerivFilter(dx, dy, ksize int, scale, delta float64, borderType BorderType) Filter {
	_ = borderType
	if ksize <= 0 {
		return filterFunc(func(src *cv.Mat) *cv.Mat {
			return cv.Scharr(src, dx, dy, scale, delta)
		})
	}
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.Sobel(src, dx, dy, ksize, scale, delta)
	})
}

// CreateLaplacianFilter returns a [Filter] computing the Laplacian with aperture
// ksize (1 or 3), mirroring cv::cuda::createLaplacianFilter. The result is
// scaled, offset by delta and clamped. borderType is advisory. It delegates to
// [cv.Laplacian].
func CreateLaplacianFilter(ksize int, scale, delta float64, borderType BorderType) Filter {
	_ = borderType
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.Laplacian(src, ksize, scale, delta)
	})
}

// ---------------------------------------------------------------------------
// Rank filters (box max/min, median)
// ---------------------------------------------------------------------------

// CreateBoxMaxFilter returns a [Filter] replacing each sample with the maximum
// over a ksize rectangular window, mirroring cv::cuda::createBoxMaxFilter. This
// is a dilation by a rectangular structuring element. anchor must be
// [AnchorCenter]; borderType is advisory. It delegates to
// [cv.GetStructuringElement] and [cv.Dilate].
func CreateBoxMaxFilter(ksize image.Point, anchor image.Point, borderType BorderType) Filter {
	requirePositiveOdd(ksize.X, "CreateBoxMaxFilter (ksize.X)")
	requirePositiveOdd(ksize.Y, "CreateBoxMaxFilter (ksize.Y)")
	requireCenterAnchor(anchor, "CreateBoxMaxFilter")
	_ = borderType
	kernel := cv.GetStructuringElement(cv.MorphRect, ksize.Y, ksize.X)
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.Dilate(src, kernel, 1)
	})
}

// CreateBoxMinFilter returns a [Filter] replacing each sample with the minimum
// over a ksize rectangular window, mirroring cv::cuda::createBoxMinFilter. This
// is an erosion by a rectangular structuring element. anchor must be
// [AnchorCenter]; borderType is advisory. It delegates to
// [cv.GetStructuringElement] and [cv.Erode].
func CreateBoxMinFilter(ksize image.Point, anchor image.Point, borderType BorderType) Filter {
	requirePositiveOdd(ksize.X, "CreateBoxMinFilter (ksize.X)")
	requirePositiveOdd(ksize.Y, "CreateBoxMinFilter (ksize.Y)")
	requireCenterAnchor(anchor, "CreateBoxMinFilter")
	_ = borderType
	kernel := cv.GetStructuringElement(cv.MorphRect, ksize.Y, ksize.X)
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.Erode(src, kernel, 1)
	})
}

// CreateMedianFilter returns a [Filter] replacing each sample with the median of
// its windowSize×windowSize neighbourhood, mirroring the intent of cv::cuda::
// createMedianFilter. windowSize must be positive and odd. It delegates to
// [cv.MedianBlur].
func CreateMedianFilter(windowSize int) Filter {
	requirePositiveOdd(windowSize, "CreateMedianFilter")
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.MedianBlur(src, windowSize)
	})
}

// ---------------------------------------------------------------------------
// Morphology
// ---------------------------------------------------------------------------

// CreateMorphologyFilter returns a [Filter] performing the morphological
// operation op over the structuring element kernel, mirroring cv::cuda::
// createMorphologyFilter. kernel is a single-channel [cv.Mat] whose non-zero
// entries define the footprint — build one with [cv.GetStructuringElement].
// iterations repeats the operation (values < 1 are treated as 1). anchor must be
// [AnchorCenter]. It delegates to [cv.MorphologyEx] (or [cv.Erode]/[cv.Dilate]).
func CreateMorphologyFilter(op MorphOp, kernel *cv.Mat, anchor image.Point, iterations int) Filter {
	if kernel == nil || kernel.Empty() {
		panic("cudafilters: CreateMorphologyFilter requires a non-empty kernel")
	}
	requireCenterAnchor(anchor, "CreateMorphologyFilter")
	cvOp := op.toCVMorphType()
	k := kernel.Clone()
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.MorphologyEx(src, k, cvOp, iterations)
	})
}

// CreateErodeFilter is a shortcut for [CreateMorphologyFilter] with
// [MorphErode].
func CreateErodeFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter {
	return CreateMorphologyFilter(MorphErode, kernel, anchor, iterations)
}

// CreateDilateFilter is a shortcut for [CreateMorphologyFilter] with
// [MorphDilate].
func CreateDilateFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter {
	return CreateMorphologyFilter(MorphDilate, kernel, anchor, iterations)
}

// CreateOpenFilter is a shortcut for [CreateMorphologyFilter] with [MorphOpen].
func CreateOpenFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter {
	return CreateMorphologyFilter(MorphOpen, kernel, anchor, iterations)
}

// CreateCloseFilter is a shortcut for [CreateMorphologyFilter] with
// [MorphClose].
func CreateCloseFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter {
	return CreateMorphologyFilter(MorphClose, kernel, anchor, iterations)
}

// CreateMorphologyGradientFilter is a shortcut for [CreateMorphologyFilter] with
// [MorphGradient].
func CreateMorphologyGradientFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter {
	return CreateMorphologyFilter(MorphGradient, kernel, anchor, iterations)
}

// CreateTopHatFilter is a shortcut for [CreateMorphologyFilter] with
// [MorphTophat].
func CreateTopHatFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter {
	return CreateMorphologyFilter(MorphTophat, kernel, anchor, iterations)
}

// CreateBlackHatFilter is a shortcut for [CreateMorphologyFilter] with
// [MorphBlackhat].
func CreateBlackHatFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter {
	return CreateMorphologyFilter(MorphBlackhat, kernel, anchor, iterations)
}

// ---------------------------------------------------------------------------
// Running sums
// ---------------------------------------------------------------------------

// CreateRowSumFilter returns a [Filter] computing, for every pixel, the sum of
// its ksize horizontal neighbours, mirroring cv::cuda::createRowSumFilter.
//
// Honesty note: the running sum can exceed 255. Because [cv.Mat] holds 8-bit
// samples, sums saturate at 255 (OpenCV's CUDA version would widen the depth).
// ksize must be positive and odd. It delegates to [cv.Filter2DSep] with an
// all-ones horizontal kernel.
func CreateRowSumFilter(ksize int) Filter {
	requirePositiveOdd(ksize, "CreateRowSumFilter")
	row := onesKernel(ksize)
	col := []float64{1}
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.Filter2DSep(src, row, col, 0)
	})
}

// CreateColumnSumFilter returns a [Filter] computing, for every pixel, the sum
// of its ksize vertical neighbours, mirroring cv::cuda::createColumnSumFilter.
//
// Honesty note: as with [CreateRowSumFilter], sums saturate at 255 because
// [cv.Mat] is 8-bit. ksize must be positive and odd. It delegates to
// [cv.Filter2DSep] with an all-ones vertical kernel.
func CreateColumnSumFilter(ksize int) Filter {
	requirePositiveOdd(ksize, "CreateColumnSumFilter")
	row := []float64{1}
	col := onesKernel(ksize)
	return filterFunc(func(src *cv.Mat) *cv.Mat {
		return cv.Filter2DSep(src, row, col, 0)
	})
}

// onesKernel returns a length-n slice of 1.0 weights.
func onesKernel(n int) []float64 {
	k := make([]float64, n)
	for i := range k {
		k[i] = 1
	}
	return k
}
