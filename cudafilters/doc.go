// Package cudafilters is a pure-Go, CPU-backed, API-compatible mirror of
// OpenCV's cudafilters module. It reproduces the shape of that module — a
// [Filter] object interface obtained from Create* factory functions and applied
// with [Filter.Apply], together with a [GpuMat] device-matrix wrapper and a
// no-op [Stream] — but performs every computation on the CPU using the
// standard-library-only OpenCV port github.com/malcolmston/opencv (imported here
// as cv). It uses no cgo, no CUDA, no GPU and no third-party code.
//
// # Honesty note
//
// There is NO GPU acceleration here. OpenCV's real cudafilters offloads
// convolution and morphology to an NVIDIA GPU; this package instead delegates
// to the CPU filter primitives in the root package ([cv.Filter2D],
// [cv.GaussianBlur], [cv.Sobel], [cv.Erode], [cv.MorphologyEx], and friends).
// The [GpuMat] type does not allocate device memory — it holds an ordinary
// [cv.Mat]; Upload and Download are deep copies that model the host⇆device
// transfer boundary without any real transfer. The [Stream] type is a no-op
// placeholder so that call sites written against the CUDA API compile and run
// unchanged. The value of this package is source compatibility and identical
// numerical results to the corresponding root-package call, not speed.
//
// # Filter objects
//
// Following OpenCV, a filter is a reusable object created once and applied many
// times:
//
//	f := cudafilters.CreateGaussianFilter(image.Pt(5, 5), 1.0, 0)
//	dst := f.Apply(src)
//
// where src and dst are [GpuMat] values. Every factory returns a [Filter] whose
// Apply delegates to the matching root-package primitive, so
// f.Apply(src).Download() is byte-for-byte equal to calling that primitive on
// src.Download() directly. This is asserted by the package tests.
//
// # Factory functions
//
// Linear filtering: [CreateBoxFilter], [CreateLinearFilter],
// [CreateSeparableLinearFilter], [CreateGaussianFilter], [CreateBlurFilter].
//
// Derivatives: [CreateSobelFilter], [CreateScharrFilter], [CreateDerivFilter],
// [CreateLaplacianFilter].
//
// Rank filters: [CreateBoxMaxFilter], [CreateBoxMinFilter], [CreateMedianFilter].
//
// Morphology: [CreateMorphologyFilter] (with an [MorphOp] selector), plus the
// named shortcuts [CreateErodeFilter], [CreateDilateFilter], [CreateOpenFilter],
// [CreateCloseFilter], [CreateMorphologyGradientFilter], [CreateTopHatFilter]
// and [CreateBlackHatFilter].
//
// Running sums: [CreateRowSumFilter], [CreateColumnSumFilter].
//
// # Convenience wrappers
//
// Each factory has a one-call wrapper (for example [BoxFilter], [GaussianBlur],
// [Sobel], [Erode], [MorphologyEx]) that builds the filter, applies it to a
// single [GpuMat] and returns the result, for the common case where a filter is
// used exactly once.
//
// # Parameter conventions
//
// Two-dimensional kernel sizes and anchors are given as [image.Point] (X is the
// horizontal/column extent, Y is the vertical/row extent), matching the OpenCV
// Go bindings. The anchor sentinel [AnchorCenter] (that is, image.Pt(-1, -1))
// selects the kernel centre; the underlying engine only supports centred
// anchors, so a non-centred anchor is rejected. Border handling in the root
// package is always edge replication (BORDER_REPLICATE); a [BorderType] argument
// is accepted for API compatibility but is advisory — see [BorderType].
//
// Unlike OpenCV's cudafilters, the factories omit the srcType/dstType arguments:
// the engine operates on the single 8-bit unsigned depth of [cv.Mat], so those
// arguments would carry no information.
package cudafilters
