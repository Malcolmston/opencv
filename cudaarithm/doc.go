// Package cudaarithm is a CPU-backed, API-compatible mirror of OpenCV's
// cudaarithm module. It provides the shape of OpenCV's CUDA per-element and
// matrix-reduction API — a [GpuMat] device-matrix type, a no-op async [Stream],
// and a large family of element-wise, mathematical, channel and reduction
// operations — implemented entirely in pure Go on top of the root module
// github.com/malcolmston/opencv (imported as cv) and the Go standard library.
//
// # Honest scope: there is no GPU
//
// This package performs NO GPU computation. There is no cgo, no CUDA runtime,
// and no device memory. Every operation runs on the CPU against the ordinary
// host-resident [cv.Mat] that a [GpuMat] wraps. The CUDA vocabulary is
// reproduced so that code written against OpenCV's cudaarithm reads and ports
// naturally, but the following are deliberate, honest fictions:
//
//   - [GpuMat] is a thin wrapper around a *cv.Mat. "Uploading" and
//     "downloading" ([GpuMat.Upload], [GpuMat.Download], [NewGpuMat]) copy the
//     matrix rather than moving it across a PCIe bus, because host and device
//     memory are the same memory here. The copy semantics are preserved (the
//     returned/stored Mat does not alias the caller's), which keeps behaviour
//     predictable, but there is no acceleration.
//   - [Stream] is a no-op. Every operation accepts an optional trailing
//     stream argument for source compatibility, but work is always performed
//     synchronously before the call returns, so [Stream.WaitForCompletion] has
//     nothing to wait for.
//
// # The uint8 constraint
//
// The root package's [cv.Mat] stores 8-bit unsigned samples (depth CV_8U).
// This package therefore operates on 8-bit matrices and, like the root
// package's own arithmetic, rounds by adding 0.5 and truncating toward zero
// before saturating into [0,255]. Operations whose mathematical range exceeds a
// byte — [Pow], [Exp], [Gemm], [Magnitude] and friends — saturate their
// results; each documents this. The Fourier routines [DFT], [IDFT] and
// [MulSpectrums] cannot represent complex spectra in a uint8 matrix at all, so
// they exchange the dedicated float-valued [ComplexMat] type instead of a
// [GpuMat]; this is the one place the API shape necessarily diverges, and it is
// documented on those functions.
//
// # Operation families
//
// Element-wise: [Add], [Subtract], [Multiply], [Divide], [AbsDiff],
// [AddWeighted], [BitwiseAnd], [BitwiseOr], [BitwiseXor], [BitwiseNot], [Min],
// [Max], [Compare], [Threshold] and [Abs] mirror the root package's saturating
// arithmetic on [GpuMat] operands.
//
// Mathematical: [Pow], [Exp], [Log], [Sqrt], [Magnitude], [Phase],
// [CartToPolar] and [PolarToCart] compute in float64 and saturate to 8 bits.
//
// Channel and layout: [Split], [Merge], [Transpose], [Flip], [LUT] and
// [Normalize] rearrange or remap samples.
//
// Reductions: [Sum], [AbsSum], [SqrSum], [MinMax], [MinMaxLoc], [CountNonZero],
// [Mean], [MeanStdDev] and [Norm] collapse a matrix to scalars.
//
// Linear algebra and spectra: [Gemm] is a saturating general matrix multiply;
// [DFT]/[IDFT] are a genuine (naive, exact) discrete Fourier transform and its
// inverse, and [MulSpectrums] multiplies two spectra element-wise.
//
// All functions validate their arguments and panic with a descriptive message
// on programmer error (mismatched shapes, empty inputs), matching the root
// package's convention. Inputs are never mutated; every result is freshly
// allocated.
package cudaarithm
