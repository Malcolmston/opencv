// Package cudabgsegm is a CPU-backed, API-compatible mirror of OpenCV's
// cudabgsegm module — the GPU background-segmentation subtractors that in
// upstream OpenCV live under the cv::cuda namespace (createBackgroundSubtractorMOG,
// createBackgroundSubtractorMOG2, createBackgroundSubtractorGMG and
// createBackgroundSubtractorFGD).
//
// # Honest scope
//
// There is no CUDA, no GPU and no cgo here. This package is pure Go built only
// on the standard library, the root package [github.com/malcolmston/opencv]
// (imported under the alias cv) and the sibling CPU package
// [github.com/malcolmston/opencv/bgsegm], to which every model delegates. The
// point is source and shape compatibility: code written against the OpenCV CUDA
// bgsegm API — a [GpuMat] holding the frame, a [Stream] threaded through each
// call, factory functions named createBackgroundSubtractor* and the matching
// getter/setter methods — ports across with minimal edits, while the actual
// pixels are classified on the CPU by the [github.com/malcolmston/opencv/bgsegm]
// models.
//
// Because the compute is identical to the CPU package, the GPU-specific
// performance characteristics of the real module are not reproduced. The
// [Stream] type is a no-op: it exists so signatures match, and it does nothing.
// The per-call learningRate argument accepted by the MOG-family Apply methods is
// likewise accepted for signature compatibility; the CPU models derive their own
// adaptive rate from the configured history and frame count, so a caller-supplied
// rate other than the OpenCV "auto" sentinel (-1) is documented where it is
// ignored.
//
// # Types
//
// A [GpuMat] is a thin wrapper around a *[cv.Mat]. In real OpenCV a GpuMat lives
// in device memory and must be uploaded to and downloaded from host memory
// explicitly; here [GpuMat.Upload] and [GpuMat.Download] are ordinary copies, but
// they are provided so the upload/download idiom compiles and runs unchanged.
//
// # Models
//
//   - [BackgroundSubtractorMOG]  — original KaewTraKulPong–Bowden Gaussian mixture,
//     via [bgsegm.BackgroundSubtractorMOG].
//   - [BackgroundSubtractorMOG2] — Zivkovic adaptive Gaussian mixture with shadow
//     detection, via [bgsegm.BackgroundSubtractorMOG2].
//   - [BackgroundSubtractorGMG]  — Godbehere–Matsukawa–Goldberg Bayesian model,
//     via [bgsegm.BackgroundSubtractorGMG].
//
// # Deferred
//
// OpenCV's cudabgsegm also exposes createBackgroundSubtractorFGD (the Li et al.
// foreground–background statistical model). Neither the [bgsegm] package nor the
// root cv package supplies an FGD-equivalent model, so it is intentionally not
// mirrored here. It is deferred until a CPU FGD implementation exists to delegate
// to; adding it would otherwise mean writing the algorithm in this package, which
// is out of scope for a thin compatibility mirror.
//
// # Masks
//
// Every Apply returns a fresh single-channel foreground mask wrapped in a
// [GpuMat]; samples are [bgsegm.BackgroundValue] (0), [bgsegm.ForegroundValue]
// (255) or, when shadow detection is enabled, [bgsegm.ShadowValue] (127).
package cudabgsegm
