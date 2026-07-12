// Package cudastereo is a standard-library-only, CPU-backed mirror of OpenCV's
// cudastereo module (cv::cuda stereo correspondence). It reproduces that
// module's public API — a [GpuMat] device-matrix wrapper, a no-op [Stream],
// and the StereoBM / StereoBeliefPropagation / StereoConstantSpaceBP /
// StereoSGM matchers plus the DisparityBilateralFilter, reprojectImageTo3D and
// drawColorDisp helpers — so code written against the CUDA module ports across
// with only cosmetic changes.
//
// # Honesty about the "GPU"
//
// There is no GPU here and no CUDA. This package is pure Go: it imports only the
// Go standard library, the root cv package
// (github.com/malcolmston/opencv, aliased cv) and the sibling
// github.com/malcolmston/opencv/stereo package. There is no cgo, no device
// memory and no kernel launch. [GpuMat] is a thin wrapper around a host
// [github.com/malcolmston/opencv.Mat]; Upload and Download are ordinary deep
// copies; [Stream] is an inert placeholder whose WaitForCompletion returns
// immediately because every operation already ran synchronously on the CPU.
// Passing a nil [Stream] is always valid.
//
// The value of the package is API compatibility and identical numerical
// behaviour, not acceleration. Every routine here runs on the CPU and will be
// slower than a real CUDA build.
//
// # What is delegated and what is implemented here
//
// The block matchers reuse the tested engines in the sibling stereo package:
//
//   - [StereoBM] delegates to stereo.StereoBM (local SAD block matching).
//   - [StereoSGM] delegates to stereo.StereoSGM (semi-global matching).
//   - [ReprojectImageTo3D] delegates to stereo.ReprojectImageTo3D.
//
// The two belief-propagation matchers, which the stereo package deliberately
// leaves out of scope, are implemented here from scratch:
//
//   - [StereoBeliefPropagation] is a genuine hierarchical loopy belief
//     propagation matcher (Felzenszwalb–Huttenlocher): a truncated-linear data
//     cost, a truncated-linear smoothness prior evaluated with the O(ndisp)
//     lower-envelope message update, four-connected message passing iterated to
//     (approximate) convergence, and a coarse-to-fine pyramid over NumLevels.
//   - [StereoConstantSpaceBP] is a constant-space belief propagation matcher:
//     each pixel keeps only its NrPlane cheapest disparity hypotheses, so the
//     message storage is independent of the disparity range, and messages are
//     passed between neighbouring pixels' plane sets.
//
// [DisparityBilateralFilter] and [DrawColorDisp] are likewise implemented here
// directly on the host Mat.
//
// # Disparity maps
//
// Matchers return a single-channel 8-bit [GpuMat] whose samples are integer
// disparities in pixels. As in the stereo package, the value 0 doubles as the
// "no reliable match" marker for the block matchers; the belief-propagation
// matchers fill every pixel.
//
// # Geometry
//
// All matchers assume a rectified pair: row-aligned cameras, so a scene point
// shares an image row in both views and its horizontal offset is the disparity
// d = xLeft - xRight, searched over d in [0, NumDisparities). Rectify the input
// yourself first; this package does not calibrate.
package cudastereo
