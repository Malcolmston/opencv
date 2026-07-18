// Package superres is a pure-Go, standard-library-only toolkit for image
// super-resolution, high-quality interpolation and the supporting building
// blocks those techniques need. It is a subpackage of the OpenCV port
// github.com/malcolmston/opencv (imported here as cv) and operates on the
// library's central [cv.Mat] type — a dense, row-major, channel-interleaved
// matrix of 8-bit samples — rather than defining any competing image type.
//
// Nothing in this package uses cgo, a GPU, trained model files or any
// third-party dependency: every algorithm is a classical, deterministic,
// weight-free method implemented against the Go standard library (math, sort
// and the cv core). Given the same input, every function returns the same
// output on every run and platform.
//
// # Scope
//
// The package is organised around the classic single- and multi-frame
// super-resolution pipeline:
//
//   - High-quality interpolation and resampling: bicubic (Keys/Catmull-Rom),
//     windowed-sinc Lanczos, cubic B-spline, Mitchell–Netravali and the
//     nearest/linear baselines, all through a single separable, anti-aliased
//     resampler (see resample.go). Downscaling automatically widens the kernel
//     to suppress aliasing.
//   - Edge-directed interpolation: the New Edge-Directed Interpolation (NEDI)
//     scheme of Li & Orchard, which estimates local covariance to interpolate
//     along rather than across edges (see nedi.go).
//   - Iterative back-projection super-resolution: repeatedly reprojects an
//     upscaled estimate through the imaging model and corrects the residual
//     (see backprojection.go).
//   - Example-free / self-example single-image SR: gradient-profile
//     sharpening applied across a progressive upscaling pyramid, needing no
//     external training dictionary (see examplefree.go).
//   - Sub-pixel shift estimation: gradient-based (Lucas–Kanade) and
//     phase-correlation translation estimators plus a sub-pixel image shifter
//     (see subpixel.go).
//   - Multi-frame fusion SR: register several low-resolution frames and fuse
//     them onto a common high-resolution grid by robust shift-and-add,
//     averaging or median combination (see fusion.go).
//   - Post-upscale sharpening: separable Gaussian blur, unsharp masking,
//     Laplacian sharpening and a combined upscale-then-sharpen convenience
//     (see sharpen.go).
//   - Quality metrics: MSE, MAE, PSNR and a windowed SSIM for validating the
//     above (see metrics.go).
//
// # Conventions
//
// Images are [cv.Mat] values with any positive channel count; each channel is
// processed independently, so grayscale (1 channel) and RGB (3 channel)
// inputs both work. Colour order (RGB vs BGR) is irrelevant to every routine
// here. Internally the iterative routines carry full-precision float64 planes
// and only quantise back to 8-bit at the end, avoiding round-off accumulation.
// Coordinates use the pixel-centre convention: output pixel (x, y) maps to
// source coordinate (x+0.5)·sx − 0.5, matching the root package's [cv.Resize].
package superres
