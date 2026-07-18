// Package filters2 is a standard-library-only collection of advanced image
// filtering, denoising and multi-scale analysis routines built on top of the
// parent package's [github.com/malcolmston/opencv.Mat] image type.
//
// It complements the elementary linear filters offered by the root package
// with the heavier machinery of computer-vision filtering:
//
//   - Edge-preserving smoothing and denoising: bilateral and joint (cross)
//     bilateral filtering, the guided filter of He et al., non-local means,
//     Perona-Malik anisotropic diffusion and the Kuwahara filter.
//   - Rank filters: plain and adaptive median, min/max, midpoint and
//     alpha-trimmed-mean filters.
//   - Sharpening: unsharp masking and high-boost filtering.
//   - Multi-scale band-pass analysis: the Laplacian-of-Gaussian, the
//     difference-of-Gaussians and Marr-Hildreth zero-crossing edges.
//   - Oriented analysis: Gabor kernels and Gabor filter banks, and the
//     Freeman-Adelson steerable first- and second-derivative-of-Gaussian
//     filters.
//
// # Conventions
//
// Denoising and rank filters accept and return [cv.Mat] values and preserve
// the channel count of their input, processing each channel independently
// unless documented otherwise (the bilateral and non-local-means range terms
// use the full colour vector). Coordinates follow the parent package: x is the
// column, y is the row, origin at the top-left, and all border handling uses
// edge replication.
//
// Linear analysis filters whose output is signed and unbounded (Laplacian of
// Gaussian, difference of Gaussians, Gabor and steerable responses) operate on
// single-channel input and return the exported [FloatImage] helper — a plain
// single-channel float64 raster that converts to and from [cv.Mat] rather than
// duplicating it. Use [FloatImage.Normalize] to map a response to a viewable
// 8-bit image or [FloatImage.ToMat] to clamp it directly.
//
// The implementation is pure Go, uses no cgo and no third-party dependencies,
// is CPU-only and fully deterministic.
package filters2
