// Package cudawarping is a CPU-backed, API-compatible mirror of OpenCV's
// cudawarping module (the cv::cuda geometric image transformations) built on the
// standard-library-only OpenCV port github.com/malcolmston/opencv, imported here
// as cv.
//
// # No GPU is used
//
// Despite the name, this package contains no CUDA, no cgo and no GPU code. It
// exists so that Go programs written against OpenCV's cudawarping surface — the
// [GpuMat] container, a [Stream] handle, and warping calls that take flags,
// border modes and streams — compile and run using only the Go standard library
// and the root cv package. Every [GpuMat] holds its pixels in ordinary host
// memory (a wrapped [cv.Mat]) and every operation is computed synchronously on
// the CPU. [Upload] and [GpuMat.Download] are deep copies rather than
// host↔device transfers, and [Stream] is an inert handle whose methods are
// no-ops. Results match the CPU reference implementation in cv; they are not
// bit-exact with a real CUDA build, and there is no performance benefit over
// calling cv directly. Use this package for portability and drop-in source
// compatibility, not for acceleration.
//
// # Containers
//
// [NewGpuMat] allocates a blank device matrix and [Upload] moves a host
// [cv.Mat] into one; [GpuMat.Download] copies it back. [GpuMat.Clone],
// [GpuMat.Size], [GpuMat.Channels], [GpuMat.Empty] and [GpuMat.Release] round
// out the container, and [NewStream] creates a no-op stream that every
// operation accepts and ignores.
//
// # Geometric transforms
//
// [GpuMat.Resize] scales an image with nearest, bilinear, bicubic or
// pixel-area interpolation (nearest and bilinear delegate to [cv.Resize]; cubic
// and area are computed locally). [GpuMat.WarpAffine] and
// [GpuMat.WarpPerspective] apply 2×3 affine and 3×3 projective transforms,
// honouring the [WarpInverseMap] flag (the matrix already maps destination to
// source) and the full set of [BorderMode] values with a constant border value.
// [GpuMat.Rotate] performs an arbitrary-angle rotation about the origin with an
// optional shift, while [GpuMat.Rotate90] does the three lossless right-angle
// rotations. [GpuMat.Remap] resamples through explicit coordinate maps, and
// [BuildWarpAffineMaps], [BuildWarpPerspectiveMaps] and [BuildRotationMaps]
// pre-compute those maps.
//
// # Pyramids, flips and borders
//
// [GpuMat.PyrDown] and [GpuMat.PyrUp] are the Gaussian-pyramid resampling steps,
// [GpuMat.Transpose] swaps rows and columns, [GpuMat.Flip] mirrors about either
// or both axes (using OpenCV's integer flip-code convention) and
// [GpuMat.CopyMakeBorder] pads an image under any [BorderMode].
//
// # Polar warps
//
// [GpuMat.WarpPolar] converts between Cartesian and polar coordinates about a
// chosen centre, with a linear ([WarpPolarLinear]) or logarithmic
// ([WarpPolarLog]) radial axis and an optional inverse ([WarpInverseMap]) pass;
// [GpuMat.LinearPolar] and [GpuMat.LogPolar] are the convenience forms.
//
// # Conventions and determinism
//
// Sizes are given as an [image.Point] whose X is the width and Y is the height,
// matching cv::Size. Interpolation, warp, border and polar-mode constants take
// the same numeric values as their OpenCV counterparts so flags copied from
// OpenCV code keep their meaning. Every function is deterministic and performs
// no concurrent work, so identical inputs always yield identical output. The
// package imports only the Go standard library and the root cv package; it
// imports no sibling cv/* subpackages and modifies no state outside a returned
// value.
package cudawarping
