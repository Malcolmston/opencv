// Package fuzzy is a standard-library-only implementation of degree-0 fuzzy
// (F-)transform image processing built on the root cv package
// (github.com/malcolmston/opencv). It ports a useful subset of OpenCV's fuzzy /
// ft contrib module: fuzzy-partition kernels, the two-dimensional degree-0
// F-transform (forward components and inverse reconstruction), F-transform
// smoothing, and F-transform image inpainting.
//
// # The F-transform in one paragraph
//
// The degree-0 F-transform covers the image domain with a grid of overlapping
// fuzzy basis (membership) functions — a "fuzzy partition". Each basis function
// contributes one component: the weighted average of the pixels beneath it,
// using the basis as the weight. Those components are a coarse, node-resolution
// summary of the image. The inverse transform paints each component back through
// its basis and, because the bases form a partition of unity (they sum to 1 at
// every pixel), the overlapping contributions blend into a smooth reconstruction
// of the original. Skipping unknown pixels while averaging — and skipping bases
// that see none — turns the very same machinery into an inpainter.
//
// # Basis functions and kernels
//
//   - [LinearBasis] is a triangular membership function.
//   - [SinusBasis] is a smoother raised-cosine membership function.
//
// Both satisfy the Ruspini partition-of-unity condition when partition nodes are
// spaced one kernel radius apart, which every routine here assumes. [CreateKernel]
// builds the square (2*radius+1) fuzzy-partition kernel as a [cv.FloatMat] from a
// basis function and radius (the outer product of the 1-D membership vector with
// itself), mirroring OpenCV's ft::createKernel.
//
// # Transform API
//
//   - [FT02DComponents] computes the forward components over a partition, with an
//     optional per-pixel validity mask (ft::FT02D_components).
//   - [FT02DInverse] reconstructs a full-size image from components
//     (ft::FT02D_inverse).
//   - [FT02DProcess] does both in one call (ft::FT02D_process).
//   - [Filter] is the convenience smoother: build a kernel, transform, invert.
//   - [Inpaint] reconstructs masked-out pixels ([OneStep] and [Iterative]),
//     mirroring ft::inpaint.
//
// # Conventions
//
// Images are the root package's [cv.Mat] (dense, row-major, 8-bit). Grayscale
// (1-channel) and colour (typically 3-channel RGB) images are both supported;
// every routine is channel-agnostic and processes each channel independently.
// Three-channel data is treated as RGB, matching the root cv and Go image
// conventions, not OpenCV's native BGR. Reconstructed sample values are rounded
// half-up and clamped to [0, 255]. Inputs are never mutated; results are fresh
// allocations.
//
// The partition is padded by one radius on the top and left (following OpenCV's
// ft module) so border pixels are covered; the inverse additionally normalises
// each pixel by the total basis weight reaching it, so partition-of-unity
// interior pixels are reproduced exactly and thinly-covered border pixels are
// handled without darkening.
//
// [Inpaint]'s mask marks the pixels to reconstruct (non-zero == unknown). This
// is the opposite polarity to OpenCV's raw ft::inpaint mask but matches the
// "mask of unknown pixels" convention used elsewhere in this module; internally
// it is inverted into the validity mask the F-transform consumes.
//
// # Determinism
//
// Every function is fully deterministic: no randomness, no floating-point
// parallelism, and a fixed accumulation order. Identical inputs always yield
// byte-identical outputs.
//
// # Fidelity and deferred features
//
// The implementations favour clarity and correctness over raw speed and are
// faithful in spirit rather than bit-exact with OpenCV. The following parts of
// the OpenCV ft module are intentionally NOT implemented:
//
//   - Higher-degree transforms (the F1 / FT12D family with linear, degree-1
//     components). Only the degree-0 (FT02D) transform is provided.
//   - The ft::MULTI_STEP inpainting algorithm (multi-resolution scheme). Only
//     [OneStep] and an [Iterative] variant are provided.
//   - Precomputed/streamed kernels and the multi-channel createKernel overload
//     that takes two separate 1-D vectors; here the kernel is always the square
//     symmetric outer product and is applied per channel.
//   - OpenCV's exact border-darkening behaviour; this port normalises borders
//     instead (see above).
package fuzzy
