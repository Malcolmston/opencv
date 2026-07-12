// Package fuzzy is a standard-library-only implementation of fuzzy (F-)transform
// image processing built on the root cv package (github.com/malcolmston/opencv).
// It ports a useful subset of OpenCV's fuzzy / ft contrib module: fuzzy-partition
// kernels (symmetric, single-vector and two-vector forms), the two-dimensional
// degree-0 and degree-1 F-transforms (forward components and inverse
// reconstruction), a fast separable process variant, F-transform smoothing, and
// F-transform image inpainting (one-step, iterative, single-step and
// multi-resolution), together with numeric quality reporting.
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
// # Degree-1 (linear) F-transform
//
// The degree-1 transform replaces each node's single average with a local linear
// polynomial c00 + c10*(x-x0) + c01*(y-y0), whose coefficients are found by a
// fuzzy-weighted least-squares fit under the basis (the full 3x3 normal system is
// solved, so the fit stays correct at borders and under a validity mask). Because
// each node now carries a gradient, the inverse blends sloped planes rather than
// constants and reproduces ramps and gradients almost exactly — far better than
// degree-0 at the same radius. See [FT12DComponents], [FT12DInverse],
// [FT12DProcess], [FT12DPolynomial] and [Components1].
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
// itself), mirroring OpenCV's ft::createKernel. [CreateKernel1D] returns just the
// 1-D membership vector, [CreateKernelVec] forms the square kernel from a supplied
// vector, and [CreateKernelAB] forms the (possibly anisotropic) outer product of
// two vectors, mirroring OpenCV's two-vector createKernel overloads.
//
// # Transform API
//
//   - [FT02DComponents] computes the degree-0 forward components over a partition,
//     with an optional per-pixel validity mask (ft::FT02D_components).
//   - [FT02DInverse] reconstructs a full-size image from components, also exposed
//     as [Components.Reconstruct] (ft::FT02D_inverse).
//   - [FT02DProcess] does both in one call (ft::FT02D_process).
//   - [FT02DFLProcess] and [FT02DFLProcessFloat] are the fast, separable
//     linear-time process variants (ft::FT02D_FL_process[_float]).
//   - [FT02DIteration] runs a single masked inpainting step and reports progress
//     (ft::FT02D_iteration).
//   - [FT12DComponents] / [FT12DInverse] / [FT12DProcess] / [FT12DPolynomial] are
//     the degree-1 (linear-polynomial) family; [Components1.Reconstruct] and
//     [Components1.CoeffPlane] read them back.
//   - [Filter] is the convenience degree-0 smoother; [FilterLinear], [FilterSinus],
//     [FilterMultiRadius] and the gradient-preserving [FilterDegree1] are named
//     variants.
//   - [Inpaint] reconstructs masked-out pixels ([OneStep] and [Iterative]), and
//     [InpaintMultiStep] adds a coarse-to-fine multi-resolution scheme for large
//     holes, mirroring ft::inpaint.
//   - [TransformError] and [MaskedError] report reconstruction quality (MAE, RMSE,
//     max error and PSNR) as [ErrorStats].
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
// faithful in spirit rather than bit-exact with OpenCV. The degree-0 and degree-1
// (FT02D and FT12D) transforms, the fast separable process variant, two-vector
// kernels, the single-step iteration, the multi-resolution inpainter and quality
// reporting are all provided. The following parts of the OpenCV ft module remain
// intentionally NOT implemented:
//
//   - Degree-2 and higher transforms (only degrees 0 and 1 are provided).
//   - GPU/streamed kernels and OpenCV's internal fixed-point pipelines; every
//     routine here works in float64 on the CPU.
//   - OpenCV's exact border-darkening behaviour; this port normalises borders
//     instead (see above).
package fuzzy
