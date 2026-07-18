// Package pyramids implements image pyramids and scale-space machinery on top
// of the parent cv module's image types.
//
// The package builds directly on the parent package's [cv.Mat] (dense 8-bit
// images) and [cv.FloatMat] (single-channel float64 grids). Rather than
// introducing an incompatible image type, every routine here consumes and
// produces those two types so results interoperate with the rest of the
// module. Because pyramid arithmetic — Laplacian differences, wavelet detail
// coefficients, difference-of-Gaussian responses — routinely goes negative and
// must round-trip losslessly, the internal representation is the float64
// [cv.FloatMat]. Convert an 8-bit image to that domain with [GrayFloat] or
// [ChannelFloat], operate, then convert back with [FloatToMat] (clamping) or
// [FloatToMatNormalized] (contrast-stretched for visualisation).
//
// # What is provided
//
// Gaussian and Laplacian pyramids ([BuildGaussianPyramid],
// [BuildLaplacianPyramid]) with exact reconstruction; the single-level
// building blocks [PyrDownFloat] and [PyrUpFloat]; multi-resolution image
// blending ([BlendLaplacian], [BlendLaplacianMat], [MultiBandBlendMat]);
// difference-of-Gaussian scale space and blob detection ([BuildScaleSpace],
// [BuildDoGScaleSpace], [DetectDoGExtrema]); steerable first- and
// second-derivative filters with an oriented steerable pyramid
// ([SteerableBasisG1], [SteerG1], [BuildSteerablePyramid]); and an orthonormal
// Haar wavelet transform and pyramid ([HaarForward], [BuildWaveletPyramid]).
//
// # Conventions
//
// Coordinates follow the parent package: x is the column, y is the row, origin
// top-left. Neighbourhood operations replicate the border sample. All routines
// are pure Go, deterministic and CPU-only, matching the module's
// standard-library-only policy.
package pyramids
