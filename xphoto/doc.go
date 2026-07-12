// Package xphoto is a standard-library-only port of a useful subset of
// OpenCV's xphoto contrib module, built on top of the root cv package
// (github.com/malcolmston/opencv). It collects "extra" photo-processing
// algorithms that sit alongside the mainline photo module: automatic white
// balance, per-channel gain application, exemplar/shift-based inpainting, oil
// painting stylization and BM3D denoising. It also provides several classical
// colour-constancy estimators (shades-of-gray, white-patch, gray-edge), gamma
// correction, dark-channel-prior haze removal, frequency-selective
// reconstruction (FSR) inpainting and Durand bilateral tone mapping.
//
// # Data model
//
// Every function operates on the root package's [cv.Mat], a dense row-major
// matrix of 8-bit unsigned samples. Three-channel data is treated as RGB,
// matching Go's image package and the root cv conventions, not OpenCV's native
// BGR. Where a function is channel-agnostic it says so; where it requires a
// specific channel count it validates and panics on mismatch, mirroring the
// root package's fail-fast style.
//
// # Dependencies
//
// xphoto imports only the root cv package and the Go standard library (math,
// sort, fmt, image). It uses no cgo and no third-party code. Small numeric
// helpers (edge-replicated sampling, rounding-and-clamping to [0,255], luma,
// Sobel gradients) are reimplemented locally rather than reaching into cv's
// unexported internals, so the package is self-contained.
//
// # Determinism
//
// All algorithms are fully deterministic: they use no randomness, no
// goroutine-dependent iteration order and no floating-point reductions whose
// result depends on scheduling. Given the same input and parameters a function
// always returns byte-identical output. This makes the results reproducible and
// testable.
//
// # White balance
//
// The three white balancers share the [WhiteBalancer] interface
// (BalanceWhite). [SimpleWB] performs an independent per-channel histogram
// stretch with configurable input/output ranges and percentile clipping.
// [GrayworldWB] applies the gray-world assumption with a saturation threshold.
// [LearningBasedWB] approximates OpenCV's learning-based estimator: it extracts
// the same brightness/edge-weighted chromaticity color features and combines
// them with a gray-edge estimate through a robust (median) combiner. See the
// Deferred section for the honesty note on the trained regressor.
// [ApplyChannelGains] multiplies the R, G and B channels by explicit gains.
//
// Each balancer also has an OpenCV-style factory ([CreateSimpleWB],
// [CreateGrayworldWB], [CreateLearningBasedWB]) and getX/setX property
// accessors, so code ported from the C++ API reads unchanged.
//
// Beyond those three, the package ports the classical colour-constancy family:
// [ShadesOfGray] estimates the illuminant with a general Minkowski p-norm
// (p == 1 is gray-world, p -> infinity is white-patch), [WhitePatchWB] is the
// max-RGB special case, [GrayEdgeWB] applies the same norm to image gradients,
// and [AutoWhiteBalance] is a tuning-free entry point using a moderate norm.
//
// # Other photo operations
//
// [GammaCorrection] applies a LUT-based power-law transfer function.
// [Dehaze] / [DarkChannelDehazer] remove haze with He et al.'s dark channel
// prior (atmospheric-light estimation, dark-channel transmission and an
// edge-aware guided-filter refinement). [DctDenoising] denoises with sliding-
// window DCT hard-thresholding. [Bm3dDenoisingTwoStep] and
// [Bm3dDenoisingStep2] add BM3D's empirical-Wiener second stage. [InpaintFSR]
// reconstructs masked regions by frequency-selective reconstruction.
// [OilpaintingColorSpace] parametrises the oil-painting intensity source.
// [TonemapDurand] performs bilateral-filter base/detail tone mapping.
//
// # Algorithms and fidelity
//
// The implementations favour clarity and correctness over raw speed and are
// faithful in spirit rather than bit-exact with OpenCV:
//
//   - [Inpaint] fills a masked region with a SHIFTMAP-style exemplar search:
//     an onion-peel front is advanced inward, and each unknown pixel is filled
//     from the best-matching known patch (the translation/"shift" whose
//     overlapping known samples minimise sum-of-squared difference). This
//     mirrors the intent of xphoto::inpaint's shift-map / patch-based fill.
//   - [Oilpainting] reproduces xphoto::oilPainting exactly in structure: within
//     each neighbourhood it finds the most common intensity bucket and outputs
//     the mean colour of the pixels in that bucket, collapsing texture into
//     flat, painterly patches.
//   - [Bm3dDenoising] implements a genuine single-step (hard-threshold) BM3D:
//     block matching groups similar patches, a separable 2D-DCT plus a 1D-DCT
//     across the group form the collaborative transform, coefficients are
//     hard-thresholded, and inverse-transformed blocks are aggregated with
//     confidence weights.
//
// # Deferred
//
// The following OpenCV xphoto features are intentionally out of scope or
// simplified, and are documented here so callers are not misled:
//
//   - LearningBasedWB's trained regression model. OpenCV ships a gradient-boosted
//     regression tree trained on a labelled illuminant dataset and loaded from a
//     .yml model file. This port has no trained model; it replaces the regressor
//     with a self-contained robust estimator over the same colour/edge features.
//     Its accuracy is therefore approximate, not identical to OpenCV's.
//   - The full SHIFTMAP EM optimisation. OpenCV's shift-map inpainting solves a
//     global energy with a multi-scale expansion-move / EM scheme. This port
//     uses a greedy onion-peel exemplar fill, which is local rather than
//     globally optimal.
//   - True high-dynamic-range input for [TonemapDurand]. OpenCV's Durand
//     operator maps a floating-point HDR radiance map to LDR; this port operates
//     on the package's 8-bit [cv.Mat], so it functions as a faithful local-
//     contrast / detail-preserving operator over a low-dynamic-range proxy
//     rather than a genuine HDR-to-LDR compressor.
//   - [InpaintFSR] uses binary support weighting and a per-block DCT
//     matching-pursuit model. OpenCV's FSR additionally applies an isotropic
//     spatial weighting function and a residual-energy stopping rule; this port
//     uses a fixed iteration budget per block, so it is faithful in structure
//     but not bit-exact.
package xphoto
