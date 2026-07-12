// Package hdr is a standard-library-only implementation of the high-dynamic-range
// imaging pipeline from OpenCV's photo module, built on top of the root cv
// package (github.com/malcolmston/opencv).
//
// It depends only on the Go standard library (math, sort) and the root cv
// package: no cgo, no third-party code, and no dependency on any sibling cv/*
// subpackage. Small numeric helpers (rounding-and-clamping to [0,255], luma,
// separable Gaussian blur, image pyramids) are reimplemented locally rather
// than reaching into the internals of other packages.
//
// # The pipeline
//
// HDR imaging fuses a bracket of low-dynamic-range (LDR) exposures of a static
// scene into a single image that captures detail across the whole tonal range.
// The classic three-stage pipeline, all of which this package implements, is:
//
//  0. Optionally align a hand-held bracket. [AlignMTB] implements Ward's (2003)
//     median-threshold-bitmap aligner (createAlignMTB), recovering the
//     whole-pixel translation between frames with [AlignMTB.CalculateShift] and
//     registering a whole stack with [AlignMTB.Process].
//  1. Calibrate the camera response function (CRF) that maps scene radiance to
//     recorded 8-bit pixel values. [CalibrateDebevec] recovers it with the
//     Debevec & Malik (1997) least-squares method (data term + second-order
//     smoothness + a hat weighting); [CalibrateRobertson] offers the Robertson
//     (1999) iterative maximum-likelihood alternative. Recovered curves can be
//     inspected and repaired with the [CameraResponse] accessors
//     ([CameraResponse.Response], [CameraResponse.ChannelCurve],
//     [CameraResponse.IsMonotonic], [CameraResponse.EnforceMonotonic],
//     [CameraResponse.Normalize]).
//  2. Merge the LDR stack into a linear radiance map. [MergeDebevec] performs
//     the weighted log-radiance average using the calibrated response and the
//     per-image exposure times, producing a [Radiance] (a multi-channel float
//     image); [MergeDebevecFunc] lets the caller supply a robustness
//     [WeightFunc] ([HatWeight], [TentWeight], [GaussianWeight],
//     [UniformWeight]). [MergeRobertson] is the Robertson-estimator merge.
//     [MergeMertens] (and the reusable [MergeMertensProcessor]) instead perform
//     Mertens exposure fusion, blending the stack directly into a displayable
//     image with no response or exposure times required.
//  3. Tonemap the radiance map into a displayable 8-bit image. All tonemappers
//     satisfy the [Tonemap] interface. [TonemapGamma] is a plain gamma curve;
//     [TonemapReinhard], [TonemapDrago], [TonemapMantiuk], [TonemapDurand]
//     (bilateral base/detail) and [TonemapMantiukGradient] (gradient-domain
//     Poisson reconstruction) compress dynamic range while preserving local
//     detail. [DetailEnhance] and [EdgePreservingFilter] finish a tonemapped
//     image, and [ApplyColorMap] / [Radiance.Visualize] render a radiance map in
//     false colour.
//
// Radiance maps can be measured with [Radiance.MinMax], [Radiance.Mean],
// [Radiance.LogAverageLuminance] and [Radiance.DynamicRange], and exchanged with
// other tools through the PFM ([WritePFM]/[ReadPFM]) and Radiance RGBE
// ([WriteHDR]/[ReadHDR]) float-image formats.
//
// # Conventions
//
// Three-channel images are treated as RGB, matching Go's image package and the
// root cv conventions, not OpenCV's native BGR. LDR inputs are the root
// package's [github.com/malcolmston/opencv.Mat] (8-bit). Exposure times are in
// seconds, one per image, and must be strictly positive.
//
// Linear scene radiance is stored in a [Radiance] value. The root package's
// [github.com/malcolmston/opencv.FloatMat] is single-channel only, so this
// package defines [Radiance] as its own dense, interleaved multi-channel float
// image; a single channel of it can be extracted as a cv.FloatMat with
// [Radiance.ChannelFloatMat]. A [CameraResponse] holds, per channel, a 256-entry
// lookup table giving the linear radiance that corresponds to each 8-bit pixel
// value.
//
// # Determinism
//
// Every routine is fully deterministic. Calibration draws its sample pixels
// from a fixed, evenly spaced grid (never from a random number generator), so
// repeated calls on the same inputs return bit-identical results.
//
// # Deferred
//
// The following OpenCV HDR features are intentionally out of scope:
//
//   - Sub-pixel and rotational alignment. [AlignMTB] corrects whole-pixel
//     translation only, as in OpenCV.
//   - Run-length-encoded (RLE) Radiance RGBE scanlines. [WriteHDR] and [ReadHDR]
//     use the uncompressed "flat" encoding, which every reader accepts.
//   - Bad-pixel / ghost removal and any GPU acceleration.
//
// [TonemapMantiukGradient] provides a genuine gradient-domain operator (gradient
// attenuation followed by a Poisson solve); [TonemapMantiuk] remains the faster
// single-surround local-contrast approximation.
package hdr
