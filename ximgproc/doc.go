// Package ximgproc is a from-scratch, standard-library-only port of a useful
// subset of OpenCV's contrib module ximgproc ("extended image processing"). It
// is built entirely on top of the root package
// github.com/malcolmston/opencv (imported here as cv) and the Go standard
// library; it uses no cgo, no third-party code and it does not import any of
// the sibling cv/* subpackages.
//
// The module collects edge-preserving smoothers, morphological skeletons,
// local (adaptive) binarization methods and a superpixel segmenter — tools
// that live outside the core OpenCV modules but are staples of practical image
// processing pipelines. Every routine consumes and produces the central
// [cv.Mat] type (row-major, channel-interleaved, 8-bit samples), so results
// compose directly with the filters, colour conversions and thresholds in the
// root package.
//
// # Contents
//
//   - [GuidedFilter] — the edge-preserving smoothing filter of He, Sun and Tang
//     (2013). It behaves like a bilateral filter (smooths flat regions while
//     preserving strong edges) but has a simple linear-model formulation and no
//     range-Gaussian falloff, so it does not suffer from gradient reversal.
//   - [Thinning] — Zhang–Suen morphological thinning that reduces a binary
//     shape to a one-pixel-wide skeleton.
//   - [AnisotropicDiffusion] — Perona–Malik non-linear diffusion, an iterative
//     smoother whose conduction coefficient falls off across strong gradients so
//     that edges are preserved (or even sharpened) while noise is averaged away.
//   - [NiBlackThreshold] — local (per-pixel) binarization using a window mean
//     and standard deviation, with the classic Niblack, Sauvola, Wolf–Jolion and
//     NICK threshold formulas selectable by [NiBlackVariant].
//   - [SuperpixelSLIC] — Simple Linear Iterative Clustering, which groups pixels
//     into compact, roughly equally sized superpixels by k-means in a joint
//     colour-plus-position (Lab + xy) space, followed by a connectivity
//     enforcement pass so that every returned label is a single connected
//     region.
//   - [PeiLinNormalization] (bonus) — the moment-based affine normalization of
//     Pei and Lin (1995), returning the 2×3 transform that maps an image to a
//     canonical, translation/scale/shear-normalized frame.
//   - [FastLineDetector] (bonus) — a line-segment detector built from a Canny
//     edge map and a lightweight Hough accumulator.
//
// # Edge-preserving filters
//
//   - [DTFilter] — the domain-transform filter of Gastal and Oliveira (2011),
//     with all three primitives selectable by [DTMode]: normalized convolution
//     ([DTFilterNC]), interpolated convolution ([DTFilterIC]) and recursive
//     filtering ([DTFilterRF]).
//   - [JointBilateralFilter] — a cross/joint bilateral filter that takes its
//     edge structure from a separate guidance image.
//   - [RollingGuidanceFilter] — the scale-aware rolling-guidance filter of Zhang
//     et al. (2014), which removes small structures while sharpening large edges.
//   - [FastGlobalSmootherFilter] — the weighted-least-squares smoother of Min et
//     al. (2014), solved by separable tridiagonal sweeps.
//   - [AdaptiveManifoldFilter] — a high-dimensional Gaussian filter evaluated on
//     a small tree of adaptive manifolds (Gastal and Oliveira, 2012).
//   - [BilateralTextureFilter] — the patch-based structure/texture separator of
//     Cho et al. (2014).
//   - [WeightedMedianFilter] — a guided weighted-median filter (Zhang et al.,
//     2014) for edge-preserving, impulse-robust smoothing.
//
// # Gradients and analysis
//
//   - [GradientDericheX], [GradientDericheY] — Deriche's recursively implemented
//     optimal edge operator, returning signed [cv.FloatMat] gradients.
//   - [GradientPaillouX], [GradientPaillouY] — Paillou's damped-oscillatory
//     recursive edge operator (alpha, omega).
//   - [CovarianceEstimation] — the sliding-window covariance matrix of an image.
//   - [StructuredEdgeDetectionLite] — a training-free multiscale oriented-gradient
//     edge-probability map (a lightweight stand-in for the trained model).
//   - [EdgeBoxes] — object bounding-box proposals scored from an edge map
//     (Zitnick and Dollár, 2014).
//   - [SuperpixelLSC] — Linear Spectral Clustering superpixels (Li and Chen,
//     2015), a second superpixel method alongside [SuperpixelSLIC].
//
// # Conventions
//
// Single-channel routines ([Thinning], [NiBlackThreshold]) require a
// single-channel input and panic otherwise, mirroring the root package's habit
// of failing fast on shape mismatches. Routines that accept colour input
// ([GuidedFilter] guidance, [SuperpixelSLIC], [FastLineDetector],
// [PeiLinNormalization]) convert internally using [cv.CvtColor]. All functions
// return freshly allocated Mats and never mutate their arguments.
//
// Intensities are handled on the native 8-bit scale ([0,255]); parameters such
// as the guided filter's eps and the diffusion contrast k are therefore in
// those same units unless a function's documentation says otherwise.
//
// # Determinism
//
// Every routine here is fully deterministic: [SuperpixelSLIC] seeds its cluster
// centres from a fixed grid rather than randomly, so repeated runs on identical
// input produce identical labels. There is no hidden global state and no
// dependence on map iteration order.
//
// # Deferred
//
// The following OpenCV ximgproc features are intentionally not implemented and
// are candidates for future work: the matrix-form colour-guidance variant of
// the guided filter (only a scalar/greyscale guidance model is provided here);
// the Guo–Hall thinning variant; the trained random-forest structured-edge
// model (only the training-free [StructuredEdgeDetectionLite] approximation is
// provided); the SEEDS superpixel method; the disparity/ FBS and RIC sparse-to-
// dense post-filters; and superpixel label counts beyond 256 (labels are
// returned in a [cv.Mat], whose samples are 8-bit — see [SuperpixelSLIC] and
// [SuperpixelLSC]).
package ximgproc
