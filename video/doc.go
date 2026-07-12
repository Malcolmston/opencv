// Package video implements motion-analysis and tracking primitives on top of
// the root cv package, mirroring a useful subset of OpenCV's video module.
//
// The package is written entirely against the Go standard library (only math)
// and the root module github.com/malcolmston/opencv (imported as cv). It does
// not depend on any sibling subpackage: the few small helpers it needs
// (grayscale conversion, bilinear sampling, gradient computation and dense
// linear algebra) are reimplemented locally. It reuses cv.PyrDown to build
// Gaussian pyramids and cv.SobelFloat to obtain spatial image gradients.
//
// # What it provides
//
//   - [CalcOpticalFlowPyrLK] — sparse, pyramidal Lucas-Kanade optical flow that
//     tracks a set of feature points from one frame to the next. It builds a
//     Gaussian image pyramid for each frame and, per point, runs the iterative
//     Lucas-Kanade update coarse-to-fine using the spatial structure tensor
//     assembled from Sobel gradients.
//   - [BuildOpticalFlowPyramid] — constructs the Gaussian pyramid used by the
//     tracker (level 0 is the source image; each higher level halves both
//     dimensions).
//   - [CalcOpticalFlowFarneback] — a dense optical-flow field. This is a
//     deliberately simplified, block-matching approximation of Farneback's
//     polynomial-expansion algorithm; see the function documentation for the
//     limitations. The result is a two-channel float field represented by the
//     local [FlowField] type (the root cv.FloatMat is single-channel, so a
//     small two-channel float type is used here instead).
//   - [KalmanFilter] — a classic linear Kalman filter with the standard
//     Predict / Correct cycle, useful for smoothing and predicting the state of
//     a tracked object. It also supports an external control input through
//     [KalmanFilter.PredictControl] plus validated matrix setters
//     ([KalmanFilter.SetTransitionMatrix] and friends).
//   - [CalcOpticalFlowPyrLKF] — the sub-pixel (floating-point [PointF]) variant
//     of the pyramidal Lucas-Kanade tracker, and [CalcOpticalFlowFarnebackSubpixel],
//     a parabola-refined sub-pixel version of the dense block-matching flow.
//   - [DISOpticalFlow] — dense optical flow by coarse-to-fine patch-based
//     inverse search with confidence-weighted densification (Dense Inverse
//     Search).
//   - [FindTransformECC] — image alignment by Enhanced Correlation Coefficient
//     maximisation for the translation, euclidean, affine and homography motion
//     models ([MotionType]).
//   - [MeanShift] and [CamShift] — mode-seeking trackers over a probability /
//     back-projection image, with CamShift also returning the object's size and
//     orientation as a [cv.RotatedRect].
//   - [TrackerFeaturePyrLK] — a stateful sparse feature tracker built on the
//     pyramidal Lucas-Kanade flow.
//   - [EstimateAffinePartial2D] — closed-form least-squares similarity
//     (scale + rotation + translation) estimation from point correspondences.
//   - [VideoStabilizer] — a causal digital stabilizer combining grid-feature
//     motion estimation, cumulative-trajectory smoothing and compensating warps.
//   - [BackgroundSubtractorMOG2] and [BackgroundSubtractorKNN] — adaptive
//     per-pixel background models producing binary foreground masks.
//
// # Coordinate and intensity conventions
//
// Points follow the image convention used throughout the root package: X is the
// column (horizontal) and Y is the row (vertical). Multi-channel inputs are
// converted to grayscale with the same luma weights as cv.CvtColor
// (0.299 R + 0.587 G + 0.114 B) before any motion analysis. Sub-pixel image
// access uses bilinear interpolation with edge replication for out-of-range
// coordinates.
//
// # Determinism
//
// Every function in this package is deterministic: given the same inputs it
// produces the same outputs, with no reliance on randomness, maps ordering or
// wall-clock time. This makes the algorithms straightforward to test on
// synthetic, analytically shifted images.
//
// # Deferred / out of scope
//
// The following OpenCV video features are intentionally not implemented: the
// TVL1 and dual-TVL1 dense flow solvers, the variational-refinement stage of DIS
// (the patch inverse search and densification are implemented; the final
// variational smoothing pass is not), and the extended and unscented Kalman
// variants. [CalcOpticalFlowFarneback] approximates the dense flow with integer
// block matching rather than the true polynomial expansion, so it yields
// integer-valued displacements and is intended for small motions; use
// [CalcOpticalFlowFarnebackSubpixel] or [DISOpticalFlow] when sub-pixel dense
// flow is required.
package video
