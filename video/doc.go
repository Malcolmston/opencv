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
//     a tracked object.
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
// The following OpenCV video features are intentionally not implemented:
// background subtraction (MOG2/KNN), the TVL1 and dual-TVL1 dense flow solvers,
// the DIS optical-flow algorithm, higher-order motion models (affine/homography
// tracking), the extended and unscented Kalman variants, and the CamShift /
// meanShift trackers. [CalcOpticalFlowFarneback] approximates the dense flow
// with integer block matching rather than the true polynomial expansion, so it
// yields integer-valued displacements and is intended for small motions.
package video
