// Package xfeatures2d provides additional ("contrib") 2D feature detectors and
// descriptors on top of the standard-library-only OpenCV port
// github.com/malcolmston/opencv (imported here as cv). It mirrors a useful
// subset of OpenCV's opencv_contrib xfeatures2d module and complements the
// detectors in the root package (cv.FASTCorners, cv.GoodFeaturesToTrack,
// cv.CornerHarris).
//
// The package is self-contained: it depends only on the Go standard library and
// the root cv package. It deliberately does not import any sibling cv/*
// subpackage, so the keypoint and descriptor types used here are declared
// locally (see [KeyPoint]).
//
// # Overview
//
// A feature pipeline has two stages that this package covers. Detection finds
// repeatable interest points ([KeyPoint]s) — corners ([AGAST], [GFTTDetector]),
// blobs ([SimpleBlobDetector]), center–surround extrema ([StarDetector]) or
// scale-adapted corners ([HarrisLaplace]). Description summarises the image
// neighbourhood around each keypoint as a compact binary string ([BRISK]).
//
// # Detectors
//
// [SimpleBlobDetector] finds blobs by binarising the image at a sweep of
// thresholds, extracting connected components with cv.FindContours, filtering
// each candidate by area, circularity, convexity and inertia, and merging the
// surviving centers that recur across thresholds. It is the port of OpenCV's
// cv::SimpleBlobDetector.
//
// [AGAST] is an adaptive-threshold FAST corner detector. Rather than testing the
// Bresenham ring against a single fixed threshold, it computes, for every pixel,
// the largest threshold at which the pixel still passes the segment test (its
// AGAST score) and keeps the pixels whose score exceeds the requested threshold.
//
// [StarDetector] (also called CenSurE) approximates the scale-normalised
// Laplacian with bi-level center–surround box filters evaluated in constant time
// through an integral image, and reports scale-space extrema of the response.
//
// [GFTTDetector] wraps cv.GoodFeaturesToTrack (Shi–Tomasi corners) and returns
// the corners as [KeyPoint]s, optionally annotating each with its Harris
// response.
//
// [HarrisLaplace] detects Harris corners across a Gaussian scale space and keeps
// those that are also extrema of the scale-normalised Laplacian, yielding
// scale-adapted keypoints.
//
// # Descriptor
//
// [BRISK] computes a rotation-invariant binary descriptor. It samples image
// intensities on a set of concentric rings, Gaussian-smoothing each sample by a
// radius-dependent amount. Long-distance sample pairs estimate the keypoint
// orientation; the pattern is then rotated by that orientation and short-distance
// pairs are compared to produce the descriptor bits. Two descriptors are compared
// with the Hamming distance ([HammingDistance]).
//
// # Determinism
//
// Every function in this package is deterministic: identical input produces
// identical output, with no randomness or reliance on map iteration order. This
// makes the detectors and descriptors suitable for reproducible tests.
//
// # Deferred
//
// The following opencv_contrib features are intentionally left out of this port:
// SURF and SIFT (patent/complexity), FREAK, DAISY, LATCH, LUCID and VGG
// descriptors, the affine-covariant detectors, and the learned BoostDesc/PCTSignatures
// descriptors. The BRISK and AGAST implementations use a straightforward
// self-contained sampling/scoring scheme rather than OpenCV's precompiled
// decision trees, so their exact keypoint sets differ from OpenCV while following
// the same principles.
package xfeatures2d
