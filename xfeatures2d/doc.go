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
// [MSDDetector] scores each pixel by its Maximal Self-Dissimilarity — the
// average patch distance to the least dissimilar patches in a surrounding search
// area — and keeps the local maxima, favouring points that stand out from their
// neighbourhood.
//
// [TBMR] reports Tree-Based Morse Regions: extremal regions of the image
// component tree that stay stable across a range of grey levels (the same
// stability criterion as MSER), for both the bright (max-tree) and dark
// (min-tree) trees.
//
// [SURF] is a lightweight Fast-Hessian detector (and descriptor): the
// determinant of a box-filter approximation of the Hessian, evaluated through an
// integral image over a range of scales, whose scale-space maxima are the
// keypoints.
//
// # Descriptors
//
// [BRISK] computes a rotation-invariant binary descriptor. It samples image
// intensities on a set of concentric rings, Gaussian-smoothing each sample by a
// radius-dependent amount. Long-distance sample pairs estimate the keypoint
// orientation; the pattern is then rotated by that orientation and short-distance
// pairs are compared to produce the descriptor bits. Two descriptors are compared
// with the Hamming distance ([HammingDistance]).
//
// [BRIEF] compares the smoothed intensities of a fixed pseudo-random set of
// point pairs (upright, not rotation invariant). [FREAK] uses a retina-like
// pattern of overlapping receptive fields with an orientation estimate for
// rotation invariance. [LATCH] compares three small patches per bit for noise
// robustness. [BEBLID], [TEBLID] and [BoostDesc] are boosted binary descriptors
// whose bits threshold the difference of two box averages (of intensity for
// BEBLID/TEBLID, of gradient magnitude for BoostDesc); they use the same
// weak-learner form as OpenCV but a fixed, untrained (weight-free) arrangement,
// so no learned tables are embedded. All of these are compared with
// [HammingDistance].
//
// [LUCID] records the rank order of the pixels in a blurred neighbourhood, an
// illumination-invariant representation compared with [LUCIDDistance].
//
// [DAISY] and [VGG] are real-valued gradient descriptors — a ring of pooled
// orientation histograms (DAISY) and a log-polar GLOH-like grid (VGG, weight
// free) — compared with the [L2Distance]. [SURF] additionally produces a
// SURF-64 float descriptor from Haar wavelet responses on a 4×4 grid.
//
// [PCTSignatures] summarises an image as a variable-length set of weighted
// Position-Color-Texture clusters ([SignaturePoint]); two signatures are compared
// with the Signature Quadratic Form Distance [SQFD].
//
// # Matching
//
// [MatchBruteForceHamming] and [MatchBruteForceL2] produce nearest-neighbour
// [DMatch] correspondences for binary and float descriptors. [MatchGMS] filters
// such matches with Grid-based Motion Statistics (a match survives when its grid
// neighbourhood is densely supported), and [LOGOSMatcher] filters them with local
// geometric support derived from the keypoints' own scale and orientation.
//
// # Determinism
//
// Every function in this package is deterministic: identical input produces
// identical output. Descriptor sampling patterns that are "random" are generated
// once from a fixed seed, and no algorithm relies on wall-clock time or map
// iteration order. This makes the detectors and descriptors suitable for
// reproducible tests.
//
// # Approximations
//
// Several descriptors here (BEBLID, TEBLID, BoostDesc, VGG) correspond to OpenCV
// classes that ship large tables learned by boosting or discriminative
// projection. To stay self-contained and free of embedded trained data, this
// port keeps each algorithm's geometric/weak-learner form but replaces the
// learned parameters with a fixed pseudo-random, weight-free configuration; the
// per-symbol documentation notes this. Likewise [SURF] follows the SURF
// principles with plain box filters rather than the patented reference code, and
// [TBMR] realises the extremal-region stability criterion without OpenCV's
// incremental max-tree. Exact keypoint and descriptor values therefore differ
// from OpenCV while following the same principles.
//
// # Deferred
//
// SIFT and the affine-covariant detector adapters (AffineFeature2D) remain out of
// scope, as does the trained (non-weight-free) form of the learned descriptors.
package xfeatures2d
