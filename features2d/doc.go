// Package features2d provides 2D feature detection, description and matching on
// top of the standard-library-only OpenCV port github.com/malcolmston/opencv
// (imported here as cv). It mirrors a useful subset of OpenCV's features2d
// module: keypoints, binary descriptors, a detector/descriptor and a brute
// force matcher.
//
// # Overview
//
// A feature pipeline has three stages. Detection finds repeatable interest
// points ([KeyPoint]s) such as corners. Description summarises the image
// neighbourhood around each keypoint as a compact [Descriptors] vector — here a
// binary string of bits packed into bytes. Matching compares two sets of
// descriptors and reports the corresponding pairs as [DMatch]es.
//
// # Detectors and descriptors
//
// [BRIEF] computes a rotation-sensitive binary descriptor by comparing pairs of
// smoothed pixel intensities sampled from a fixed pattern around each keypoint.
// It only describes keypoints; supply them from a detector such as cv.FASTCorners
// or cv.GoodFeaturesToTrack.
//
// [ORB] is a combined detector and descriptor. It detects oriented FAST corners
// (reusing cv.FASTCorners), ranks them by a Harris response, assigns each an
// orientation from the intensity centroid of its patch, and computes a
// steered ("rotated") BRIEF descriptor. Because the descriptor is sampled
// relative to the keypoint and steered by its orientation, ORB descriptors are
// invariant to image translation and in-plane rotation, which makes them
// matchable across such transforms.
//
// [SIFT] is a full scale-invariant detector and descriptor: it builds a
// Difference-of-Gaussians scale-space pyramid, localises extrema to sub-pixel
// and sub-scale accuracy (rejecting low-contrast and edge responses), assigns
// dominant orientations and emits the classic 128-dimensional float gradient
// descriptor. [KAZE] and [AKAZE] work in a nonlinear diffusion scale space that
// preserves edges across scales; KAZE emits a 64-dimensional float M-SURF
// descriptor while AKAZE emits a compact binary M-LDB descriptor.
//
// # Detector classes
//
// Besides the combined detector/descriptors above, the package provides the
// OpenCV detector classes [FastFeatureDetector] and [AgastFeatureDetector]
// (corner detectors), [GFTTDetector] (Shi–Tomasi good-features-to-track) and
// [SimpleBlobDetector] (multi-threshold blob detection via connected
// components). They all satisfy the [Detector] interface. [KeyPointsFilter]
// thins keypoint lists (RetainBest, RemoveDuplicated, RunByImageBorder,
// RunByKeypointSize).
//
// # Matching
//
// [BFMatcher] is a brute-force matcher: every query descriptor is compared
// against every train descriptor. Binary descriptors use the Hamming distance
// ([NormHamming]); float descriptors use the Euclidean distance ([NormL2]).
// [BFMatcher.Match] returns the single best train match per query;
// [BFMatcher.KnnMatch] returns the k nearest. [RatioTest] applies Lowe's ratio
// test to a k=2 KnnMatch result to discard ambiguous matches.
//
// [FlannBasedMatcher] is an approximate nearest-neighbour matcher for float
// descriptors, built on a forest of randomised k-d trees with best-bin-first
// search; it trades a small amount of recall for speed on large descriptor
// sets.
//
// # Bag of visual words
//
// [BOWKMeansTrainer] clusters float descriptors into a visual vocabulary with
// k-means, and [BOWImgDescriptorExtractor] turns an image's descriptors into a
// normalised histogram of vocabulary-word occurrences — the standard
// bag-of-words image representation.
//
// # Evaluation
//
// [ComputeRecallPrecisionCurve] builds a recall/precision curve from matches and
// a correctness mask, and [EvaluateFeatureDetector] measures the repeatability
// of a [Detector] across an image pair related by a homography.
//
// # Drawing
//
// [DrawKeypoints] renders detected keypoints (position, size and orientation)
// onto a colour copy of an image, and [DrawMatches] draws two images side by
// side with lines connecting matched keypoints. [DrawKeypointsFlags] adds a
// colour argument and [DrawFlags] control over rich versus plain markers.
//
// # Relationship to OpenCV
//
// The scale-space detectors are genuine implementations of their algorithms but
// are not byte-identical to OpenCV's: SIFT does not up-sample the input by 2×,
// KAZE/AKAZE use a plain explicit diffusion scheme rather than OpenCV's Fast
// Explicit Diffusion, [AgastFeatureDetector] reads the OAST 9_16 mask directly
// instead of via OpenCV's precompiled decision tree, and [FlannBasedMatcher] is
// approximate by design. Each type's documentation notes where its keypoint set
// or descriptor differs from OpenCV's. Everything else — determinism, the x/y
// coordinate convention and the binary/float descriptor split — matches the
// rest of the package.
//
// # Conventions and determinism
//
// Coordinates follow the cv convention: x is the column and y is the row, with
// the origin at the top-left. All functions in this package are deterministic:
// the BRIEF sampling pattern is generated once from a fixed seed, and detection,
// description and matching perform no randomised or concurrent work, so the same
// input always yields byte-identical output.
package features2d
