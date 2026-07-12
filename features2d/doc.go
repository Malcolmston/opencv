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
// # Matching
//
// [BFMatcher] is a brute-force matcher: every query descriptor is compared
// against every train descriptor. Binary descriptors use the Hamming distance
// ([NormHamming]); float descriptors use the Euclidean distance ([NormL2]).
// [BFMatcher.Match] returns the single best train match per query;
// [BFMatcher.KnnMatch] returns the k nearest. [RatioTest] applies Lowe's ratio
// test to a k=2 KnnMatch result to discard ambiguous matches.
//
// # Drawing
//
// [DrawKeypoints] renders detected keypoints (position, size and orientation)
// onto a colour copy of an image, and [DrawMatches] draws two images side by
// side with lines connecting matched keypoints.
//
// # Conventions and determinism
//
// Coordinates follow the cv convention: x is the column and y is the row, with
// the origin at the top-left. All functions in this package are deterministic:
// the BRIEF sampling pattern is generated once from a fixed seed, and detection,
// description and matching perform no randomised or concurrent work, so the same
// input always yields byte-identical output.
package features2d
