// Package features3 provides classic feature detection and description
// primitives on top of the standard-library-only OpenCV port
// github.com/malcolmston/opencv (imported here as cv). It complements the
// module's features2d package with a self-contained collection of corner and
// interest-point operators, binary descriptors, non-linear region detectors and
// scale-space blob detectors, all written against the Go standard library with
// no third-party dependencies, no cgo and no GPU code.
//
// # Images
//
// Every operator consumes the module's central image type, cv.Mat. One-channel
// (grayscale) and three-channel (RGB) images are accepted; three-channel input
// is converted to grayscale with the BT.601 luma weights before processing.
// Locations are reported with cv.Point (integer) and cv.Point2f (sub-pixel), and
// dense response images use cv.FloatMat, matching the rest of the module.
//
// # Corners
//
// [HarrisResponse] and [CornerMinEigenVal] build the windowed structure tensor
// from Sobel gradients and return, respectively, the Harris cornerness
// R = det(M) - k*trace(M)^2 and the Shi–Tomasi smaller-eigenvalue measure.
// [HarrisCorners], [ShiTomasiCorners] and [GoodFeaturesToTrack] threshold and
// non-maximum-suppress those responses into ranked [KeyPoint] lists.
// [CornerSubPix] refines integer corners to sub-pixel accuracy by the classic
// gradient-orthogonality iteration.
//
// # FAST, AGAST and SUSAN
//
// [FASTKeyPoints] implements the FAST-9 accelerated segment test on the radius-3
// Bresenham circle; [AGASTKeyPoints] applies the adaptive generic accelerated
// segment test with the same circle. [SUSANResponse] and [SUSANCorners]
// implement the SUSAN (Smallest Univalue Segment Assimilating Nucleus) corner
// detector. All three support 3×3 non-maximum suppression via [KeypointNMS].
//
// # Descriptors
//
// [CensusTransform3x3], [CensusTransform5x5] and [CensusTransform] compute the
// census (local binary) transform used in stereo matching, comparable with the
// Hamming distance. [ComputeBRIEF] and [ComputeORB] produce bit-packed BRIEF and
// steered-BRIEF (ORB) binary descriptors around supplied keypoints, matchable
// with [MatchBinaryDescriptors].
//
// # Regions and blobs
//
// [MSERRegions] extracts Maximally Stable Extremal Regions by a component-tree
// sweep over intensity thresholds. [LoGBlobs], [DoGBlobs] and [DoHBlobs] detect
// bright/dark blobs across a Gaussian scale space using the scale-normalised
// Laplacian, Difference-of-Gaussians and determinant-of-Hessian operators.
//
// All routines are deterministic: given the same input they always return the
// same output, with no randomness beyond fixed, seeded descriptor patterns.
package features3
