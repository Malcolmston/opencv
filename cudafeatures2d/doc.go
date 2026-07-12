// Package cudafeatures2d is a CPU-backed, API-compatible mirror of OpenCV's
// cudafeatures2d module for the standard-library-only OpenCV port
// github.com/malcolmston/opencv (imported here as cv).
//
// # Honesty note
//
// This package contains NO CUDA, NO cgo and NO GPU code. It exists so that code
// written against OpenCV's GPU feature-detection API — [GpuMat] device
// matrices, [Stream] asynchronous streams, and the cuda ORB / FAST / brute-force
// matcher / good-features-to-track classes — can be ported to Go with minimal
// surface changes. Every operation runs synchronously on the CPU. [GpuMat] is a
// thin wrapper around a host *cv.Mat, [Stream] is a no-op, and the detectors and
// matcher delegate to the pure-Go implementations in the root cv package and its
// sibling github.com/malcolmston/opencv/features2d.
//
// Because the work happens on the host, "upload" and "download" are ordinary
// copies, the Async method variants complete before they return, and passing a
// [Stream] has no effect beyond documentation intent. The numerical results are
// identical to the CPU features2d module, not to OpenCV's GPU kernels (which use
// a different FAST/ORB tuning); treat this as a source-compatibility shim, not a
// performance or bit-exactness claim.
//
// # Types
//
// [GpuMat] mirrors cv::cuda::GpuMat: construct one from a host image with
// [NewGpuMat] or an empty one and [GpuMat.Upload], and retrieve results with
// [GpuMat.Download]. [Stream] mirrors cv::cuda::Stream and is inert.
//
// # Detectors and descriptors
//
// [ORB] mirrors cv::cuda::ORB (create with [CreateORB]): it detects oriented
// FAST keypoints, ranks them by Harris response, and computes rotated BRIEF
// descriptors. Descriptors are returned as a device [GpuMat] whose rows are the
// bit-packed descriptors; recover them on the host with [ORB.Convert].
//
// [FastFeatureDetector] mirrors cv::cuda::FastFeatureDetector (create with
// [CreateFastFeatureDetector]) and detects FAST-9 corners.
//
// [CornersDetector] mirrors cv::cuda::CornersDetector; build the Shi–Tomasi
// variant with [CreateGoodFeaturesToTrackDetector]. Its [CornersDetector.Detect]
// writes corner locations into a device [GpuMat].
//
// # Matching
//
// [DescriptorMatcher] mirrors cv::cuda::DescriptorMatcher; build a brute-force
// matcher with [CreateBFMatcher]. It offers [DescriptorMatcher.Match],
// [DescriptorMatcher.KnnMatch] and [DescriptorMatcher.RadiusMatch] over device
// descriptor [GpuMat]s, and [DescriptorMatcher.MatchConvert] to flatten a raw
// knn/radius result into a single []DMatch slice.
//
// # Keypoints and matches
//
// [KeyPoint] and [DMatch] are type aliases for the corresponding features2d
// types, so results interoperate directly with that package.
//
// # Determinism
//
// Like the packages it delegates to, everything here is deterministic: the same
// input always yields byte-identical keypoints, descriptors and matches.
package cudafeatures2d
