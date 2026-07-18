// Package stereo2 is a self-contained, standard-library-only toolkit for
// binocular stereo vision and depth reconstruction, built on top of the core
// [github.com/malcolmston/opencv] package.
//
// It provides the classic dense-stereo pipeline end to end:
//
//   - matching cost construction (SAD/SSD/NCC block matching, census-based
//     matching) and the raw per-pixel cost volume ([CostVolume]);
//   - cost aggregation, both local box aggregation ([AggregateBoxFilter]) and
//     the dynamic-programming path aggregation of semi-global matching
//     ([SemiGlobalAggregate], [SGMMatcher]);
//   - disparity extraction with winner-take-all and uniqueness testing;
//   - disparity refinement — left/right consistency checking ([LeftRightCheck]),
//     parabola sub-pixel interpolation ([RefineSubpixelParabola]), speckle
//     removal ([SpeckleFilter]), median smoothing and invalid-pixel filling;
//   - geometry — depth from disparity ([DepthFromDisparity]), reprojection to a
//     metric 3-D point cloud ([PointCloudFromDepth], [ReprojectImageTo3D]) and
//     least-squares / RANSAC plane fitting in the reconstructed cloud
//     ([FitPlane], [FitPlaneRANSAC]).
//
// # Conventions
//
// Rectified input images are single- or multi-channel [github.com/malcolmston/opencv.Mat]
// values; multi-channel images are read as luminance. The library follows the
// standard rectified-stereo geometry in which a scene point seen at column x in
// the left image appears at column x-d in the right image, where d >= 0 is the
// disparity. Consequently the left disparity map is the reference map returned
// by the matchers' Compute methods, and the corresponding right-referenced map
// (needed for [LeftRightCheck]) is produced by the ComputeRight methods.
//
// Disparities and depths are carried as float32 in the dedicated [DisparityMap]
// and [DepthMap] types rather than in a uint8 Mat, so sub-pixel precision and an
// explicit invalid sentinel ([InvalidDisparity], [InvalidDepth]) are preserved.
// Convert to a viewable 8-bit Mat with the ToMat methods.
//
// Everything here is pure Go, deterministic and CPU-only: no cgo, no GPU and no
// third-party dependencies.
package stereo2
