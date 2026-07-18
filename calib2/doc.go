// Package calib2 is a pure standard-library implementation of camera
// calibration and multi-view 3D geometry, built on top of the root cv package
// (github.com/malcolmston/opencv).
//
// It depends only on the Go standard library and the root cv package for its
// image and floating-point point types: there is no cgo, no third-party code
// and no GPU dependency. All numerical kernels — a symmetric Jacobi
// eigensolver, Gaussian-elimination linear solves, an RQ decomposition and the
// small fixed-size 3×3 / 3-vector helpers — are implemented from scratch and are
// fully deterministic.
//
// # Scope
//
// The package covers the standard single- and two-view geometry primitives:
//
//   - The pinhole model: [CameraMatrix] packages the intrinsics, [ProjectPoint]
//     and [ProjectPoints] map 3D points into the image through a full pinhole
//     model with Brown–Conrady radial and tangential distortion, and
//     [ProjectionMatrix] / [DecomposeProjectionMatrix] build and factor the 3×4
//     camera matrix P = K·[R|t].
//   - Rotations: [Rodrigues] and [RodriguesInverse] convert between an
//     axis-angle rotation vector and a 3×3 rotation matrix; Euler-angle and
//     unit-quaternion conversions and the elementary axis rotations round out
//     the set.
//   - Distortion: [DistortPoint] / [UndistortPoint] apply and invert the lens
//     model, [InitUndistortRectifyMap] builds resampling maps and
//     [UndistortImage] removes distortion from a [github.com/malcolmston/opencv.Mat].
//   - Calibration: [FindHomography] estimates a planar homography by the direct
//     linear transform, and [CalibrateCamera] performs full intrinsic
//     calibration from planar views with Zhang's method, returning per-view
//     poses, radial distortion and an RMS reprojection error.
//   - Two-view geometry: [Triangulate] / [TriangulatePoints] reconstruct 3D
//     points, [EssentialMatrix] and [FundamentalMatrix] build the epipolar
//     matrices, [DecomposeEssentialMatrix] recovers relative pose candidates and
//     [ComputeCorrespondEpilines] maps points to epipolar lines.
//   - Stereo: [StereoRectify] computes rectifying rotations and the 4×4
//     disparity-to-depth matrix Q, and [ReprojectImageTo3D] / [DepthFromDisparity]
//     turn disparities into metric depth.
//
// The [Matrix] type is a small dense float64 matrix used internally and exposed
// for callers who need general linear algebra alongside the calibration
// routines.
//
// # Conventions
//
// Image points use the root package's
// [github.com/malcolmston/opencv.Point2f] (floating-point x = column, y = row).
// Intrinsic and rotation matrices are 3×3 row-major [3][3]float64. Object points
// and translations are [3]float64. Rotation vectors follow the axis-angle
// convention: direction is the rotation axis, magnitude the angle in radians.
// Distortion coefficients use OpenCV's [K1, K2, P1, P2, K3] ordering, and the
// zero-valued [DistortionCoeffs] describes a distortion-free lens.
//
// # Determinism
//
// Every routine is deterministic: there is no randomness, and the iterative
// solvers (Jacobi eigen-decomposition, undistortion point iteration) run to
// fixed convergence tolerances or iteration caps.
//
// # Deferred
//
// Out of scope for this version: iterative non-linear refinement / bundle
// adjustment (the intrinsic and extrinsic estimates are closed-form and linear,
// not Levenberg–Marquardt-polished), tangential-distortion estimation during
// calibration (only the leading radial terms k1, k2 are fit), block-matching
// disparity computation, and fisheye / omnidirectional camera models.
package calib2
