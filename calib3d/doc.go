// Package calib3d is a standard-library-only implementation of the classic
// camera-geometry and multi-view routines from OpenCV's calib3d module, built on
// top of the root cv package (github.com/malcolmston/opencv).
//
// It depends only on the Go standard library and the root cv package: no cgo, no
// third-party code. Where the root package offers a building block — notably
// [github.com/malcolmston/opencv.GetPerspectiveTransform] and
// [github.com/malcolmston/opencv.WarpPerspective] — calib3d reuses it; the
// numerical kernels it needs beyond that (a symmetric Jacobi eigensolver, a 3×3
// SVD, and small matrix helpers) are implemented from scratch.
//
// # Scope
//
// The package covers the standard single- and two-view geometry primitives:
//
//   - [FindHomography] estimates a projective transform between two point sets,
//     exactly from four correspondences (via the root package's perspective
//     solver) or robustly from many noisy correspondences with RANSAC, returning
//     an inlier mask.
//   - [FindFundamentalMat] estimates the fundamental matrix with the normalized
//     eight-point algorithm, enforcing the rank-2 constraint.
//   - [RodriguesToMatrix] and [RodriguesToVector] convert between an axis-angle
//     rotation vector and a 3×3 rotation matrix in both directions.
//   - [ProjectPoints] projects 3D points into the image through a full pinhole
//     model with Brown–Conrady radial and tangential distortion.
//   - [Undistort] removes lens distortion from an image by inverse mapping and
//     bilinear resampling.
//   - [SolvePnPPlanar] recovers camera pose from a planar object via its
//     object-to-image homography, and [SolvePnP] / [SolvePnPRansac] recover pose
//     for general (planar or non-coplanar) objects, the latter robustly against
//     outliers.
//   - [TriangulatePoints] reconstructs 3D points from two views by linear
//     triangulation.
//   - [CalibrateCamera] performs full intrinsic calibration from planar views
//     with Zhang's method (including radial distortion and an RMS reprojection
//     error), and [StereoCalibrate] recovers the relative pose of a stereo rig.
//   - [FindEssentialMat], [DecomposeEssentialMat] and [RecoverPose] handle
//     calibrated two-view geometry; [DecomposeHomographyMat] factors a
//     plane-induced homography into motion and plane normal candidates.
//   - [FindChessboardCorners] and [DrawChessboardCorners] detect and visualise a
//     chessboard calibration target.
//   - [StereoRectify], [GetOptimalNewCameraMatrix] and [InitUndistortRectifyMap]
//     provide the rectification and undistortion-map machinery.
//   - [ComputeCorrespondEpilines] and the [ConvertPointsToHomogeneous] /
//     [ConvertPointsFromHomogeneous] helpers round out the small utilities.
//
// The [CameraMatrix] and [DistCoeffs] helper types package the pinhole
// intrinsics and distortion coefficients and convert to the plain [3][3]float64
// matrices and []float64 slices the functions accept.
//
// # Conventions
//
// Image points use the root package's [github.com/malcolmston/opencv.Point]
// (integer x = column, y = row). Intrinsic matrices are 3×3 row-major
// [3][3]float64. Rotation vectors follow the axis-angle convention: direction is
// the rotation axis, magnitude the angle in radians. Distortion coefficients use
// OpenCV's [K1, K2, P1, P2, K3] ordering, and a nil slice means no distortion.
//
// # Determinism
//
// The RANSAC paths of [FindHomography] and [SolvePnPRansac] seed their own
// pseudo-random generators with fixed constants, so results are fully
// reproducible across runs — a deliberate difference from OpenCV.
//
// # Deferred
//
// Out of scope for this version: iterative non-linear refinement / bundle
// adjustment (the intrinsic and extrinsic estimates are closed-form and linear,
// not Levenberg–Marquardt-polished as in OpenCV), tangential-distortion
// estimation during calibration (only the leading radial terms k1, k2 are fit),
// block-matching and semi-global disparity computation, fisheye and
// omnidirectional camera models, and charuco/circle-grid target detection. These
// build naturally on the primitives here.
package calib3d
