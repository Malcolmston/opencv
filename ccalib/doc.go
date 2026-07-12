// Package ccalib is a standard-library-only port of OpenCV's ccalib module —
// custom calibration patterns and the omnidirectional (fisheye / catadioptric)
// camera model — built on top of the root cv package
// (github.com/malcolmston/opencv).
//
// It depends only on the Go standard library and the root cv package: no cgo, no
// third-party code. The numerical kernels it needs (a symmetric Jacobi
// eigensolver, a 3×3 SVD, Gaussian elimination, a normalised-DLT homography
// solver and a Levenberg–Marquardt optimiser) are implemented from scratch in
// linalg.go.
//
// # Omnidirectional camera model
//
// The [Omnidir] namespace value is the Go rendering of OpenCV's cv::omnidir
// namespace. It implements the unified (Mei / Scaramuzza) sphere model, in which
// a camera-frame point is projected onto the unit sphere, shifted by the mirror
// parameter Xi, projected to the normalized plane, distorted with a
// Brown–Conrady term and mapped to pixels through the intrinsics:
//
//   - [omnidirNS.ProjectPoints] projects 3D points through a full
//     omnidirectional model (rotation, translation, Xi and distortion).
//   - [omnidirNS.Undistort] maps distorted pixels to a rectified pinhole image,
//     and [omnidirNS.UndistortImage] rectifies a whole image to a perspective,
//     cylindrical or longitude–latitude view.
//   - [omnidirNS.InitUndistortRectifyMap] builds the remap tables those routines
//     rely on, for use with [github.com/malcolmston/opencv.Remap].
//   - [omnidirNS.Calibrate] recovers the intrinsics (including Xi) and per-view
//     poses from planar views, seeding each pose with a linear pose-from-rays
//     solver and polishing everything with Levenberg–Marquardt.
//   - [omnidirNS.StereoCalibrate] recovers the relative pose of two
//     omnidirectional cameras, and [omnidirNS.MultiCameraCalibration] resolves a
//     whole rig of cameras and the pattern instances they observe by pose-graph
//     propagation.
//
// The [OmniModel] type bundles the intrinsic parameters, and [Pose] the rigid
// transforms returned by the multi-camera helper.
//
// # Custom and random-dot patterns
//
// [CustomPattern] builds a feature-based target from an arbitrary template image
// and finds it inside a scene, returning 3D↔2D correspondences.
// [RandomPatternGenerator] synthesises a deterministic random-dot target and
// [RandomPatternCornerFinder] detects it in captured views. Both share a
// self-contained feature core (features.go) that matches dots by a rotation-,
// scale- and translation-invariant geometric descriptor and verifies the match
// with a homography, so no appearance-based descriptor library is required.
//
// # Conventions
//
// Object points are [3]float64 world coordinates (planar targets use Z = 0);
// image points are [2]float64 sub-pixel pixel coordinates. Intrinsic matrices
// are 3×3 row-major [3][3]float64. Rotation vectors follow the axis-angle
// convention (direction is the axis, magnitude the angle in radians). Omni
// distortion coefficients use the [K1, K2, P1, P2] ordering; a nil slice means
// no distortion.
//
// # Determinism
//
// [RandomPatternGenerator] is seeded explicitly, and every solver is
// deterministic (no randomised RANSAC), so results are fully reproducible.
package ccalib
