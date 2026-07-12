// Package aruco provides square fiducial marker generation and detection on top
// of the standard-library-only OpenCV port github.com/malcolmston/opencv
// (imported here as cv). It mirrors a useful subset of OpenCV's aruco module:
// predefined marker dictionaries, marker image synthesis, robust detection from
// a natural image, drawing of detections, planar marker boards, ChArUco boards
// and diamonds, tunable detection, and pose estimation for single markers and
// whole boards (with optional lens-distortion handling) plus ChArUco camera
// calibration.
//
// # Markers and dictionaries
//
// An ArUco marker is a square grid of black and white cells surrounded by a
// solid black border (a one-cell quiet ring). The inner grid encodes an
// identifier: a [Dictionary] with N-by-N inner cells is called a "N x N"
// dictionary. This package ships predefined dictionaries [Dict4x4] and
// [Dict5x5] (via [GetPredefinedDictionary]) and the larger 6x6 families
// [Dict6x6] and [Dict6x6Small] (via [GetPredefinedDictionary6x6]); each holds a
// deterministically generated family of markers whose bit patterns are mutually
// well separated in Hamming distance, both from one another and from their own
// 90/180/270-degree rotations. That separation is what lets detection recover an
// identifier, and the marker's orientation, from a noisy reading. Build a bespoke
// family of any size with [GenerateCustomDictionary].
//
// # Generating markers
//
// [GenerateMarker] renders a marker of a chosen identifier to a single-channel
// [cv.Mat] at a requested pixel size, ready to print or paste into a scene.
//
// # Detecting markers
//
// [DetectMarkers] runs the classic ArUco pipeline against an image:
//
//  1. convert to grayscale and adaptively threshold it,
//  2. trace contours and keep convex four-vertex quadrilaterals of sufficient
//     area (via cv.FindContours and cv.ApproxPolyDP),
//  3. perspectively unwarp each candidate to a canonical square (via
//     cv.GetPerspectiveTransform and cv.WarpPerspective),
//  4. Otsu-threshold the square and read its cell grid,
//  5. match the grid against the dictionary under all four rotations within a
//     Hamming tolerance, recovering the identifier and corner order.
//
// The returned corners are ordered so that the first corner is always the
// marker's own top-left cell, which makes the ordering invariant to the way the
// marker happens to be rotated in the image. [DrawDetectedMarkers] overlays the
// detections on a colour copy of the image.
//
// [DetectMarkersWithParams] is a configurable variant driven by
// [DetectorParameters]: it sweeps a range of adaptive-threshold window sizes,
// bounds candidates by perimeter and corner spacing, tunes the polygon
// approximation, and can refine corners to subpixel accuracy with [CornerSubPix].
// Obtain defaults from [DefaultDetectorParameters].
//
// # Boards
//
// A [GridBoard] is a planar array of markers built with [NewGridBoard] and
// rendered with [DrawPlanarBoard]; because it presents many markers at once, its
// pose is far more stable than a single marker's. [RefineDetectedMarkers] uses a
// board's known layout to rescue markers the first detection pass missed.
//
// A [CharucoBoard] ([NewCharucoBoard], [DrawCharucoBoard]) interleaves markers
// with a chessboard so that [InterpolateCornersCharuco] can localise the
// chessboard corners to subpixel accuracy. A ChArUco "diamond"
// ([DrawCharucoDiamond], [DetectCharucoDiamond]) is a compact four-marker target
// whose ids form a signature.
//
// # Pose and calibration
//
// [EstimatePoseSingleMarkers] recovers an approximate rotation and translation
// for each detected marker from a planar homography and a pinhole camera matrix;
// [EstimatePoseSingleMarkersWithDistortion] first removes lens distortion with
// [UndistortImagePoints]. [EstimatePoseBoard] and [EstimatePoseCharucoBoard]
// fuse every visible marker or chessboard corner into one robust board pose, and
// [CalibrateCameraCharuco] estimates a camera's focal length from several tilted
// views of a ChArUco board by Zhang's planar method. These closed-form
// estimators do not perform non-linear (Levenberg-Marquardt) refinement.
//
// # Conventions and determinism
//
// Coordinates follow the cv convention: x is the column and y is the row, with
// the origin at the top-left. Cell bits use 1 for white and 0 for black. Every
// function in this package is deterministic: dictionaries are generated from a
// fixed seed and no function performs randomised or concurrent work, so the
// same input always yields the same output.
package aruco
