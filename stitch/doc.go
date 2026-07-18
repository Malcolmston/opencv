// Package stitch implements image stitching and panorama construction on top of
// the core [github.com/malcolmston/opencv] image types, using only the Go
// standard library.
//
// The package covers the full classic stitching pipeline, split across a few
// files so each stage can be used on its own:
//
//   - homography.go — floating-point correspondences ([PointF], [Match]), the
//     projective [Homography] type, a normalized-DLT estimator and a
//     RANSAC-robust estimator for recovering the transform between two images
//     from matched feature points.
//   - features.go — a deterministic Harris corner detector and a
//     normalized-cross-correlation matcher, so a pair of overlapping images can
//     be aligned end to end without any external feature library.
//   - warp.go — cylindrical and spherical projection ([WarpCylindrical],
//     [WarpSpherical]) and projective warping into a shared canvas
//     ([WarpPerspectiveToCanvas]).
//   - canvas.go — integer [Bounds] arithmetic and the geometry needed to place
//     warped images onto a common mosaic canvas.
//   - align.go — bundle-free incremental alignment: composing pairwise
//     homographies into a globally-consistent set of transforms
//     ([GlobalTransforms], [IncrementalAligner]).
//   - seam.go — dynamic-programming seam finding ([FindVerticalSeamDP],
//     [DPSeamFinder]) that routes the cut between two overlapping images through
//     the lowest-difference pixels.
//   - exposure.go — gain-based exposure compensation ([GainCompensator],
//     [EstimateGains]) that equalises brightness across the mosaic.
//   - blend.go — feather and multi-band (Burt–Adelson) blending
//     ([FeatherBlender], [MultiBandBlender]) plus the Gaussian/Laplacian pyramid
//     primitives they rely on.
//
// # Conventions
//
// Images are the core [github.com/malcolmston/opencv.Mat] type (row-major,
// channel-interleaved 8-bit samples); per-pixel weight and cost maps are the
// core FloatMat type. Coordinates follow the image convention: x is the column
// and y is the row, with the origin at the top-left. Floating-point image
// coordinates use [PointF].
//
// A "layer" ([Layer]) is one source image already warped onto the full mosaic
// canvas together with a per-pixel coverage/feather weight; blending and seam
// finding operate on layers so the canvas-placement stage is shared across
// blenders.
//
// Everything here is CPU-only, deterministic (RANSAC takes an explicit seed),
// and free of third-party dependencies.
package stitch
