// Package stitching assembles overlapping images into a single panorama on top
// of the standard-library-only OpenCV port github.com/malcolmston/opencv
// (imported here as cv). It mirrors a useful subset of OpenCV's stitching module
// with a self-contained, deterministic implementation that depends only on the
// root cv package and the Go standard library.
//
// # Pipeline
//
// [Stitcher.Stitch] runs the classic feature-based panorama pipeline:
//
//  1. Detection. Shi–Tomasi corners are found in each image with
//     cv.GoodFeaturesToTrack.
//  2. Description. Every corner is summarised by a normalised intensity-patch
//     descriptor — a mean-subtracted, L2-normalised window of pixels — so that
//     descriptor distance is invariant to local brightness and contrast.
//  3. Matching. Descriptors of adjacent images are matched by nearest neighbour
//     and filtered with Lowe's ratio test, which rejects ambiguous matches in
//     repetitive texture.
//  4. Estimation. The pairwise homography is fitted with a normalised Direct
//     Linear Transform inside a RANSAC loop, giving a model robust to the
//     outliers that survive matching. See [Stitcher.EstimateTransform].
//  5. Composition. Neighbour homographies are chained into a common reference
//     frame, every image is warped into a shared canvas with
//     cv.WarpPerspective, and the overlaps are blended. See
//     [Stitcher.ComposePanorama].
//
// # Blending
//
// Two blenders are provided. [Feather] performs distance-weighted alpha
// blending, cross-dissolving overlaps with a feather ramp that is highest at
// each image's centre. [MultiBandBlend] performs Laplacian-pyramid multi-band
// blending, which blends low frequencies over a wide transition and high
// frequencies sharply for a more seamless result. Both satisfy the [Blender]
// interface, so a custom blender can be supplied via [Stitcher].Blender.
//
// # Coordinate and transform conventions
//
// Transforms are cv.PerspectiveMatrix values (row-major 3×3 homographies) that
// map a point (x, y) — x is the column, y is the row — from a source image into
// a destination frame. [Stitcher.EstimateTransform](a, b) returns the homography
// that maps image b into image a's frame.
//
// # Determinism
//
// Every stage is deterministic: corner detection and matching are exhaustive and
// order-stable, and the RANSAC sampler is seeded (see [RANSACParams].Seed), so
// the same inputs always produce the same panorama.
//
// # Scope
//
// The package covers planar (homography) stitching with feather and multi-band
// blending. Cylindrical and spherical warping, exposure compensation, seam
// finding by graph cut, and global bundle adjustment are intentionally out of
// scope; images are related pairwise and chained rather than jointly optimised.
package stitching
