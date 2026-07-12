// Package videostab is a pure-Go port of a working subset of OpenCV's videostab
// module: global-motion estimation, camera-trajectory smoothing and the
// one-pass / two-pass video stabilization pipelines, together with the
// supporting border-inpainting and deblurring stages.
//
// It is written entirely against the Go standard library, the root module
// github.com/malcolmston/opencv (imported as cv) and the sibling subpackage
// github.com/malcolmston/opencv/video (pyramidal Lucas-Kanade tracking and the
// closed-form similarity fit); the companion github.com/malcolmston/opencv/optflow
// package supplies interchangeable dense-flow front-ends. There is no cgo.
//
// # Pipeline overview
//
// Stabilization proceeds in four stages, each represented by a small, testable
// component:
//
//  1. Global motion estimation. [MotionEstimatorRansacL2] and
//     [MotionEstimatorL1] fit a 2-D transform ([MotionModel]:
//     translation, similarity, affine, homography, …) to a set of point
//     correspondences — the former with a RANSAC hypothesis loop plus L2
//     refinement, the latter by iteratively reweighted least squares (an L1
//     fit). [KeypointBasedMotionEstimator] turns two frames into
//     correspondences by detecting corner features and tracking them with
//     optical flow, then delegates to one of those estimators.
//
//  2. Trajectory smoothing. The per-frame motions are composed into a camera
//     trajectory ([GetMotion]) and smoothed. [GaussianMotionFilter] replaces
//     each frame's transform with a Gaussian-weighted average of its
//     neighbours; [LpMotionStabilizer] fits a smooth path by minimising the L1
//     norm of the path's first, second and third derivatives;
//     [MotionStabilizationPipeline] chains several such stabilizers.
//
//  3. Warping and border handling. [OnePassStabilizer] (causal, Gaussian) and
//     [TwoPassStabilizer] (non-causal, whole-trajectory) warp each frame by its
//     stabilization transform. The empty border that warping exposes is filled
//     by an [Inpainter] — [ColorInpainter] (single-frame diffusion),
//     [ColorAverageInpainter] (temporal average) or [MotionInpainter]
//     (motion-compensated) — optionally chained in an [InpaintingPipeline].
//
//  4. Deblurring. [WeightingDeblurer] sharpens each stabilized frame by blending
//     in detail from sharper neighbours, where sharpness is measured by
//     [CalcBlurriness].
//
// # Conventions
//
// Transforms are 3×3 homogeneous matrices ([Motion]) that map a source point to
// its destination. A per-frame motion motions[i] maps frame i to frame i+1;
// [GetMotion] composes these to relate any two frames. Points follow the image
// convention used throughout the root package (X is the column, Y is the row).
//
// # Determinism
//
// RANSAC sampling uses a seeded math/rand source (see
// [MotionEstimatorRansacL2.SetSeed]) so every result is reproducible; no
// function relies on wall-clock time or map iteration order. This makes the
// stabilizer straightforward to test on synthetic, analytically jittered
// sequences.
package videostab
