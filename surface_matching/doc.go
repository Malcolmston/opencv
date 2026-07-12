// Package surface_matching is a standard-library-only implementation of 3-D
// surface matching by Point-Pair Features (PPF), mirroring the algorithm of
// OpenCV's contrib surface_matching module (Drost, Ulrich, Navab & Ilic,
// "Model Globally, Match Locally: Efficient and Robust 3D Object Recognition",
// CVPR 2010).
//
// It depends only on the Go standard library and, nominally, the root cv
// package (github.com/malcolmston/opencv). It uses no cgo, no third-party code,
// and no sibling cv/* subpackage: because the module works with 3-D point
// clouds rather than images, every numerical kernel it needs — 3×3 matrix
// algebra, a symmetric Jacobi eigensolver, a 3×3 SVD and quaternion averaging —
// is implemented locally in linalg.go.
//
// # Pipeline
//
// The workflow recognises a rigid model object inside a larger scene cloud and
// recovers the pose (rotation and translation) that places the model in the
// scene:
//
//   - [PointCloud] holds oriented points (coordinates plus unit normals).
//     [PointCloud.ComputeNormals], [ComputeNormalsPC3d] and
//     [ComputeNormalsRadius] estimate normals from local PCA;
//     [PointCloud.VoxelDownsample], [SamplePCUniform] and
//     [SamplePCByQuantization] thin a cloud; [LoadPLY] and [ReadPLY] read ASCII
//     and binary PLY, and [WritePLY]/[WritePLYBinary] export it.
//   - [PPF3DDetector.TrainModel] (or the boundary-robust
//     [PPF3DDetector.TrainModelSpread]) hashes every oriented point pair of the
//     model into a quantised 4-D feature (distance F1 plus three angles F2, F3,
//     F4) together with the pair's local alpha angle.
//   - [PPF3DDetector.Match] votes sampled scene pairs against that table in a
//     Hough scheme, extracts accumulator peaks as candidate [Pose3D] values,
//     and clusters and averages them, returning candidates sorted by vote;
//     [PPF3DDetector.MatchInstances] additionally separates several distinct
//     occurrences of the model in one scene by non-maximum suppression.
//   - [ICP] refines any pose by point-to-point ([ICP.Register],
//     [ICP.RegisterKD]), point-to-plane ([ICP.RegisterPointToPlane]) or
//     coarse-to-fine multi-resolution ([ICP.RegisterMultiScale]) iterative
//     closest point, accelerated by a [KDTree3D], returning the tightened pose
//     and its residual.
//   - [ScorePose] and [VerifyPose] verify a hypothesised pose by its geometric
//     inlier ratio.
//
// # Conventions
//
// Space is right-handed. A [Pose3D] maps a model point p to R·p + T with a
// proper rotation matrix R (det R = +1); normals transform by R alone. Points
// and vectors are [Vec3] (= [3]float64) and matrices [Mat3] (= [3][3]float64,
// row-major).
//
// Distances are in the cloud's own units; the detector has no notion of metres
// versus millimetres. Scale-dependent parameters are given relative to the
// model's bounding-box diagonal ([PointCloud.Diameter]): a relative sampling or
// distance step of 0.05 means "one twentieth of the model diameter". This makes
// the same parameters work across differently scaled data.
//
// Angles are radians throughout. Pair angles (F2, F3, F4) lie in [0, π]; the
// voting alpha lies in [−π, π).
//
// # Determinism
//
// The package is fully deterministic and uses no randomness and no wall-clock
// time. Down-sampling is grid based and emits points in sorted cell order;
// scene reference points are chosen by a fixed stride; hash-table iteration
// order never affects results because votes are commutative. Identical inputs
// always yield identical poses, which is what the package's tests assert.
//
// # Scope and deferred features
//
// Implemented and tested: oriented point clouds with PCA normal estimation
// (k-nearest and fixed-radius variants) and several thinning strategies (voxel,
// uniform stride and quantisation sampling); ASCII and binary PLY reading and
// writing; a static balanced [KDTree3D] (nearest, k-nearest and radius queries)
// backing the ICP correspondence search, normal estimation and pose scoring;
// the full PPF train/match Hough-voting pipeline with model-frame alpha,
// boundary-spread indexing, a flattened bucketed-hash accelerator
// ([PPFHashIndex]) and multi-instance detection; pose clustering, vote-weighted
// averaging and non-maximum suppression; point-to-point, KD-accelerated,
// point-to-plane and multi-resolution ICP with median-based correspondence
// rejection; and geometric pose scoring and verification.
//
// Deliberately deferred (not implemented here):
//
//   - The PLY readers assume the vertex element appears first (as writers in
//     this package and OpenCV emit) and ignore colour, faces and other
//     non-vertex elements.
//   - No learned or hyperparameter auto-tuning of the sampling steps, and no
//     GPU or multi-threaded execution; the code favours clarity and determinism
//     over raw throughput.
package surface_matching
