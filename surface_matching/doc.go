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
//     [PointCloud.ComputeNormals] estimates normals from local PCA and
//     [PointCloud.VoxelDownsample] thins a cloud; [LoadPLY] reads ASCII PLY.
//   - [PPF3DDetector.TrainModel] hashes every oriented point pair of the model
//     into a quantised 4-D feature (distance F1 plus three angles F2, F3, F4)
//     together with the pair's local alpha angle.
//   - [PPF3DDetector.Match] votes sampled scene pairs against that table in a
//     Hough scheme, extracts accumulator peaks as candidate [Pose3D] values,
//     and clusters and averages them, returning candidates sorted by vote.
//   - [ICP] refines any pose by point-to-point iterative closest point,
//     returning the tightened pose and its residual.
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
// Implemented and tested: oriented point clouds with PCA normal estimation and
// voxel down-sampling; ASCII-PLY loading; the full PPF train/match Hough-voting
// pipeline with model-frame alpha; pose clustering and vote-weighted averaging;
// and point-to-point ICP with median-based correspondence rejection.
//
// Deliberately deferred (not implemented here):
//
//   - No spatial acceleration structure. Both PPF matching and ICP nearest-
//     neighbour search are brute force (O(n²)); there is no KD-tree or FLANN.
//     This bounds practical use to the modest clouds typical of unit tests and
//     small objects.
//   - ICP is point-to-point only; the point-to-plane variant that uses normals
//     is not provided.
//   - The PLY reader is ASCII only and ignores colour and any non-vertex
//     elements; binary PLY is unsupported.
//   - No multi-instance non-maximum suppression beyond simple pose clustering,
//     and no learned or hyperparameter auto-tuning of the sampling steps.
package surface_matching
