// Package matching2 implements descriptor matching and multiple-view geometry
// on top of the parent cv module, using only the Go standard library.
//
// It complements the feature detectors in the features2d subpackage and the
// calibration routines in calib3d by providing a self-contained, deterministic
// toolkit for the two halves of the classic structure-from-motion pipeline:
//
//   - Descriptor matching. Brute-force matching with the L1, L2 and Hamming
//     norms ([BFMatcher], [HammingMatcher]), k-nearest-neighbour and radius
//     matching, Lowe's ratio test ([RatioTest]), symmetric cross-check
//     ([CrossCheck]) and an approximate FLANN-style kd-tree matcher
//     ([KDTree], [FLANNMatcher]).
//
//   - Two-view geometry. Robust model fitting with RANSAC and least-median-of-
//     squares ([RANSAC], [LMedS]), planar homography estimation
//     ([FindHomographyDLT], [FindHomographyRANSAC]), the normalized eight-point
//     fundamental matrix ([FindFundamentalMat]), essential-matrix estimation
//     and pose recovery ([FindEssentialMat], [DecomposeEssentialMat],
//     [RecoverPose]), epipolar geometry ([EpipolarLine], [SampsonDistance]),
//     linear triangulation ([TriangulatePoint]) and the DLT solution to the
//     perspective-n-point problem ([SolvePnPDLT]).
//
// # Conventions
//
// Two-dimensional image points use the parent module's core.Point2d and
// three-dimensional world points use core.Point3d, so results interoperate with
// the core and calib3d subpackages. Following calib3d, 3×3 and 3×4 matrices are
// plain fixed-size arrays ([3][3]float64, [3][4]float64) rather than a bespoke
// matrix type; the [Mat3Mul], [Mat3Inverse] and related helpers operate on
// them. Descriptors are float rows ([][]float64) compared with a floating-point
// [NormType], or bit-packed binary rows ([][]byte) compared with the Hamming
// distance.
//
// # Determinism
//
// Every routine in this package is deterministic. The random-sample estimators
// ([RANSAC], [LMedS] and the RANSAC-based geometry helpers) draw their samples
// from a caller-seeded or fixed-seed generator, so identical inputs always
// produce identical output. This is a deliberate departure from OpenCV, whose
// RANSAC results vary between runs.
//
// The package depends only on the Go standard library: no cgo, no third-party
// modules and no GPU. All computation is CPU-only and single-precision-free
// (float64 throughout).
package matching2
