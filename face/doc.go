// Package face is a from-scratch, standard-library-only port of a useful subset
// of OpenCV's contrib face module: classic (non-neural) face recognition. It
// implements the three canonical recognizers — Eigenfaces, Fisherfaces and
// Local Binary Pattern Histograms — plus the underlying Local Binary Pattern
// operator, using only the Go standard library and the root
// [github.com/malcolmston/opencv] package. There is no cgo and there are no
// third-party dependencies.
//
// Face *detection* (finding faces in an image) is a separate concern and lives
// in the objdetect subpackage as a Haar cascade classifier; this package
// assumes you already have cropped face images and focuses on identifying whose
// face each one is.
//
// # The recognizer interface
//
// Every model implements [FaceRecognizer]:
//
//	Train(images []*cv.Mat, labels []int)
//	Predict(img *cv.Mat) (label int, confidence float64)
//
// Train fits the model to labelled faces; Predict returns the best-matching
// label and a confidence score. Following OpenCV, the confidence is a distance
// in the model's feature space, so lower is better and an exact match scores 0.
// Malformed input (no images, mismatched label count, a nil/empty image, or
// Predict before Train) panics rather than returning an error, matching the
// root package's convention for programmer error.
//
// # Recognizers
//
//   - [EigenFaceRecognizer] — Eigenfaces. Flattens each face, mean-centres the
//     set, and runs a principal-component analysis (PCA); faces are projected
//     onto the leading eigenvectors ("eigenfaces") and matched by nearest
//     neighbour. The PCA is computed from scratch: the mean-centred data's Gram
//     matrix is eigendecomposed with a cyclic Jacobi solver (see the internal
//     linear-algebra kernels) and mapped back to covariance eigenvectors, the
//     Turk–Pentland small-matrix trick. Keep the top K components with
//     [NewEigenFaceRecognizer]; fewer components give a coarser, lower-rank
//     reconstruction, which [EigenFaceRecognizer.Reconstruct] exposes directly.
//
//   - [FisherFaceRecognizer] — Fisherfaces. Reduces dimensionality with PCA
//     (to N−C dimensions for N images and C classes) and then applies a linear
//     discriminant analysis that maximises between-class scatter over
//     within-class scatter, yielding up to C−1 discriminant axes. The
//     generalized eigenproblem is solved by whitening the within-class scatter
//     and eigendecomposing the transformed between-class scatter. Fisherfaces
//     models class structure explicitly and is more robust to illumination than
//     Eigenfaces.
//
//   - [LBPHFaceRecognizer] — Local Binary Pattern Histograms. Encodes each face
//     as a grid of local texture histograms and matches with the chi-square
//     histogram distance. Because LBP depends only on the ordering of
//     neighbouring pixel intensities, LBPH is inherently robust to monotonic
//     brightness changes and needs no common face geometry.
//
// # Local Binary Patterns
//
// [LBP] computes the basic 3×3, 8-neighbour LBP code image (codes 0–255) and
// [LBPUniform] computes the uniform-pattern variant (labels 0–58, collapsing
// the non-uniform codes). The neighbour weighting is the fixed Ojala et al.
// convention documented on those functions, so codes are reproducible and can
// be checked by hand. Both operators reduce colour input to luma and return a
// Mat two pixels smaller in each dimension, since the pattern is undefined on
// the border.
//
// # Determinism
//
// Nothing in this package uses randomness: training and prediction are fully
// deterministic functions of their inputs, so repeated runs produce identical
// results. There is no hidden global state.
//
// # Relationship to the root package
//
// All image input and output uses the root [cv.Mat] type; colour reduction
// reuses the root's BT.601 luma weights and resampling reuses [cv.Resize]. The
// package deliberately does not import the sibling cv/* subpackages.
//
// # Deferred
//
// The following members of OpenCV's face module are intentionally out of scope:
// facial-landmark models (FacemarkLBF, FacemarkAAM, FacemarkKazemi) and deep
// face embeddings / DNN-based recognition. Face detection remains in objdetect.
package face
