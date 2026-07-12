// Package face is a from-scratch, standard-library-only port of a useful subset
// of OpenCV's contrib face module: classic (non-neural) face recognition. It
// implements the three canonical recognizers — Eigenfaces, Fisherfaces and
// Local Binary Pattern Histograms — plus the underlying Local Binary Pattern
// operator, using only the Go standard library and the root
// [github.com/malcolmston/opencv] package. There is no cgo and there are no
// third-party dependencies.
//
// Beyond recognition the package now also provides a self-contained,
// integral-image Haar face *detector* ([GetFacesHAAR]), a trainable facial
// *landmark* localiser in the FacemarkLBF spirit ([FacemarkLBF]), a
// biologically-inspired feature descriptor ([BIF]) and a Minimum Average
// Correlation Energy filter ([MACE]) — so a full pipeline (detect, align by
// landmarks, describe, verify or identify) can be built without leaving this
// package. The objdetect subpackage still hosts a general Haar cascade
// classifier for production detection.
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
// The extended family adds [LBPCircular], which samples any number of neighbours
// (up to eight) on a circle of arbitrary radius with bilinear interpolation, and
// [LBPUniformRotInvariant], the rotation-invariant uniform ("riu2") operator
// whose labels (0–9) are unchanged by in-plane rotation of the texture.
//
// # Persistence, thresholds and richer prediction
//
// Every recognizer can be serialised to and from an [io.Writer]/[io.Reader] with
// [encoding/gob] via its Save/Load methods; a round trip reproduces predictions
// exactly. SetThreshold/GetThreshold set a maximum acceptable match distance,
// beyond which the threshold-aware PredictThreshold methods return [Unknown]
// rather than a spurious label. PredictCollect exposes the full, distance-sorted
// ranking of training samples behind a plain Predict, for k-nearest-neighbour
// voting or confidence analysis. The Eigenfaces and Fisherfaces subspaces are
// exposed as vectors and as renderable images (EigenVectors, MeanFace,
// EigenFaceImage, DiscriminantAxes, FisherFaceImage).
//
// # Detection, landmarks, descriptors and correlation filters
//
//   - [GetFacesHAAR] detects upright, face-like regions with a fixed set of
//     Haar-like features evaluated in O(1) over an integral image, across scales
//     and positions, with non-maximum suppression.
//
//   - [FacemarkLBF] localises facial landmarks inside a face rectangle using a
//     Supervised Descent cascade of ridge regressors over shape-indexed local
//     features, converging from a learned mean shape toward the true landmarks.
//
//   - [BIF] computes Biologically Inspired Features: a Gabor filter bank pooled
//     across scale bands and a spatial grid into a compact, contrast-normalised
//     descriptor.
//
//   - [MACE] synthesises a Minimum Average Correlation Energy filter from one
//     subject's images and verifies a query by the peak-to-sidelobe ratio of its
//     correlation output, using a from-scratch 2D discrete Fourier transform.
//
// # Determinism
//
// Nothing in this package uses randomness: training and prediction (including
// landmark fitting, detection, BIF and MACE) are fully deterministic functions
// of their inputs, so repeated runs produce identical results. The only global
// state is an internal side table that records each recognizer's optional
// recognition threshold, keyed by the recognizer, and it does not affect
// determinism.
//
// # Relationship to the root package
//
// All image input and output uses the root [cv.Mat] type; colour reduction
// reuses the root's BT.601 luma weights and resampling reuses [cv.Resize]. The
// package deliberately does not import the sibling cv/* subpackages.
//
// # Deferred
//
// The following members of OpenCV's face module remain out of scope: the
// active-appearance and Kazemi landmark models (FacemarkAAM, FacemarkKazemi;
// the FacemarkLBF-style regression cascade is provided by [FacemarkLBF]) and
// deep face embeddings / DNN-based recognition. The general trained Haar cascade
// classifier remains in objdetect; [GetFacesHAAR] here is a self-contained,
// model-free detector rather than a cascade loader.
package face
