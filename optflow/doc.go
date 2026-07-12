// Package optflow implements extended and dense optical-flow algorithms on top
// of the root cv package, mirroring a useful subset of OpenCV's contrib optflow
// module.
//
// The package is written entirely against the Go standard library and the root
// module github.com/malcolmston/opencv (imported as cv). It deliberately does
// not depend on any sibling subpackage — in particular it is distinct from and
// self-contained relative to cv/video, which offers only sparse pyramidal
// Lucas-Kanade and a block-matching Farneback stand-in. Everything optflow needs
// (grayscale conversion, gradients, Gaussian pyramids, sub-pixel sampling and a
// small local Lucas-Kanade solver) is reimplemented locally so the module has no
// hidden coupling.
//
// # What it provides
//
//   - [FlowField] — a dense two-channel float64 motion field storing an
//     interleaved (u, v) displacement per pixel, with [FlowField.At],
//     [FlowField.MeanFlow], [FlowField.Magnitude] and [FlowField.MaxMagnitude].
//   - [CalcOpticalFlowDenseHS] — Horn-Schunck variational dense flow. Couples
//     brightness constancy with a global smoothness prior and solves the
//     Euler-Lagrange equations by Jacobi iteration. Best for small, smooth
//     motions.
//   - [CalcOpticalFlowDIS] — a simplified Dense Inverse Search: coarse-to-fine
//     patch matching on a Gaussian pyramid. Recovers larger displacements than a
//     single-scale search by inheriting and refining estimates level by level.
//   - [CalcOpticalFlowSparseToDense] — tracks sparse seed points with local
//     Lucas-Kanade and interpolates the matches to a dense field using
//     edge-aware (bilateral) weighting that keeps motion boundaries crisp.
//   - [CalcOpticalFlowDenseTVL1] and [DualTVL1OpticalFlow] — the duality-based
//     TV-L1 method of Zach, Pock & Bischof: an L1 (robust) brightness-constancy
//     data term with total-variation regularisation, minimised by an alternating
//     primal-dual scheme coarse-to-fine with per-level warping.
//   - [DeepFlow] — a lightweight DeepFlow: a dense integer descriptor-matching
//     field seeds a coarse-to-fine variational refinement (matching + data +
//     smoothness), recovering large displacements a differential solver cannot.
//   - [CalcOpticalFlowPCAFlow] — PCAFlow: sparse Lucas-Kanade matches are fitted
//     in a low-dimensional cosine (PCA-prior) subspace by ridge least squares,
//     giving an inherently smooth global-motion field.
//   - [CalcOpticalFlowSparseRLOF] and [CalcOpticalFlowDenseRLOF] — Robust Local
//     Optical Flow: illumination-robust, Huber-reweighted Lucas-Kanade, densified
//     by edge-aware interpolation.
//   - [CalcOpticalFlowSimpleFlow] — SimpleFlow: a non-iterative, bilaterally
//     weighted candidate-flow search with softmin sub-pixel estimation.
//   - [InterpolateFlow] and [InterpolateFlowGuided] — scattered-data
//     densification of sparse flow samples, geometric or edge-aware.
//   - [ReadOpticalFlow] / [WriteOpticalFlow] (and the io variants [ReadFlow] /
//     [WriteFlow]) — the Middlebury .flo binary interchange format.
//   - [EndpointError] / [AverageEndpointError] (AEE), [AngularError] /
//     [AverageAngularError] and [EndpointErrorStats] — the standard optical-flow
//     accuracy metrics against a ground-truth field.
//   - [FlowToColor] — renders a flow field as an RGB image using the Middlebury
//     colour-wheel convention (hue = direction, saturation = magnitude).
//   - [WarpByFlow] — warps an image by a flow field (inverse remap), so that
//     WarpByFlow(prev, flow(prev→next)) approximates next.
//
// # Coordinate and intensity conventions
//
// Points follow the image convention used throughout the root package: X is the
// column (horizontal) and Y is the row (vertical). A flow component u is a
// horizontal displacement (positive → right) and v is vertical (positive →
// down). Multi-channel inputs are converted to grayscale with the same luma
// weights as cv.CvtColor (0.299 R + 0.587 G + 0.114 B) before any motion
// analysis. Sub-pixel access uses bilinear interpolation with edge replication
// for out-of-range coordinates.
//
// # Determinism
//
// Every algorithm here is fully deterministic: identical inputs yield
// bit-identical outputs. No randomness, no goroutine scheduling and no
// map-iteration order affects any result. Ties in the discrete searches are
// broken by fixed rules (smaller displacement magnitude, then row-major order).
//
// # Scope notes
//
// The dense TV-L1 solver is a faithful implementation of the primal-dual
// duality-based method. [DeepFlow], [CalcOpticalFlowPCAFlow], the RLOF trackers
// and [CalcOpticalFlowSimpleFlow] are self-contained ports that capture the
// defining mechanism of each algorithm (descriptor matching + variational
// refinement, a learned-basis subspace fit, robust reweighted tracking and
// bilateral candidate search respectively) rather than bit-exact reproductions
// of the OpenCV contrib code and its trained models. [CalcOpticalFlowDIS]
// remains a lightweight approximation of the DIS family rather than a faithful
// reimplementation of the full gradient-based inverse-search operator.
package optflow
