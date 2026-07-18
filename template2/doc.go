// Package template2 implements classic template matching on top of the parent
// package's [cv.Mat] image type.
//
// The package is a pure-Go, standard-library-only companion to the core opencv
// module. It never defines its own image type: every routine that consumes an
// image accepts a [cv.Mat], every routine that produces a dense score map
// returns a [cv.FloatMat], and match locations are reported with the parent
// package's coordinate conventions (x is the column, y is the row, origin
// top-left). Results therefore interoperate directly with the rest of the
// library — drawing, I/O, [cv.MinMaxLoc], and so on.
//
// # Similarity measures
//
// [MatchTemplate] slides a template over a source image and returns a dense
// score map under one of the measures selected by [Method]:
//
//   - [MethodSAD] — sum of absolute differences (lower is better).
//   - [MethodSSD] — sum of squared differences (lower is better).
//   - [MethodSSDNormed] — normalised sum of squared differences (lower).
//   - [MethodCrossCorr] — raw cross-correlation (higher is better).
//   - [MethodNCC] — normalised cross-correlation in [-1,1] (higher).
//   - [MethodCorrCoeff] — mean-subtracted covariance (higher).
//   - [MethodZNCC] — zero-mean normalised cross-correlation in [-1,1] (higher).
//
// Convenience wrappers [MatchSAD], [MatchSSD], [MatchNCC], [MatchZNCC] and
// friends select a measure directly. [BestMatch] returns the single strongest
// [Match]; [FindMatches] returns every location that passes a score threshold.
//
// # Beyond a single pass
//
// The package layers the standard extensions on top of the basic pass:
//
//   - Fast normalised matching via integral images: [Integral], [FastNCC] and
//     [FastZNCC] compute window statistics in O(1) per shift.
//   - Sub-pixel peak refinement: [ParabolicPeak], [RefinePeak] and
//     [RefineMatch] fit a quadratic to the score surface around a peak.
//   - Non-maximum suppression: [NonMaxSuppression] and
//     [NonMaxSuppressionDistance] prune overlapping detections.
//   - Multi-scale matching: [BuildPyramid], [Pyramid] and [MatchMultiScale]
//     search across a range of template scales.
//   - Rotation-invariant matching: [RotateTemplate] and
//     [MatchRotationInvariant] search across a range of template rotations.
//
// # Channels
//
// The basic measures operate channel-agnostically: sums run over every sample
// of every channel, matching the parent package's [cv.MatchTemplate]. The
// integral-image fast paths operate on grayscale and convert multi-channel
// input with [ToGrayscale] first.
//
// All routines are deterministic and CPU-only.
package template2
