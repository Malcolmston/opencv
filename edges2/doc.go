// Package edges2 is a standard-library-only toolkit of edge, line and circle
// detection built on top of the parent package's
// [github.com/malcolmston/opencv.Mat] image type.
//
// It complements the elementary edge helpers of the root package with a
// self-contained collection of classical feature-detection machinery:
//
//   - Gradient operators: Sobel, Scharr, Prewitt and Roberts cross, each
//     returning a signed [GradientField] from which magnitude and orientation
//     are derived.
//   - Edge detectors: the full Canny pipeline (Gaussian smoothing, gradient,
//     non-maximum suppression and double-threshold hysteresis) exposed both as
//     a one-shot [Canny] and as its composable stages, plus the
//     Marr-Hildreth Laplacian-of-Gaussian zero-crossing detector and a
//     model-free multi-scale [StructuredEdges] strength map.
//   - Line and circle fitting: the standard and probabilistic Hough transforms
//     ([HoughLines], [HoughLinesP]), the gradient Hough circle transform
//     ([HoughCircles]) and a simplified region-growing line-segment detector
//     ([LSD]).
//   - Post-processing: 8-connected [LinkEdges] chaining, gradient orientation
//     histograms and a dense histogram-of-oriented-gradients ([HOG])
//     descriptor.
//
// # Conventions
//
// Every routine consumes single-channel (Channels == 1) [cv.Mat] values and
// panics on multi-channel input. Coordinates follow the parent package's image
// convention: x is the column, y is the row and the origin is at the top-left.
// Border samples are handled by edge replication (BORDER_REPLICATE).
//
// Signed, unbounded responses are carried by the [FloatGrid] helper, which
// converts to and from [cv.Mat] rather than duplicating that type. Detected
// features are returned as plain [Line], [Segment], [Circle], [Point] and
// [EdgeChain] value types.
//
// The implementation is pure Go, uses no cgo and no third-party dependencies,
// is CPU-only and fully deterministic.
package edges2
