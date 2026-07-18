// Package segment2 provides image-segmentation algorithms built on the
// standard-library-only OpenCV port github.com/malcolmston/opencv (imported here
// as cv). Every routine consumes and produces the library's native [cv.Mat]
// image type; no competing image representation is introduced.
//
// The package is deterministic and depends only on the Go standard library and
// the root cv package. It groups the classic families of segmentation:
//
//   - Clustering in colour space: [KMeansSegment], [QuantizeColors],
//     [MeanShiftFilter] and [MeanShiftSegment].
//   - Marker-controlled flooding: [Watershed], [WatershedFromMarkers] and the
//     [DistanceTransform] / [GradientMagnitude] helpers used to build markers.
//   - Region-based growing and partitioning: [RegionGrow], [SeededRegionGrow],
//     [SplitAndMerge].
//   - Graph-based segmentation: [Felzenszwalb] minimum-spanning-forest.
//   - Superpixels: [SLIC] and [SLICO].
//   - Foreground extraction by graph cut: [GrabCut], backed by a genuine
//     max-flow / min-cut ([MaxFlow]).
//   - Boundary evolution: [ActiveContour] parametric snakes.
//   - Thresholded connected components: [ConnectedComponents],
//     [ThresholdComponents] and the [ComponentStats] measurements.
//
// # Label maps
//
// Because a single-channel [cv.Mat] can hold at most 256 distinct values, the
// dense segmenters return a [LabelMap] — a flat row-major []int labelling with
// helpers ([LabelMap.Colorize], [LabelMap.BoundaryMask], [LabelMap.RegionSizes],
// [LabelMap.BoundingRects] and others) that turn a labelling back into a
// viewable [cv.Mat] or into measurements.
//
// # Colour convention
//
// Colour distances are computed directly in the channel space of the input
// [cv.Mat]; a three-channel image is treated as an ordinary Euclidean colour
// vector and a single-channel image as intensity. Grayscale conversion for the
// gradient-based routines uses a fixed luminance weighting so results are
// reproducible.
package segment2
