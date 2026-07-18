// Package connected provides connected-component analysis for binary images,
// built entirely on the parent module's [github.com/malcolmston/opencv.Mat]
// type and the Go standard library.
//
// The package covers the classic blob-analysis toolkit found in OpenCV's
// connectedComponents / connectedComponentsWithStats and the scipy.ndimage
// measurement family:
//
//   - Labelling: a two-pass union-find labeller ([Label], [LabelWithStats])
//     that assigns a unique integer to every 4- or 8-connected foreground
//     region and returns a compact, gap-free [Labels] map.
//   - Component statistics: per-component area, bounding box and centroid
//     ([Component], [ComponentStats]), plus convenience selectors such as
//     [LargestComponent], [FilterByArea] and [RemoveSmallComponents].
//   - Blob analysis: richer shape descriptors ([Blob], [AnalyzeBlobs]) with
//     perimeter, circularity, extent and equivalent diameter.
//   - Flood fill: a span-based (scanline) region fill ([FloodFill],
//     [FloodFillMask], [FloodFillTolerance]) that grows a region from a seed.
//   - Topology: interior hole filling ([FillHoles]), hole counting
//     ([CountHoles]) and the [EulerNumber].
//   - Boundaries: inner/outer boundary extraction ([Boundary], [OuterBoundary]),
//     perimeter measurement ([Perimeter]) and Moore boundary tracing
//     ([TraceBoundary]).
//
// # Conventions
//
// Inputs are single-channel [github.com/malcolmston/opencv.Mat] images treated
// as binary: any non-zero sample is foreground, a zero sample is background.
// Outputs produced by this package use 255 for foreground and 0 for background
// so they round-trip cleanly through the parent package's thresholding and
// morphology routines. Coordinates follow the parent package: x is the column,
// y is the row, and the origin is the top-left pixel.
//
// Connectivity is selected with [Conn4] or [Conn8]. Every routine is
// deterministic, CPU-only and free of third-party dependencies.
package connected
