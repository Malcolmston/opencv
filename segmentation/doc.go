// Package segmentation provides region-based image segmentation built on the
// stdlib-only OpenCV port github.com/malcolmston/opencv (imported here as cv).
//
// The package groups four classic segmentation techniques, each mirroring the
// behaviour of the equivalent cv2 routine while depending only on the root cv
// package and the Go standard library:
//
//   - [FloodFill] grows a connected region from a seed point, adding pixels
//     whose colour lies within a tolerance of an already-filled neighbour
//     (OpenCV's floating-range flood fill). It reports the pixel count and the
//     bounding rectangle of the filled region and, like cv2.floodFill, mutates
//     the input image in place.
//   - [Watershed] performs marker-controlled watershed segmentation using
//     Meyer's priority-flooding algorithm. Basins grow outward from labelled
//     markers in order of increasing image-gradient magnitude, driven by a
//     min-heap (container/heap). Pixels where two basins meet form the
//     watershed line.
//   - [GrabCut] separates foreground from background with an iterative
//     Gaussian-mixture colour model. Each iteration re-fits per-class GMMs and
//     relabels the soft-labelled pixels. The global min-cut of the classic
//     GrabCut energy is approximated by iterated conditional modes (ICM): a
//     per-pixel MAP data term combined with a Potts smoothness term over the
//     4-neighbourhood. See [GrabCut] for the details of this approximation.
//   - [MeanShiftFiltering] and [PyrMeanShiftFiltering] perform edge-preserving
//     mean-shift smoothing in the joint spatial-range domain, collapsing each
//     pixel onto the mode of its local colour distribution.
//
// # Conventions
//
// Coordinates follow the root package: a [cv.Point] carries x (column) and y
// (row) with the origin at the top-left, and a [cv.Scalar] holds up to four
// colour components interpreted as (R, G, B) for three-channel images. A
// [cv.Mat] is a dense row-major matrix of 8-bit samples.
//
// # Label encoding
//
// Because [cv.Mat] stores unsigned 8-bit samples it cannot hold OpenCV's signed
// CV_32S marker labels. [Watershed] therefore encodes labels in [1, 254] and
// marks watershed-line pixels with the sentinel [WatershedMarker] (255) rather
// than OpenCV's -1. [GrabCut] uses the standard GrabCut mask codes [GcBgd],
// [GcFgd], [GcPrBgd] and [GcPrFgd]; a pixel is foreground when its low bit is
// set, so mask&1 recovers the binary segmentation exactly as in cv2.
//
// All functions are deterministic: given the same inputs they produce identical
// output, with no dependence on map iteration order or randomised seeding.
package segmentation
