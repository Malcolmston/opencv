// Package linedescriptor is a from-scratch, standard-library-only port of a
// useful subset of OpenCV's contrib module line_descriptor. It is built
// entirely on top of the root package github.com/malcolmston/opencv (imported
// here as cv) and the Go standard library: it uses no cgo, no third-party code
// and it does not import any of the sibling cv/* subpackages.
//
// The module detects straight line segments in an image, describes each
// segment with a compact binary code so that segments can be recognised across
// images, and matches those codes. Every routine consumes the central
// [cv.Mat] type (row-major, channel-interleaved, 8-bit samples), so results
// compose directly with the filters, gradients and colour conversions in the
// root package.
//
// # Contents
//
//   - [KeyLine] — a detected line segment: its two endpoints, orientation,
//     length and detector response, mirroring OpenCV's cv::line_descriptor::KeyLine.
//   - [LSDDetector] — a Line Segment Detector. It follows the design of
//     Grompone von Gioi et al.'s LSD: per-pixel image gradients are turned into
//     "level-line" orientations, connected pixels that share an orientation are
//     grown into line-support regions, each region is approximated by a
//     rectangle whose principal axis is the segment, and a validation step
//     rejects regions that are not thin and dense enough to be a genuine line.
//     See [LSDDetector.Detect] for the full description of the algorithm and
//     its simplifications.
//   - [BinaryDescriptor] — an LBD-style ("Line Band Descriptor") descriptor.
//     For each line it builds a band-shaped support region running along the
//     segment, splits it into parallel bands, gathers per-band gradient
//     statistics and turns them into a fixed-length binary code that is
//     invariant to translation of the segment. See [BinaryDescriptor.Compute].
//   - [BinaryDescriptorMatcher] — a brute-force Hamming matcher over the binary
//     codes, offering both a best-match ([BinaryDescriptorMatcher.Match]) and a
//     k-nearest-neighbour ([BinaryDescriptorMatcher.KnnMatch]) query.
//   - [DrawKeylines] — renders detected segments onto an image for
//     visualisation, mirroring cv::line_descriptor::drawKeylines.
//
// # Coordinate and angle conventions
//
// Points use [cv.Point] where X is the column and Y is the row. A segment's
// [KeyLine.Angle] is the direction from its start point to its end point,
// measured with math.Atan2(dy, dx) in radians in the range (-π, π]. Because a
// line has no intrinsic head or tail, callers that compare orientations should
// treat angles as equal modulo π.
//
// # Determinism
//
// Every routine is deterministic: it performs no concurrency, draws no random
// numbers and breaks all ordering ties by index, so the same input always
// yields byte-identical output.
//
// # Deferred features
//
// The following pieces of the upstream module are intentionally not
// implemented and are documented here so callers know what is missing:
//
//   - Multi-octave scale-pyramid LBD. Upstream LSDDetector and BinaryDescriptor
//     build a Gaussian pyramid and detect/describe lines at several scales,
//     tagging each KeyLine with its octave and merging matches across octaves.
//     This port works at a single scale; every [KeyLine.Octave] is 0.
//   - EDLines, the edge-drawing line detector offered as an alternative front
//     end in some builds of the upstream module.
package linedescriptor
