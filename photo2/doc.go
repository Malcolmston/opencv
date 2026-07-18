// Package photo2 is a standard-library-only computational-photography toolkit
// built on the root cv package (github.com/malcolmston/opencv). It provides
// high-dynamic-range (HDR) tonemapping, exposure fusion, edge-preserving
// smoothing, detail enhancement, artistic stylization and colour grading, in
// the spirit of OpenCV's photo and hdr modules but implemented entirely in Go
// with no cgo and no third-party dependencies.
//
// # Image representation
//
// photo2 operates on the root package's two image types and does not introduce
// a parallel one:
//
//   - [cv.Mat] is a dense row-major matrix of 8-bit samples, channel
//     interleaved. It is the standard display-referred (LDR) image. Grayscale
//     uses one channel, RGB three. Three-channel data is treated as RGB
//     (matching Go's image package), not OpenCV's native BGR.
//   - [cv.FloatMat] is a single-channel float64 matrix. A colour image in
//     linear or working space is represented as a slice of [cv.FloatMat], one
//     entry per channel, all sharing the same dimensions. This is the natural
//     container for HDR radiance maps and intermediate results whose range is
//     not confined to [0,255]. Use [ToFloat] and [FromFloat] to convert to and
//     from [cv.Mat].
//
// # Conventions
//
// Float channel values produced by [ToFloat] and consumed by [FromFloat] are
// nominally in [0,1] for display-referred images; HDR radiance channels may
// exceed 1. All routines are deterministic and CPU-only. Functions panic on
// nil, empty or mismatched inputs with a message prefixed "photo2:". Internal
// helpers and helper types carry the unexported "photo2" prefix.
package photo2
