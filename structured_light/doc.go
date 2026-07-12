// Package structured_light is a standard-library-only implementation of the
// structured-light pattern generation and decoding routines from OpenCV's
// contrib "structured_light" module, built on top of the root cv package
// (github.com/malcolmston/opencv).
//
// It depends only on the Go standard library and the root cv package: no cgo,
// no third-party code, and it imports no sibling cv/* subpackage. The root
// package supplies the [github.com/malcolmston/opencv.Mat] container (8-bit,
// row-major, channel-interleaved); everything specific to structured light is
// implemented here from scratch.
//
// # What structured light is
//
// A structured-light scanner projects a known sequence of patterns onto a
// scene and observes them with a camera. Because the projected pattern is
// known, each camera pixel can be labelled with the projector pixel (column
// and/or row) that illuminated it. That per-pixel camera→projector
// correspondence is the raw ingredient for triangulating 3-D geometry. This
// package produces the patterns and recovers the correspondence; it does not
// perform projector-camera calibration or triangulation (see the deferred list).
//
// # Gray-code patterns
//
// [GrayCodePattern] encodes the projector column and row indices in binary
// reflected Gray code, one bit per projected image. Gray code is used because
// adjacent codes differ in exactly one bit, so a decoding error never produces
// a wildly wrong coordinate. For each bit a pattern and its photometric inverse
// are projected; comparing the two makes bit decisions independent of surface
// albedo and ambient light. A fully-lit (white) and fully-dark (black)
// reference pair is also projected to build a shadow mask and to gauge the
// per-pixel contrast used for the robust-bit test.
//
// The pattern vector produced by [GrayCodePattern.Generate] is laid out
// column-bits first, then row-bits, with each bit immediately followed by its
// inverse:
//
//	[col0 col0inv col1 col1inv ... row0 row0inv row1 row1inv ...]
//
// [GrayCodePattern.Decode] consumes a captured stack in that exact order,
// together with the white/black references, and returns a [Decoded] map giving,
// for every camera pixel, the projector column and row it corresponds to plus a
// validity mask.
//
// # Sinusoidal (phase-shifting) patterns
//
// [SinusoidalPattern] implements N-step phase-shifting profilometry. It projects
// N sinusoidal fringe patterns, each phase-shifted by a fixed amount, and
// recovers a wrapped phase map with [SinusoidalPattern.ComputeWrappedPhase].
// The wrapped phase lies in (-π, π]; [UnwrapPhaseMap] removes the 2π
// discontinuities along the fringe direction to yield a continuous absolute
// phase that is proportional to the projector coordinate.
//
// # Determinism
//
// Everything here is deterministic: pattern generation is a pure function of the
// [GrayCodeParams] / [Params] dimensions, and decoding is a pure function of its
// inputs. No randomness, no clocks, no goroutines. Identical inputs always yield
// identical outputs, which is what makes the tests in this package exact.
//
// # Visualization
//
// Decoded coordinate maps and phase maps are plain Go slices. [CoordMapToMat]
// and [PhaseMapToMat] scale them into a single-channel [github.com/malcolmston/opencv.Mat]
// suitable for saving or display, and [MaskToMat] renders a boolean mask.
//
// # Deferred
//
// The following parts of the OpenCV module are intentionally NOT implemented:
//
//   - Projector-camera calibration and triangulation to 3-D points. Decoding
//     stops at the 2-D camera→projector correspondence; no depth is produced.
//   - Multi-frequency / temporal phase unwrapping. [UnwrapPhaseMap] performs a
//     simple line-by-line spatial unwrap along the fringe direction, which is
//     exact for a clean, monotonic phase ramp but is not quality-guided and can
//     be defeated by noise or true 2π-per-pixel gradients.
//   - The Fourier-transform profilometry (FTP) and marker-based absolute-phase
//     methods of cv::structured_light::SinusoidalPattern. Only the classic
//     N-step phase-shifting method is provided.
//   - Reading/writing real projector-camera captures; the "capture" step in the
//     tests is simulated by sampling the generated patterns at a known mapping.
package structured_light
