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
// phase that is proportional to the projector coordinate. [NStepWrappedPhase]
// is a stack-only, arbitrary-step generalization of the same estimator.
//
// # Temporal phase unwrapping
//
// A single fringe frequency wraps every period, so a spatial unwrap fails
// wherever the surface is discontinuous. The temporal routines resolve the
// ambiguity per-pixel instead, from a set of frequencies:
// [MultiFrequencyUnwrap] performs hierarchical ratio unwrapping and
// [HeterodyneUnwrap] the two-/three-frequency beat method. Both take
// [FrequencyPhase] levels and recover an absolute phase whose range far exceeds
// 2π. [CombineGrayAndPhase] fuses a coarse Gray-code fringe order with a fine
// phase-shift wrap (the Gray-code-plus-phase-shift hybrid), and
// [QualityGuidedUnwrap] follows a [PhaseGradientQuality] map to unwrap 2-D
// fields along their most reliable path first.
//
// # Fourier-transform profilometry
//
// [FTPWrappedPhase] (and the fixed-band [FTPWrappedPhaseBand]) recover phase
// from a single fringe image by band-pass filtering one sideband in the
// frequency domain — a one-shot alternative to multi-image phase shifting,
// implemented with a dependency-free DFT.
//
// # Quality maps and masking
//
// [ComputeDataModulation], [ComputeAmplitude] and [ComputeBackground] turn a
// phase-shift stack into per-pixel confidence and signal maps. [ShadowMask],
// [OverexposureMask] and [ModulationMask] (combined with [CombineMasks]) reject
// unlit, saturated and low-contrast pixels before decoding.
//
// # Binary vs. Gray encoding
//
// [CodePattern] generates and decodes column/row codes under a selectable
// [Encoding] — reflected Gray (robust, the default) or natural binary — so the
// two schemes can be compared through an identical pipeline.
//
// # Triangulation and stereo
//
// Given a decoded correspondence and calibrated [CameraMatrix] projection
// matrices (built with [NewPinhole]), [TriangulatePoint] and [Triangulate]
// reconstruct 3-D world points into a [PointCloud] by the linear DLT method.
// [StereoDecode] converts two cameras' decodings into projector-referenced
// [StereoMatch] correspondences that [TriangulateStereo] reconstructs.
// Calibration itself (recovering the intrinsics/pose) is still out of scope;
// the projection matrices are supplied by the caller.
//
// # Pattern export
//
// [WritePatternPNG], [EncodePatternPNG], [SavePatternPNG] and
// [SavePatternStack] serialize generated pattern stacks to PNG for projection.
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
// The following parts of the OpenCV module remain intentionally NOT implemented:
//
//   - Projector-camera calibration. Triangulation is provided
//     ([TriangulatePoint], [Triangulate], [TriangulateStereo]) but the
//     projection matrices must be supplied by the caller; this package does not
//     recover intrinsics, distortion or pose from calibration captures.
//   - Marker-based absolute-phase disambiguation of
//     cv::structured_light::SinusoidalPattern. Absolute phase is instead
//     obtained by temporal unwrapping ([MultiFrequencyUnwrap],
//     [HeterodyneUnwrap]), Gray-code fusion ([CombineGrayAndPhase]) or FTP
//     ([FTPWrappedPhase]).
//   - Lens-distortion modelling; the pinhole [CameraMatrix] is linear only.
//   - Reading/writing real projector-camera captures; the "capture" step in the
//     tests is simulated by sampling the generated patterns at a known mapping.
package structured_light
