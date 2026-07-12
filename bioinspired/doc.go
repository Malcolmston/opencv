// Package bioinspired is a from-scratch, standard-library-only port of a useful
// subset of OpenCV's bioinspired contrib module: biologically-inspired
// (retina) vision models. It mirrors the human retina's two main pathways to
// denoise and enhance images (the parvocellular channel), extract motion and
// moving contours (the magnocellular channel), and compress high dynamic range
// for display (fast tone mapping).
//
// Like the parent package [github.com/malcolmston/opencv] (imported here under
// the alias cv), bioinspired is written entirely against the Go standard
// library (only math, plus fmt for error messages). It uses no cgo and no
// third-party dependencies, and it does not import any of the sibling cv/*
// subpackages: helper routines such as separable low-pass filtering and colour
// conversion are reimplemented locally in floating point.
//
// # The retina model
//
// The retina is modelled as a cascade of spatio-temporal filters, following the
// Gipsa-lab model of Benoit, Caplier, Durette and Herault. Two pathways share a
// common front end:
//
//   - Photoreceptors apply a local luminance adaptation (a Naka-Rushton /
//     Michaelis-Menten compression) whose reference is a spatio-temporal
//     low-pass of the luminance. This compresses dynamic range and boosts faint
//     detail in dark regions.
//   - Horizontal cells low-pass the photoreceptor signal in space and time; the
//     photoreceptor-minus-horizontal difference forms the outer-plexiform-layer
//     (OPL) band-pass, which removes the local mean and highlights contours.
//
// The inner plexiform layer (IPL) then splits into two channels:
//
//   - The parvocellular (parvo) channel spatially smooths the signal for noise
//     reduction, adds back a fraction of the OPL band-pass for detail/colour
//     enhancement, and applies a ganglion-cell contrast normalisation. Its
//     output is a denoised, sharpened image the same shape as the input. See
//     [Retina.GetParvo].
//   - The magnocellular (magno) channel differences the signal against a
//     temporal low-pass carried across frames (a transient / temporal high-pass
//     response), rectifies it and smooths it spatially. Its output is strong at
//     moving edges and near zero on a static scene. See [Retina.GetMagno].
//
// A [Retina] is stateful. Feed frames one at a time with [Retina.Run], read the
// latest result with [Retina.GetParvo] / [Retina.GetMagno] (or the unquantised
// [Retina.GetParvoRAW] / [Retina.GetMagnoRAW]), and call [Retina.ClearBuffers]
// to reset the temporal state between independent sequences. Parameters are
// exposed through [RetinaParameters] and [DefaultRetinaParameters].
//
// # Fast tone mapping
//
// [RetinaFastToneMapping] reuses the photoreceptor and ganglion-cell local
// adaptation stages, without any temporal state, to compress a high-dynamic-
// range-ish image into the displayable [0,255] range while lifting shadow
// detail. See [RetinaFastToneMapping.ProcessFrame].
//
// # Conventions
//
//   - Inputs and outputs are 8-bit [cv.Mat] values. Single-channel Mats are
//     treated as luminance; three-channel Mats are treated as RGB (not BGR),
//     matching the parent package. Parvo output keeps the input channel count;
//     magno output is always single-channel.
//   - All internal computation is in float64 to preserve the small signals
//     produced by band-pass and temporal-difference filtering; quantisation to
//     8-bit happens only at the [cv.Mat] output boundary, with round-to-nearest
//     and clamping to [0,255].
//   - Spatial filtering uses a separable, zero-phase, first-order recursive
//     low-pass (a two-sided exponential kernel) with unit DC gain, so flat
//     regions keep their mean. Temporal filtering uses an exponential recursive
//     low-pass whose state is carried across [Retina.Run] calls.
//
// # Determinism
//
// All functions are fully deterministic: no randomness, no clocks, no
// concurrency. The same sequence of inputs and parameters always yields
// bit-identical outputs. Types in this package are not safe for concurrent use.
//
// # Deferred / simplifications versus OpenCV
//
// This is a faithful but deliberately simplified port. The following aspects of
// the full Benoit et al. model and OpenCV's implementation are not implemented:
//
//   - No log-polar cortical (retino-cortical) sampling or foveal magnification;
//     all processing is on the uniform Cartesian pixel grid.
//   - The OPL/IPL filters are modelled as separable exponential low-pass and
//     difference-of-low-pass stages rather than the exact coupled
//     photoreceptor / horizontal-cell / bipolar / amacrine differential
//     equations with their full coefficient derivation.
//   - No spatial-constant adaptation driven by the local signal
//     (EnableSpatialConstantsAdaptation is accepted but not acted upon), and no
//     separate ON/OFF ganglion-cell rectified pathways.
//   - No colour multiplexing/demultiplexing on a Bayer mosaic; colour is
//     processed as independent RGB planes sharing a luminance reference.
//   - RetinaFastToneMapping approximates OpenCV's dedicated fast-tone-mapping
//     class with two cascaded local-adaptation stages rather than the full
//     retina filter bank.
package bioinspired
