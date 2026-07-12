// Package xphoto is a standard-library-only port of a useful subset of
// OpenCV's xphoto contrib module, built on top of the root cv package
// (github.com/malcolmston/opencv). It collects "extra" photo-processing
// algorithms that sit alongside the mainline photo module: automatic white
// balance, per-channel gain application, exemplar/shift-based inpainting, oil
// painting stylization and BM3D denoising.
//
// # Data model
//
// Every function operates on the root package's [cv.Mat], a dense row-major
// matrix of 8-bit unsigned samples. Three-channel data is treated as RGB,
// matching Go's image package and the root cv conventions, not OpenCV's native
// BGR. Where a function is channel-agnostic it says so; where it requires a
// specific channel count it validates and panics on mismatch, mirroring the
// root package's fail-fast style.
//
// # Dependencies
//
// xphoto imports only the root cv package and the Go standard library (math,
// sort, fmt, image). It uses no cgo and no third-party code. Small numeric
// helpers (edge-replicated sampling, rounding-and-clamping to [0,255], luma,
// Sobel gradients) are reimplemented locally rather than reaching into cv's
// unexported internals, so the package is self-contained.
//
// # Determinism
//
// All algorithms are fully deterministic: they use no randomness, no
// goroutine-dependent iteration order and no floating-point reductions whose
// result depends on scheduling. Given the same input and parameters a function
// always returns byte-identical output. This makes the results reproducible and
// testable.
//
// # White balance
//
// The three white balancers share the [WhiteBalancer] interface
// (BalanceWhite). [SimpleWB] performs an independent per-channel histogram
// stretch with configurable input/output ranges and percentile clipping.
// [GrayworldWB] applies the gray-world assumption with a saturation threshold.
// [LearningBasedWB] approximates OpenCV's learning-based estimator: it extracts
// the same brightness/edge-weighted chromaticity color features and combines
// them with a gray-edge estimate through a robust (median) combiner. See the
// Deferred section for the honesty note on the trained regressor.
// [ApplyChannelGains] multiplies the R, G and B channels by explicit gains.
//
// # Algorithms and fidelity
//
// The implementations favour clarity and correctness over raw speed and are
// faithful in spirit rather than bit-exact with OpenCV:
//
//   - [Inpaint] fills a masked region with a SHIFTMAP-style exemplar search:
//     an onion-peel front is advanced inward, and each unknown pixel is filled
//     from the best-matching known patch (the translation/"shift" whose
//     overlapping known samples minimise sum-of-squared difference). This
//     mirrors the intent of xphoto::inpaint's shift-map / patch-based fill.
//   - [Oilpainting] reproduces xphoto::oilPainting exactly in structure: within
//     each neighbourhood it finds the most common intensity bucket and outputs
//     the mean colour of the pixels in that bucket, collapsing texture into
//     flat, painterly patches.
//   - [Bm3dDenoising] implements a genuine single-step (hard-threshold) BM3D:
//     block matching groups similar patches, a separable 2D-DCT plus a 1D-DCT
//     across the group form the collaborative transform, coefficients are
//     hard-thresholded, and inverse-transformed blocks are aggregated with
//     confidence weights.
//
// # Deferred
//
// The following OpenCV xphoto features are intentionally out of scope or
// simplified, and are documented here so callers are not misled:
//
//   - LearningBasedWB's trained regression model. OpenCV ships a gradient-boosted
//     regression tree trained on a labelled illuminant dataset and loaded from a
//     .yml model file. This port has no trained model; it replaces the regressor
//     with a self-contained robust estimator over the same colour/edge features.
//     Its accuracy is therefore approximate, not identical to OpenCV's.
//   - The second (Wiener) BM3D stage. OpenCV's BM3D runs a hard-threshold basic
//     estimate followed by an empirical-Wiener refinement using the basic
//     estimate as an oracle. This port implements only the basic estimate
//     (BM3D_STEP1 / step-1-of-2), which is a valid, weaker denoiser.
//   - The full SHIFTMAP EM optimisation. OpenCV's shift-map inpainting solves a
//     global energy with a multi-scale expansion-move / EM scheme. This port
//     uses a greedy onion-peel exemplar fill, which is local rather than
//     globally optimal.
//   - The FSR (Frequency-Selective Reconstruction) inpainting variant and the
//     TonemapDurand / dct/bilateral image-quality bits of xphoto.
package xphoto
