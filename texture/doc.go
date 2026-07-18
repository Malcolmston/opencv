// Package texture implements classic image-texture analysis on top of the
// root package's [cv.Mat] image type, using only the Go standard library.
//
// Texture is the spatial arrangement of intensities in an image — the visual
// "feel" of a surface such as grass, brick, fabric or a medical scan. Unlike
// colour or shape, texture is a statistical property of a neighbourhood rather
// than of a single pixel, so every routine here summarises how gray levels are
// distributed and co-arranged across a region.
//
// The package groups the standard families of texture descriptors:
//
//   - Gray-Level Co-occurrence Matrix (GLCM) and the 14 Haralick features
//     ([GLCM], [NewGLCM], [ComputeGLCM], [GLCM.Haralick]).
//   - Local Binary Patterns, including uniform and rotation-invariant variants
//     ([LBP], [LBPUniform], [LBPRotationInvariant], [LBPUniformHistogram]).
//   - Gabor texture energy from oriented, band-pass filters
//     ([GaborKernel], [GaborEnergy], [GaborFeatures]).
//   - Laws texture-energy measures from separable 1-D masks
//     ([LawsEnergyMaps], [LawsFeatures]).
//   - Tamura perceptual features — coarseness, contrast, directionality
//     ([TamuraFeatures]).
//   - Gray-Level Run-Length Matrix statistics
//     ([RunLengthMatrix], [NewRunLengthMatrix], [RunLengthMatrix.Features]).
//   - Fractal texture dimension via box counting
//     ([BoxCountingDimension], [DifferentialBoxCounting], [Lacunarity]).
//
// # Images and gray levels
//
// Every routine accepts a [cv.Mat]. Three-channel input is reduced to
// luminance with the Rec. 601 weights (0.299 R, 0.587 G, 0.114 B); a
// single-channel Mat is used directly; any other channel count falls back to
// channel 0. Co-occurrence, run-length and LBP-histogram routines additionally
// quantise the 8-bit luminance into a smaller number of gray levels (a common
// choice is 8, 16 or 32), which controls the size of the matrices and the
// statistical stability of the features.
//
// All functions are deterministic, CPU-only and free of third-party
// dependencies. Invalid arguments (empty images, non-positive level counts,
// out-of-range parameters) cause a panic whose message is prefixed "texture:".
package texture
