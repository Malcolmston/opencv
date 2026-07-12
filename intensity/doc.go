// Package intensity is a from-scratch, standard-library-only port of a useful
// subset of OpenCV's contrib intensity_transform module, together with a small
// set of classic point operations and a low-light tone-mapping pipeline.
//
// The package sits on top of the root module github.com/malcolmston/opencv
// (imported as cv) and the Go standard library (only math). It uses no cgo and
// no third-party dependencies, and it does not import any of the other cv/*
// subpackages. Every function operates on the package's central image type,
// [cv.Mat] (8-bit unsigned samples, one or three channels), and returns a
// freshly allocated [cv.Mat]; inputs are never mutated.
//
// # Point (per-pixel) transforms
//
// These functions map each sample through a fixed 256-entry lookup table, so
// they apply identically to every channel of a multi-channel image and run in
// O(pixels):
//
//   - [GammaCorrection] — power-law s = 255·(r/255)^γ. γ = 1 is the identity;
//     γ < 1 brightens midtones, γ > 1 darkens them.
//   - [LogTransform] — s = c·log(1+r), c = 255/log(256). Compresses the
//     dynamic range, expanding dark detail. Maps 0→0 and 255→255.
//   - [ExpTransform] — the exact inverse of [LogTransform], s = exp(r/c)−1,
//     expanding bright detail. Maps 0→0 and 255→255.
//   - [ContrastStretching] — a three-segment piecewise-linear map through the
//     control points (r1,s1) and (r2,s2); the endpoints are reproduced exactly.
//   - [IntensityLevelSlicing] — highlights the intensity band [low,high],
//     optionally preserving the background.
//   - [BitPlaneSlicing] — extracts a single bit plane (0 = LSB … 7 = MSB) as a
//     binary 0/255 image.
//   - [Solarize] — inverts samples at or above a threshold (the classic
//     darkroom / Sabattier effect).
//   - [Posterize] — quantises each channel to a small number of evenly spaced
//     intensity levels.
//   - [Invert] — photographic negative, s = 255 − r.
//
// The 256-entry tables that drive these maps can also be built directly and
// reused: [GammaLUT] and [ToneCurveLUT] return tables that [ApplyLUT] applies to
// any image.
//
// # Global (data-dependent) transforms
//
//   - [AutoscaleContrast] — per-channel min–max normalisation that stretches
//     the observed range to the full [0,255] endpoints (a linear contrast
//     stretch).
//   - [HistogramMatching] — histogram specification: remaps an image so that
//     its cumulative distribution approximates that of a reference image, per
//     channel.
//   - [AutoContrast] — per-channel percentile stretch (PIL's autocontrast): clip
//     a fraction of each channel's tails, then stretch to the full range.
//   - [AutoLevels] — luminance-driven black/white-point stretch applied to every
//     channel, so contrast is expanded without shifting colour balance.
//   - [ContrastLimitedStretch] — a percentile stretch whose gain is capped, so a
//     nearly flat image is not amplified into noise.
//
// # Adaptive tone and gamma
//
//   - [AutoGamma] / [AutoGammaValue] — an automatic power-law whose exponent is
//     chosen from the image mean so the mid-tone is driven toward 0.5.
//   - [AGCWD] — Adaptive Gamma Correction with Weighting Distribution (Huang et
//     al. 2013): a per-intensity exponent from a reshaped histogram.
//   - [ToneCurve] / [ToneCurveLUT] — a photo-editor "curves" adjustment fitting a
//     natural cubic spline through control points ([CurvePoint]).
//   - [LogAdaptiveTonemap] — the adaptive logarithmic operator of Drago et al.
//     (2003): lifts shadows and compresses highlights.
//
// # Retinex
//
//   - [SingleScaleRetinex] / [MultiScaleRetinex] / [MSRCR] — the retinex family
//     (Jobson et al. 1997), discounting the illumination via a log-domain
//     Gaussian surround, with optional multi-scale fusion and colour
//     restoration. [DefaultRetinexScales] supplies the classic scale set.
//
// # Local (spatial) operators
//
//   - [UnsharpMask] — high-pass sharpening with an optional threshold.
//   - [DodgeAndBurn] — a surround-guided local exposure adjustment that lightens
//     shadows and darkens highlights to even out illumination.
//   - [CLAHEColor] — colour Contrast-Limited Adaptive Histogram Equalisation,
//     applying luminance CLAHE while preserving hue.
//
// # Tone mapping
//
//   - [BIMEF] — a Bio-Inspired Multi-Exposure Fusion pipeline for low-light
//     enhancement, after Ying et al. (2017). See the [BIMEF] documentation for
//     the approximation used and what is deferred.
//   - [BIMEFRefined] / [BIMEFWithParams] — the fully realised BIMEF pipeline
//     ([BIMEFParams], [DefaultBIMEFParams]): an edge-preserving
//     weighted-least-squares illumination map and an entropy-maximising exposure
//     ratio, the refinements [BIMEF] defers.
//
// # Conventions
//
// Multi-channel images are treated as independent channels for the point and
// global transforms (there is no colour-space coupling); [BIMEF] instead uses
// the per-pixel channel maximum as its illumination estimate. All functions
// panic on an empty image, and each documents any additional preconditions.
// Rounding matches the root package: a value is biased by +0.5 and truncated
// toward zero before being clamped into [0,255], so results agree with cv's own
// transforms at the boundary.
package intensity
