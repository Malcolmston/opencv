// Package mcc provides Macbeth/ColorChecker chart detection and color
// correction on top of the standard-library-only OpenCV port
// github.com/malcolmston/opencv (imported here as cv). It mirrors a useful
// subset of OpenCV's mcc contrib module: reference chart data, a
// CCheckerDetector-style automatic detector, and a ColorCorrectionModel (CCM)
// that learns a linear least-squares mapping from a camera's measured patch
// colors to the chart's reference colors and applies it to arbitrary images.
//
// # Conventions
//
// Every image is a [cv.Mat] of 8-bit unsigned samples. Color images are
// three-channel RGB (channel 0 red, 1 green, 2 blue), never BGR — matching the
// root package. Patch and pixel colors are carried through the pipeline as
// float64 triples in the 0..255 sRGB range so that averaging and matrix math do
// not lose precision; they are only quantized back to uint8 when written into a
// Mat.
//
// All colorimetric math is done locally in this package with no dependency on
// the root package's (unexported) color internals: sRGB<->linear companding,
// linear-RGB<->CIE XYZ (sRGB primaries), XYZ<->CIE L*a*b* under the D65 white
// point, and the CIE76 Delta E metric. See color.go.
//
// # Color science
//
// Beyond the CIE76 basics, the package provides a fuller color-difference and
// color-space toolkit:
//
//   - Delta E metrics: [DeltaE94] and [DeltaE94Textiles] (CIE94), [DeltaE2000]
//     and [DeltaE2000Weighted] (CIEDE2000, matching the Sharma et al. reference
//     data), and [DeltaECMC] with the [DeltaECMCAcceptability] (2:1) and
//     [DeltaECMCPerceptibility] (1:1) presets. [DeltaE2000RGB] wraps sRGB inputs.
//   - Color spaces: [LabToLCh]/[LChToLab] (cylindrical Lab), [XYZToxyY]/[XYYToXYZ]
//     (chromaticity), white-point-parameterised [XYZToLab]/[LabToXYZ],
//     [LinearRGBToXYZ], [XYZToRGB], [LabToRGB], and the pure power-law
//     [GammaExpand]/[GammaCompress].
//   - Chromatic adaptation: [ChromaticAdaptation] and [AdaptationMatrix] between
//     the [WhiteD65], [WhiteD50] and [WhiteA] illuminants using the [Bradford],
//     [VonKries] or [XYZScaling] cone-response models, with [ApplyMatrix3] for
//     applying a 3x3 transform.
//
// # ColorChecker Digital SG
//
// [DigitalSGReference] provides the 140-patch (10-row by 14-column) ColorChecker
// Digital SG chart, with [DigitalSGReferenceRGB]/[DigitalSGReferenceLab] and the
// [DigitalSGRows]/[DigitalSGCols]/[DigitalSGNumPatches] geometry accessors.
// [RenderSGChart] draws it, [DigitalSGOuterQuad] gives the outer corners, and
// [SampleSGWithHint] samples a [CCheckerSG] from a known quad, reporting error
// against the reference just like [CChecker].
//
// # Richer color correction
//
// [ColorCorrectionModel] (built with [TrainColorCorrection]) extends [CCM] with
// full 2nd- and 3rd-degree polynomials ([ModelPoly2], [ModelPoly3]) and
// exposure-invariant Finlayson root-polynomials ([ModelRootPoly2],
// [ModelRootPoly3]), optional [LinIdentity]/[LinSRGB]/[LinGamma] linearization,
// white-balance-first fitting and weighted least squares (see [LuminanceWeights]).
// [ColorCorrectionModel.Infer] color-corrects an image with clamping,
// [ColorCorrectionModel.MeanDeltaE2000] scores it, and
// [ColorCorrectionModel.Report] gives a per-patch Lab and Delta E breakdown.
// [DetectorParameters] bundles and validates the detector's tuning knobs.
//
// # Reference charts
//
// [CheckerType] enumerates the supported charts. [Macbeth24] is the classic
// 24-patch X-Rite/GretagMacbeth ColorChecker arranged as 4 rows by 6 columns;
// [Vinyl] is a second, independently-tabulated 24-patch reference set (the
// BabelColor community average) with the same geometry but slightly different
// values. [ReferenceChart] returns the per-patch reference sRGB and (derived)
// Lab values in row-major grid order.
//
// # Rendering
//
// [RenderChart] synthesizes a canonical, deterministic chart image from the
// reference values: each patch is a solid square separated from its neighbours
// (and the image edge) by a black gap. It is handy for demonstrations, unit
// tests, and as ground truth for the detector.
//
// # Detection
//
// [CCheckerDetector] recovers a chart from an image and samples its patches:
//
//  1. convert to grayscale and threshold so the dark inter-patch gaps become
//     background and the patches become foreground,
//  2. trace external contours (via cv.FindContours) and keep convex four-vertex
//     quadrilaterals of plausible, roughly-square size as patch candidates,
//  3. take the convex hull of the candidate centres and pick the four extreme
//     corners (the corner patches) as the chart's frame,
//  4. for each of the eight possible corner orderings, build a perspective
//     transform (via cv.GetPerspectiveTransform) that maps a canonical grid onto
//     the image, sample all patch centres, and score the reading against the
//     reference; keep the best-scoring orientation.
//
// The perspective step makes detection robust to moderate keystone/rotation.
// [CCheckerDetector.DetectWithHint] skips the search when the caller already
// knows the chart's four outer corners (top-left, top-right, bottom-right,
// bottom-left of the patch array), which is the most robust path.
//
// # Color correction
//
// [TrainCCM] fits a [CCM] by linear least squares (normal equations) mapping
// measured colors to reference colors. Three model shapes are offered:
// [CCMLinear3x3] (pure 3x3 gain), [CCMAffine3x4] (3x3 plus an offset), and
// [CCMPolynomial] (a degree-2 polynomial expansion for mild non-linearity).
// An optional linearization step ([CCMConfig.Linearize]) fits in linear-light
// space, undoing the sRGB gamma (or a caller-supplied [CCMConfig.Gamma]) before
// the matrix and re-applying it afterwards, which usually lowers residual
// error. [CCM.Apply] color-corrects a whole image; [CCM.ApplyRGB] a single
// color.
//
// # Error metrics
//
// [DeltaE76] is the CIE76 Delta E between two Lab colors; [DeltaERGB] converts a
// pair of sRGB colors first. [CChecker.PatchErrors], [CChecker.MeanError] and
// [CChecker.MaxError] report per-patch and aggregate Delta E of a detection
// against its reference. [CCM.MeanError] reports residual error after
// correction.
//
// # Determinism
//
// Every routine here is deterministic: no randomness, no goroutines, no
// wall-clock or environment input. Given the same image and parameters the same
// chart, measurements and correction matrix are produced.
//
// # Deferred
//
// The following parts of OpenCV's mcc module (and adjacent color science) are
// intentionally NOT implemented here:
//
//   - No trained or machine-learning-based detector (OpenCV's mcc ships an
//     optional DNN net-based CCheckerDetector); detection here is purely
//     classical contour/geometry. The 140-patch DigitalSG is sampled from a
//     supplied quad ([SampleSGWithHint]) rather than searched for automatically.
//   - The 18-patch ColorChecker Passport geometry and reference table is not
//     provided (the 24-patch [Macbeth24]/[Vinyl] and 140-patch DigitalSG charts
//     are).
//   - The DigitalSG reference reproduces the real 10x14 geometry with the
//     classic 24 patches anchored to [Macbeth24] and a deterministic Lab sweep
//     for the remaining cells; it is not a substitute for a factory-measured SG
//     chart's tabulated values.
//   - No networked or on-disk chart databases, camera profiles or ICC output.
//   - The correction models are fitted by (weighted) least squares; robust
//     (saturation-masked) outlier rejection and iterative refinement are not
//     included.
package mcc
