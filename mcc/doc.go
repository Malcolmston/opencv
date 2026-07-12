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
//     classical contour/geometry.
//   - No 140-patch ColorChecker Digital SG or 18-patch Passport geometry and
//     reference tables; only the 24-patch layout is provided as real data (two
//     tabulations: [Macbeth24] and [Vinyl]).
//   - Delta E is CIE76 only; CIE94, CIEDE2000 and CMC are not implemented.
//   - The CCM covers 3x3, 3x4 and a fixed degree-2 polynomial; arbitrary
//     polynomial degrees, root-polynomial models, per-channel weighting, robust
//     (saturation-masked) fitting and iterative refinement are not included.
//   - No networked or on-disk chart databases, camera profiles or ICC output.
//   - Chromatic-adaptation white points other than D65 are not supported.
package mcc
