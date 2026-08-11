# intensity — from-scratch, standard-library-only port of a useful subset of OpenCV's contrib

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/intensity.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/intensity)

Package intensity is a from-scratch, standard-library-only port of a useful
subset of OpenCV's contrib intensity_transform module, together with a small
set of classic point operations and a low-light tone-mapping pipeline.

The package sits on top of the root module github.com/malcolmston/opencv
(imported as cv) and the Go standard library (only math). It uses no cgo and
no third-party dependencies, and it does not import any of the other cv/*
subpackages. Every function operates on the package's central image type,
`cv.Mat` (8-bit unsigned samples, one or three channels), and returns a
freshly allocated `cv.Mat`; inputs are never mutated.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/intensity"
```

## Usage

This is the package's own `ExampleAutoContrast`, so it compiles and its output is
asserted on every `go test ./intensity/`.

```go
m := cv.NewMat(1, 4, 1)
	m.Set(0, 0, 0, 50)
	m.Set(0, 1, 0, 100)
	m.Set(0, 2, 0, 150)
	m.Set(0, 3, 0, 200)
	out := intensity.AutoContrast(m, 0)
	fmt.Println(out.At(0, 0, 0), out.At(0, 1, 0), out.At(0, 2, 0), out.At(0, 3, 0))
```

```
0 85 170 255
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AGCWD(img *cv.Mat, alpha float64) *cv.Mat` | AGCWD applies Adaptive Gamma Correction with Weighting Distribution, after Huang, Cheng and Chiu (2013), "Efficient Contrast Enhancement Using… |
| `func ApplyLUT(img *cv.Mat, lut []uint8) *cv.Mat` | ApplyLUT maps every sample of every channel of img through the 256-entry lookup table lut and returns a new `cv.Mat` of the same shape. |
| `func AutoContrast(img *cv.Mat, clipPercent float64) *cv.Mat` | AutoContrast performs a per-channel percentile contrast stretch, the operation PIL calls autocontrast. |
| `func AutoGamma(img *cv.Mat) *cv.Mat` | AutoGamma applies an automatic power-law correction whose exponent is chosen from the image's own statistics so that the mean luminance is driven… |
| `func AutoGammaValue(img *cv.Mat) float64` | AutoGammaValue returns the exponent `AutoGamma` would use for img without applying it: gamma = log(0.5)/log(mean), where mean is the average… |
| `func AutoLevels(img *cv.Mat, lowPercent, highPercent float64) *cv.Mat` | AutoLevels performs a luminance-driven levels adjustment. |
| `func AutoscaleContrast(img *cv.Mat) *cv.Mat` | AutoscaleContrast performs a per-channel min–max contrast stretch: for each channel it finds the smallest and largest sample value present and… |
| `func BIMEF(img *cv.Mat) *cv.Mat` | BIMEF enhances a low-light image with a Bio-Inspired Multi-Exposure Fusion pipeline after Ying et al. |
| `func BIMEFRefined(img *cv.Mat) *cv.Mat` | BIMEFRefined enhances a low-light image with the full Bio-Inspired Multi-Exposure Fusion pipeline of Ying et al. |
| `func BIMEFWithParams(img *cv.Mat, p BIMEFParams) *cv.Mat` | BIMEFWithParams runs the refined BIMEF pipeline with explicit parameters. |
| `func BitPlaneSlicing(img *cv.Mat, plane int) *cv.Mat` | BitPlaneSlicing extracts a single bit plane of img as a binary image: an output sample is 255 where bit plane is set in the corresponding input… |
| `func CLAHEColor(img *cv.Mat, clipLimit float64, tileGridSize int) *cv.Mat` | CLAHEColor extends Contrast-Limited Adaptive Histogram Equalisation to colour images. |
| `func ContrastLimitedStretch(img *cv.Mat, clipPercent, maxGain float64) *cv.Mat` | ContrastLimitedStretch is a percentile stretch whose amplification is capped. |
| `func ContrastStretching(img *cv.Mat, r1, s1, r2, s2 float64) *cv.Mat` | ContrastStretching applies a three-segment piecewise-linear intensity map defined by the two control points (r1,s1) and (r2,s2): |
| `func DefaultRetinexScales() []float64` | DefaultRetinexScales returns the classic three-scale set {15, 80, 250} used by `MultiScaleRetinex` and `MSRCR` when no scales are supplied. |
| `func DodgeAndBurn(img *cv.Mat, amount, sigma float64) *cv.Mat` | DodgeAndBurn evens out an image's illumination the way a darkroom printer dodges shadows and burns highlights. |
| `func ExpTransform(img *cv.Mat) *cv.Mat` | ExpTransform applies the exponential intensity transform |
| `func GammaCorrection(img *cv.Mat, gamma float64) *cv.Mat` | GammaCorrection applies the power-law (gamma) transform |
| `func GammaLUT(gamma float64) []uint8` | GammaLUT builds and returns the 256-entry power-law lookup table |
| `func HistogramMatching(img, reference *cv.Mat) *cv.Mat` | HistogramMatching (histogram specification) remaps img so that the cumulative distribution of each channel approximates that of the corresponding… |
| `func IntensityLevelSlicing(img *cv.Mat, low, high, value uint8, preserveBackground bool) *cv.Mat` | IntensityLevelSlicing highlights the intensity band [low, high]. |
| `func Invert(img *cv.Mat) *cv.Mat` | Invert returns the photographic negative of img, mapping every sample r to 255 − r on every channel. |
| `func LogAdaptiveTonemap(img *cv.Mat, bias float64) *cv.Mat` | LogAdaptiveTonemap applies the adaptive logarithmic tone-mapping operator of Drago et al. |
| `func LogTransform(img *cv.Mat) *cv.Mat` | LogTransform applies the logarithmic intensity transform |
| `func MSRCR(img *cv.Mat, sigmas []float64) *cv.Mat` | MSRCR enhances img with multi-scale retinex with colour restoration (MSRCR), adding a chromatic term that counteracts the greying-out MSR causes on… |
| `func MultiScaleRetinex(img *cv.Mat, sigmas []float64) *cv.Mat` | MultiScaleRetinex enhances img with multi-scale retinex (MSR): the per-channel retinex responses at each scale in sigmas are averaged, then stretched… |
| `func Posterize(img *cv.Mat, levels int) *cv.Mat` | Posterize quantises each channel of img to levels evenly spaced intensity values spanning [0,255] (the endpoints are always representable), reducing… |
| `func SingleScaleRetinex(img *cv.Mat, sigma float64) *cv.Mat` | SingleScaleRetinex enhances img with the single-scale retinex (SSR) of Jobson, Rahman and Woodell (1997). |
| `func Solarize(img *cv.Mat, threshold uint8) *cv.Mat` | Solarize inverts every sample at or above threshold, leaving lower samples unchanged: s = 255 − r for r >= threshold, and s = r otherwise. |
| `func ToneCurve(img *cv.Mat, points []CurvePoint) *cv.Mat` | ToneCurve remaps every channel of img through the natural cubic spline defined by points and returns a new `cv.Mat`. |
| `func ToneCurveLUT(points []CurvePoint) []uint8` | ToneCurveLUT builds a 256-entry lookup table by fitting a natural cubic spline through points (which must be sorted by In, strictly increasing, each… |
| `func UnsharpMask(img *cv.Mat, sigma, amount, threshold float64) *cv.Mat` | UnsharpMask sharpens img by the classic unsharp-masking rule: it subtracts a Gaussian-blurred copy to form a high-pass "mask" and adds a multiple of… |

### Types

| Type | What it is |
| --- | --- |
| `BIMEFParams` | BIMEFParams configures the refined Bio-Inspired Multi-Exposure Fusion pipeline `BIMEFWithParams`. |
| `CurvePoint` | CurvePoint is a single control point of a `ToneCurve`, mapping input intensity In to output intensity Out. |

<details>
<summary><code>BIMEFParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultBIMEFParams() BIMEFParams` | DefaultBIMEFParams returns the parameters used by `BIMEFRefined`: a mild fusion exponent, a moderately smooth weighted-least-squares illumination… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/intensity).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
