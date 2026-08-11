# photo2 — standard-library-only computational-photography toolkit built on the root cv package

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/photo2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/photo2)

Package photo2 is a standard-library-only computational-photography toolkit
built on the root cv package (github.com/malcolmston/opencv). It provides
high-dynamic-range (HDR) tonemapping, exposure fusion, edge-preserving
smoothing, detail enhancement, artistic stylization and colour grading, in the
spirit of OpenCV's photo and hdr modules but implemented entirely in Go with
no cgo and no third-party dependencies.

- `cv.Mat` is a dense row-major matrix of 8-bit samples, channel interleaved.
It is the standard display-referred (LDR) image. Grayscale uses one channel,
RGB three. Three-channel data is treated as RGB (matching Go's image package),
not OpenCV's native BGR. - `cv.FloatMat` is a single-channel float64 matrix. A
colour image in linear or working space is represented as a slice of
`cv.FloatMat`, one entry per channel, all sharing the same dimensions. This is
the natural container for HDR radiance maps and intermediate results whose
range is not confined to [0,255]. Use `ToFloat` and `FromFloat` to convert to
and from `cv.Mat`.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/photo2"
```

## Usage

This is the package's own `ExampleApplyLUT`, so it compiles and its output is
asserted on every `go test ./photo2/`.

```go
var invert [256]uint8
	for i := range invert {
		invert[i] = uint8(255 - i)
	}

	img := cv.NewMat(1, 2, 1)
	img.Data[0], img.Data[1] = 0, 200

	out := photo2.ApplyLUT(img, invert)
	fmt.Println(out.Data[0], out.Data[1])
```

```
255 55
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AdjustBrightness(img *cv.Mat, delta float64) *cv.Mat` | AdjustBrightness adds delta (in 8-bit units, may be negative) to every sample, clamping to [0,255]. |
| `func AdjustContrast(img *cv.Mat, factor float64) *cv.Mat` | AdjustContrast scales every sample around mid-grey (128): out = 128 + factor*(in-128). |
| `func AdjustSaturation(img *cv.Mat, factor float64) *cv.Mat` | AdjustSaturation scales colour saturation around the per-pixel luminance: out_c = luma + factor*(in_c - luma). |
| `func ApplyLUT(img *cv.Mat, lut [256]uint8) *cv.Mat` | ApplyLUT maps every sample of an image through the 256-entry lookup table, independently per channel. |
| `func BilateralFilter(img *cv.Mat, diameter int, sigmaColor, sigmaSpace float64) *cv.Mat` | BilateralFilter applies an edge-preserving bilateral filter to an image. |
| `func Blend(a, b *cv.Mat, alpha float64) *cv.Mat` | Blend returns the per-sample convex combination alpha*a + (1-alpha)*b. |
| `func BoxBlur(img *cv.Mat, radius int) *cv.Mat` | BoxBlur applies an averaging filter over a (2*radius+1) square window with reflected borders, independently per channel. |
| `func CLAHE(img *cv.Mat, clipLimit float64, tiles int) *cv.Mat` | CLAHE performs Contrast-Limited Adaptive Histogram Equalisation. |
| `func Cartoon(img *cv.Mat, sigmaS, sigmaR float64) *cv.Mat` | Cartoon renders an image in a cartoon style by flattening colour with an edge-preserving filter and overlaying bold black outlines detected from the… |
| `func Clamp01(f *cv.FloatMat) *cv.FloatMat` | Clamp01 returns a copy of f with every sample clamped into [0,1]. |
| `func ColorTransferReinhard(src, target *cv.Mat) *cv.Mat` | ColorTransferReinhard transfers the colour statistics of target onto src using Reinhard et al.'s (2001) method. |
| `func DetailEnhance(img *cv.Mat, sigmaS, sigmaR, factor float64) *cv.Mat` | DetailEnhance sharpens the fine texture of an image while preserving its large-scale structure and edges. |
| `func DomainTransformFilter(img *cv.Mat, mode EdgeFilterMode, sigmaS, sigmaR float64) *cv.Mat` | DomainTransformFilter applies Gastal and Oliveira's (2011) domain-transform edge-preserving filter. |
| `func DragoToneMap(channels []*cv.FloatMat, params DragoParams) *cv.Mat` | DragoToneMap applies Drago et al.'s (2003) adaptive logarithmic tone mapping. |
| `func DurandToneMap(channels []*cv.FloatMat, params DurandParams) *cv.Mat` | DurandToneMap applies Durand and Dorsey's (2002) fast bilateral-filtering tone mapping. |
| `func EdgePreservingFilter(img *cv.Mat, sigmaS, sigmaR float64) *cv.Mat` | EdgePreservingFilter smooths an image while keeping salient edges crisp, using the recursive domain-transform filter. |
| `func ExposureCompensate(img *cv.Mat, stops float64) *cv.Mat` | ExposureCompensate multiplies the linear scene value by 2^stops, simulating an exposure adjustment. |
| `func FromFloat(channels []*cv.FloatMat) *cv.Mat` | FromFloat merges per-channel float planes back into an 8-bit image. |
| `func GammaCorrection(img *cv.Mat, gamma float64) *cv.Mat` | GammaCorrection applies a power-law transfer curve out = 255*(in/255)^gamma, independently per channel. |
| `func GammaToneMap(channels []*cv.FloatMat, gamma float64) *cv.Mat` | GammaToneMap maps a linear HDR image to a display image by applying a simple gamma curve, out = clamp(in^(1/gamma))*255, independently per channel. |
| `func GaussianBlur(img *cv.Mat, sigma float64) *cv.Mat` | GaussianBlur convolves an 8-bit image with a separable Gaussian of the given standard deviation, independently per channel, using reflected borders. |
| `func GaussianBlurFloat(f *cv.FloatMat, sigma float64) *cv.FloatMat` | GaussianBlurFloat convolves a single-channel float matrix with a separable Gaussian of the given standard deviation, using reflected borders. |
| `func GaussianPyramid(f *cv.FloatMat, levels int) []*cv.FloatMat` | GaussianPyramid builds a Gaussian pyramid with the given number of levels. |
| `func GrayWorldWhiteBalance(img *cv.Mat) *cv.Mat` | GrayWorldWhiteBalance applies the grey-world assumption: it scales each colour channel so their means become equal to the overall mean, neutralising… |
| `func Grayscale(img *cv.Mat) *cv.Mat` | Grayscale converts an image to a single-channel 8-bit image using the Rec.709 luma weights. |
| `func GuidedFilter(img, guide *cv.FloatMat, radius int, eps float64) *cv.FloatMat` | GuidedFilter applies He et al.'s (2010) guided filter: an edge-preserving smoothing of img whose edges are taken from guide. |
| `func HistogramEqualize(img *cv.Mat) *cv.Mat` | HistogramEqualize performs global histogram equalisation. |
| `func HistogramStretch(img *cv.Mat) *cv.Mat` | HistogramStretch performs per-channel linear contrast stretching, remapping each channel's observed [min,max] to the full [0,255] range. |
| `func LabToRGB(channels []*cv.FloatMat) *cv.Mat` | LabToRGB converts CIELab planes (D65) back to an 8-bit sRGB image. |
| `func LaplacianPyramid(f *cv.FloatMat, levels int) []*cv.FloatMat` | LaplacianPyramid builds a Laplacian pyramid with the given number of levels. |
| `func LocalContrast(img *cv.Mat, sigma, amount float64) *cv.Mat` | LocalContrast enhances mid-scale local contrast (a "clarity" control) by amplifying the difference between the image and a Gaussian-blurred version… |
| `func LogToneMap(channels []*cv.FloatMat, bias float64) *cv.Mat` | LogToneMap compresses a linear HDR image with a logarithmic luminance curve, Ld = log(1 + bias*Lw) / log(1 + bias*Lmax), then re-introduces colour… |
| `func LogTransform(img *cv.Mat) *cv.Mat` | LogTransform applies a logarithmic transfer curve out = 255*log(1+in)/log(256), which strongly brightens shadows and compresses highlights. |
| `func Luminance(img *cv.Mat) *cv.FloatMat` | Luminance returns the Rec.709 relative luminance of an image as a single-channel `cv.FloatMat` scaled into [0,1]. |
| `func LuminanceChannels(channels []*cv.FloatMat) *cv.FloatMat` | LuminanceChannels returns the Rec.709 luminance of a set of float RGB planes. |
| `func MertensFusion(images []*cv.Mat, params MertensParams) *cv.Mat` | MertensFusion performs Mertens, Kautz and Van Reeth's (2007) exposure fusion. |
| `func Normalize(f *cv.FloatMat) *cv.FloatMat` | Normalize returns a copy of f linearly rescaled so its minimum maps to 0 and its maximum to 1. |
| `func PencilSketch(img *cv.Mat, sigmaS, sigmaR, shadeFactor float64) (gray *cv.Mat, color *cv.Mat)` | PencilSketch renders an image as a pencil drawing. |
| `func PyrDown(f *cv.FloatMat) *cv.FloatMat` | PyrDown blurs f with a Gaussian and downsamples it by a factor of two, returning a matrix of size ceil(rows/2) x ceil(cols/2). |
| `func PyrUp(f *cv.FloatMat, rows, cols int) *cv.FloatMat` | PyrUp upsamples f to the given target dimensions by pixel injection followed by a Gaussian smoothing (scaled by four to preserve brightness). |
| `func RGBToLab(img *cv.Mat) []*cv.FloatMat` | RGBToLab converts an 8-bit sRGB image to CIELab (D65). |
| `func RGBToXYZ(img *cv.Mat) []*cv.FloatMat` | RGBToXYZ converts an 8-bit sRGB image to linear CIE XYZ tristimulus planes (D65). |
| `func ReconstructLaplacianPyramid(pyr []*cv.FloatMat) *cv.FloatMat` | ReconstructLaplacianPyramid collapses a Laplacian pyramid back into a single matrix, the inverse of `LaplacianPyramid`. |
| `func ReinhardToneMap(channels []*cv.FloatMat, params ReinhardParams) *cv.Mat` | ReinhardToneMap applies Reinhard et al.'s (2002) photographic global tone reproduction operator. |
| `func Sharpen(img *cv.Mat, amount float64) *cv.Mat` | Sharpen applies a fixed 3x3 sharpening kernel whose strength is set by amount (0 leaves the image unchanged). |
| `func SimpleWhiteBalance(img *cv.Mat, pLow, pHigh float64) *cv.Mat` | SimpleWhiteBalance stretches each channel between the pLow and pHigh percentiles of its histogram, clipping outliers, which both corrects colour… |
| `func Stylization(img *cv.Mat, sigmaS, sigmaR float64) *cv.Mat` | Stylization produces a watercolour-like rendering: an edge-preserving filter flattens colour into smooth regions while edges are retained, giving a… |
| `func ToFloat(img *cv.Mat) []*cv.FloatMat` | ToFloat converts an 8-bit image to a slice of per-channel `cv.FloatMat` planes with samples scaled into [0,1]. |
| `func UnsharpMask(img *cv.Mat, ksize int, sigma, amount float64) *cv.Mat` | UnsharpMask sharpens an image by subtracting a Gaussian-blurred copy: out = in + amount*(in - blur). |
| `func Vignette(img *cv.Mat, strength float64) *cv.Mat` | Vignette darkens the image toward the corners with a smooth radial falloff, simulating a lens vignette. |
| `func XYZToRGB(channels []*cv.FloatMat) *cv.Mat` | XYZToRGB converts linear CIE XYZ planes (D65) back to an 8-bit sRGB image, applying the inverse linear transform and the sRGB gamma. |

### Types

| Type | What it is |
| --- | --- |
| `DragoParams` | DragoParams configures `DragoToneMap`. |
| `DurandParams` | DurandParams configures `DurandToneMap`. |
| `EdgeFilterMode` | EdgeFilterMode selects the variant used by `DomainTransformFilter`. |
| `MertensParams` | MertensParams configures `MertensFusion`. |
| `ReinhardParams` | ReinhardParams configures `ReinhardToneMap`. |

<details>
<summary><code>DragoParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultDragoParams() DragoParams` | DefaultDragoParams returns the recommended defaults for `DragoToneMap`. |

</details>

<details>
<summary><code>DurandParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultDurandParams() DurandParams` | DefaultDurandParams returns the recommended defaults for `DurandToneMap`. |

</details>

<details>
<summary><code>MertensParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultMertensParams() MertensParams` | DefaultMertensParams returns the recommended defaults for `MertensFusion`. |

</details>

<details>
<summary><code>ReinhardParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultReinhardParams() ReinhardParams` | DefaultReinhardParams returns the recommended defaults for `ReinhardToneMap`. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/photo2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
