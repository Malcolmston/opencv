# colorspaces2 — provides extended colour-science utilities that build on the root cv package's `cv.Mat`

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/colorspaces2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/colorspaces2)

Package colorspaces2 provides extended colour-science utilities that build on
the root cv package's `cv.Mat` image type.

- Scalar colour values and conversions. The small value types `RGB`, `HSV`,
`HSL`, `XYZ`, `Lab`, `Luv`, `YCbCr`, `YUV` and `CMYK` hold one colour in
floating-point form, and the RGBToX / XToRGB functions convert between them
with the standard CIE / ITU formulae. RGB values are sRGB, gamma-encoded and
normalised to [0,1]; the exact ranges of the other spaces are documented on
each type.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/colorspaces2"
```

## Usage

This is the package's own `ExampleApplyGamma`, so it compiles and its output is
asserted on every `go test ./colorspaces2/`.

```go
c := colorspaces2.ApplyGamma(colorspaces2.RGB{R: 0.25, G: 0.25, B: 0.25}, 2)
	fmt.Printf("%.2f %.2f %.2f\n", c.R, c.G, c.B)
```

```
0.50 0.50 0.50
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AdjustColorTemperatureMat(src *cv.Mat, kelvin float64) *cv.Mat` | AdjustColorTemperatureMat returns a new Mat that re-balances src as though the scene white were shifted from a neutral reference to the target colour… |
| `func ApplyChannelGains(src *cv.Mat, gains [3]float64) *cv.Mat` | ApplyChannelGains returns a new Mat in which channel c of every pixel of src is multiplied by gains`c` and clamped to [0,255]. |
| `func ApplyLUTMat(src *cv.Mat, lut [256]uint8) *cv.Mat` | ApplyLUTMat returns a copy of src with the per-channel lookup table lut applied to every sample. |
| `func ApplyPalette(src *cv.Mat, p Palette) *cv.Mat` | ApplyPalette returns a new Mat in which every pixel of src is replaced by its nearest colour in p. |
| `func BuildGammaLUT(gamma float64) [256]uint8` | BuildGammaLUT returns a 256-entry lookup table mapping an 8-bit input to the power-law gamma-corrected 8-bit output out =… |
| `func ChannelMaxima(src *cv.Mat) [3]uint8` | ChannelMaxima returns the maximum sample value of each of the first three channels of src, on the 8-bit [0,255] scale. |
| `func ChannelMeans(src *cv.Mat) [3]float64` | ChannelMeans returns the mean sample value of each of the first three channels of src, on the 8-bit [0,255] scale. |
| `func ChromaticAdaptMat(src *cv.Mat, srcWhite, dstWhite XYZ) *cv.Mat` | ChromaticAdaptMat returns a new Mat in which every RGB pixel of src has been Bradford-adapted from srcWhite to dstWhite (converting through XYZ). |
| `func CorrelatedColorTemperature(c XYZ) float64` | CorrelatedColorTemperature estimates the correlated colour temperature (in kelvin) of an `XYZ` colour using McCamy's cubic approximation. |
| `func DelinearizeMat(src *cv.Mat) *cv.Mat` | DelinearizeMat returns a new Mat in which each linear-light sample of src has been converted back to sRGB encoding, inverting `LinearizeMat`. |
| `func GammaCorrectMat(src *cv.Mat, gamma float64) *cv.Mat` | GammaCorrectMat returns a new Mat with power-law gamma correction applied to every sample of src (out = 255*(in/255)^(1/gamma)). |
| `func GrayWorldGains(src *cv.Mat) [3]float64` | GrayWorldGains returns the per-channel gains implied by the gray-world assumption: each channel is scaled so its mean equals the average of the three… |
| `func GrayWorldWhiteBalance(src *cv.Mat) *cv.Mat` | GrayWorldWhiteBalance returns a white-balanced copy of src under the gray-world assumption, scaling each channel so its mean matches the overall mean. |
| `func LinearToSRGB(c float64) float64` | LinearToSRGB converts a single linear-light component in [0,1] to its gamma-encoded sRGB value, inverting `SRGBToLinear`. |
| `func LinearizeMat(src *cv.Mat) *cv.Mat` | LinearizeMat returns a new Mat in which each sample of the sRGB-encoded src has been converted to linear light and rescaled to [0,255]. |
| `func NearestColorIndex(p Palette, c RGB) int` | NearestColorIndex returns the index of the palette entry closest to c in squared Euclidean RGB distance, or -1 for an empty palette. |
| `func Posterize(src *cv.Mat, levels int) *cv.Mat` | Posterize returns a new Mat in which each channel of src is reduced to the given number of levels, snapping to the nearest of levels values spread… |
| `func QuantizeUniform(src *cv.Mat, levels int) *cv.Mat` | QuantizeUniform returns a new Mat in which each channel of src is quantised to the given number of evenly spaced levels. |
| `func SRGBToLinear(c float64) float64` | SRGBToLinear converts a single gamma-encoded sRGB component in [0,1] to its linear-light value using the piecewise sRGB electro-optical transfer… |
| `func WhitePatchGains(src *cv.Mat, percentile float64) [3]float64` | WhitePatchGains returns the per-channel gains implied by the white-patch (max-RGB) assumption at the given percentile in [0,1]: each channel is… |
| `func WhitePatchWhiteBalance(src *cv.Mat) *cv.Mat` | WhitePatchWhiteBalance returns a white-balanced copy of src using the white-patch assumption, scaling each channel so its maximum maps to 255. |
| `func WhitePatchWhiteBalancePercentile(src *cv.Mat, percentile float64) *cv.Mat` | WhitePatchWhiteBalancePercentile returns a white-balanced copy of src using the white-patch assumption evaluated at the given percentile in [0,1]. |

### Types

| Type | What it is |
| --- | --- |
| `CMYK` | CMYK is a subtractive cyan/magenta/yellow/black colour with all four components in [0,1]. |
| `HSL` | HSL is a colour in the hue/saturation/lightness cylinder. |
| `HSV` | HSV is a colour in the hue/saturation/value cylinder. |
| `Lab` | Lab is a colour in CIE L*a*b* space. |
| `Luv` | Luv is a colour in CIE L*u*v* space. |
| `Matrix3` | Matrix3 is a 3x3 matrix of float64 in row-major order, used for the linear colour transforms in this package (RGB<->XYZ and chromatic adaptation). |
| `Palette` | Palette is an ordered set of representative colours, typically extracted from an image by `MedianCut`, `KMeansQuantize` or `ExtractPalette`. |
| `RGB` | RGB is an sRGB colour with gamma-encoded components normalised to [0,1]. |
| `XYZ` | XYZ is a colour in the CIE 1931 XYZ space using the sRGB primaries. |
| `YCbCr` | YCbCr is a luma/chroma colour using the ITU-R BT.601 full-range (JPEG) coefficients. |
| `YUV` | YUV is an analog luma/chroma colour using the ITU-R BT.601 weights. |

<details>
<summary><code>CMYK</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func RGBToCMYK(c RGB) CMYK` | RGBToCMYK converts a gamma-encoded `RGB` colour to `CMYK`. |

</details>

<details>
<summary><code>HSL</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func RGBToHSL(c RGB) HSL` | RGBToHSL converts a gamma-encoded `RGB` colour to `HSL`. |

</details>

<details>
<summary><code>HSV</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func RGBToHSV(c RGB) HSV` | RGBToHSV converts a gamma-encoded `RGB` colour to `HSV`. |

</details>

<details>
<summary><code>Lab</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func RGBToLab(c RGB) Lab` | RGBToLab converts a gamma-encoded sRGB colour to `Lab` (via `XYZ`). |
| `func XYZToLab(c XYZ) Lab` | XYZToLab converts a CIE `XYZ` colour to `Lab` using the D65 reference white. |

</details>

<details>
<summary><code>Luv</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func RGBToLuv(c RGB) Luv` | RGBToLuv converts a gamma-encoded sRGB colour to `Luv` (via `XYZ`). |
| `func XYZToLuv(c XYZ) Luv` | XYZToLuv converts a CIE `XYZ` colour to `Luv` using the D65 reference white. |

</details>

<details>
<summary><code>Matrix3</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BradfordMatrix(srcWhite, dstWhite XYZ) Matrix3` | BradfordMatrix returns the Bradford chromatic-adaptation matrix that maps XYZ colours viewed under the source white to their appearance under the… |
| `func VonKriesMatrix(srcWhite, dstWhite XYZ) Matrix3` | VonKriesMatrix returns the von Kries (HPE cone) chromatic-adaptation matrix mapping from the source white to the destination white. |
| `func (m Matrix3) Inverse() (Matrix3, bool)` | Inverse returns the matrix inverse and true, or the zero matrix and false if the matrix is singular. |
| `func (m Matrix3) Mul(n Matrix3) Matrix3` | Mul returns the matrix product m*n. |
| `func (m Matrix3) MulVec(v XYZ) XYZ` | MulVec multiplies the matrix by the column vector (v.X, v.Y, v.Z) and returns the resulting `XYZ` triple. |
| `func (m Matrix3) Transpose() Matrix3` | Transpose returns the transpose of the matrix. |

</details>

<details>
<summary><code>Palette</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ExtractPalette(src *cv.Mat, n int) Palette` | ExtractPalette returns the n most representative colours of src using `KMeansQuantize` with a fixed seed and a modest iteration budget, ordered by… |
| `func KMeansQuantize(src *cv.Mat, k, iterations int, seed uint64) (Palette, []int)` | KMeansQuantize clusters the pixels of src into k colours with Lloyd's algorithm and returns the cluster-centre palette together with a per-pixel… |
| `func MedianCut(src *cv.Mat, n int) Palette` | MedianCut extracts a palette of up to n colours from src using the classic median-cut algorithm: the colour box with the longest axis is repeatedly… |
| `func (p Palette) Nearest(c RGB) int` | Nearest returns the index of the palette entry closest to c in squared Euclidean RGB distance, or -1 if the palette is empty. |
| `func (p Palette) ToRGBA() []color.RGBA` | ToRGBA converts the palette to a slice of standard-library `color.RGBA` values, each fully opaque. |

</details>

<details>
<summary><code>RGB</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ApplyGamma(c RGB, gamma float64) RGB` | ApplyGamma applies a simple power-law gamma curve to each channel of c, returning out = in^(1/gamma). |
| `func CMYKToRGB(c CMYK) RGB` | CMYKToRGB converts a `CMYK` colour back to gamma-encoded `RGB`. |
| `func DominantColor(src *cv.Mat) RGB` | DominantColor returns the single most representative colour of src as the mean of its pixels in linear proportion (equivalent to a one-colour… |
| `func HSLToRGB(c HSL) RGB` | HSLToRGB converts an `HSL` colour back to gamma-encoded `RGB`. |
| `func HSVToRGB(c HSV) RGB` | HSVToRGB converts an `HSV` colour back to gamma-encoded `RGB`. |
| `func KelvinToRGB(kelvin float64) RGB` | KelvinToRGB converts a correlated colour temperature in kelvin to an approximate display `RGB` using the widely used Tanner Helland approximation. |
| `func LabToRGB(c Lab) RGB` | LabToRGB converts a `Lab` colour to gamma-encoded sRGB (via `XYZ`). |
| `func LuvToRGB(c Luv) RGB` | LuvToRGB converts a `Luv` colour to gamma-encoded sRGB (via `XYZ`). |
| `func NewRGBFromUint8(r, g, b uint8) RGB` | NewRGBFromUint8 builds an `RGB` from three 8-bit samples, dividing each by 255 so that 0 maps to 0.0 and 255 maps to 1.0. |
| `func XYZToRGB(c XYZ) RGB` | XYZToRGB converts a CIE `XYZ` colour to gamma-encoded sRGB, applying the sRGB transfer function after the linear transform. |
| `func YCbCrToRGB(c YCbCr) RGB` | YCbCrToRGB converts a full-range (JPEG) `YCbCr` colour back to `RGB`. |
| `func YUVToRGB(c YUV) RGB` | YUVToRGB converts an analog `YUV` colour back to `RGB`. |
| `func (c RGB) Clamp() RGB` | Clamp returns a copy of c with every component limited to the [0,1] range. |
| `func (c RGB) Luma() float64` | Luma returns the ITU-R BT.601 luma of c (0.299R + 0.587G + 0.114B), computed on the gamma-encoded components. |
| `func (c RGB) ToRGBA() color.RGBA` | ToRGBA converts c to a fully opaque `color.RGBA` from the standard library. |
| `func (c RGB) ToUint8() (r, g, b uint8)` | ToUint8 clamps c to [0,1] and returns the three components scaled to 8-bit integers with round-half-away-from-zero rounding. |

</details>

<details>
<summary><code>XYZ</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BradfordAdapt(c, srcWhite, dstWhite XYZ) XYZ` | BradfordAdapt adapts a single `XYZ` colour from the source white point to the destination white point using the Bradford transform. |
| `func LabToXYZ(c Lab) XYZ` | LabToXYZ converts a `Lab` colour to CIE `XYZ` using the D65 reference white. |
| `func LuvToXYZ(c Luv) XYZ` | LuvToXYZ converts a `Luv` colour to CIE `XYZ` using the D65 reference white. |
| `func PlanckianXYZ(kelvin float64) XYZ` | PlanckianXYZ returns the CIE `XYZ` chromaticity (with Y normalised to 1.0) of a black-body radiator at the given temperature in kelvin, using Kim et… |
| `func RGBToXYZ(c RGB) XYZ` | RGBToXYZ converts a gamma-encoded sRGB colour to CIE `XYZ`, linearising the sRGB transfer function first. |
| `func VonKriesAdapt(c, srcWhite, dstWhite XYZ) XYZ` | VonKriesAdapt adapts a single `XYZ` colour from the source white point to the destination white point using the von Kries transform. |

</details>

<details>
<summary><code>YCbCr</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func RGBToYCbCr(c RGB) YCbCr` | RGBToYCbCr converts a gamma-encoded `RGB` colour to full-range (JPEG) `YCbCr` using the ITU-R BT.601 luma coefficients. |

</details>

<details>
<summary><code>YUV</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func RGBToYUV(c RGB) YUV` | RGBToYUV converts a gamma-encoded `RGB` colour to analog `YUV` using the ITU-R BT.601 weights. |

</details>

### Variables

`WhitePointD65`, `WhitePointD50`, `WhitePointA`, `WhitePointE`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/colorspaces2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
