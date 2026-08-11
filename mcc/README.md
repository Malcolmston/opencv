# mcc — provides Macbeth/ColorChecker chart detection and color correction on top

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/mcc.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/mcc)

Package mcc provides Macbeth/ColorChecker chart detection and color correction
on top of the standard-library-only OpenCV port github.com/malcolmston/opencv
(imported here as cv). It mirrors a useful subset of OpenCV's mcc contrib
module: reference chart data, a CCheckerDetector-style automatic detector, and
a ColorCorrectionModel (CCM) that learns a linear least-squares mapping from a
camera's measured patch colors to the chart's reference colors and applies it
to arbitrary images.

Every image is a `cv.Mat` of 8-bit unsigned samples. Color images are
three-channel RGB (channel 0 red, 1 green, 2 blue), never BGR — matching the
root package. Patch and pixel colors are carried through the pipeline as
float64 triples in the 0..255 sRGB range so that averaging and matrix math do
not lose precision; they are only quantized back to uint8 when written into a
Mat.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/mcc"
```

## Usage

This is the package's own `ExampleCCM_Apply`, so it compiles and its output is
asserted on every `go test ./mcc/`.

```go
ref := mcc.ReferenceRGB(mcc.Macbeth24)
	measured := make([][3]float64, len(ref))
	for i, c := range ref {
		measured[i] = [3]float64{c[0] * 0.9, c[1] * 0.95, c[2] * 0.85}
	}
	model, _ := mcc.TrainCCM(measured, ref, mcc.CCMConfig{Type: mcc.CCMLinear3x3})

	img := cv.NewMat(1, 1, 3)
	img.SetPixel(0, 0, []uint8{uint8(measured[6][0]), uint8(measured[6][1]), uint8(measured[6][2])})
	out := model.Apply(img)
	fmt.Printf("channels=%d\n", out.Channels)
```

```
channels=3
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AdaptationMatrix(srcWhite, dstWhite [3]float64, method AdaptationMethod) [3][3]float64` | AdaptationMatrix returns the 3x3 matrix that adapts a CIE XYZ color measured under the source white point to its appearance under the destination… |
| `func ApplyMatrix3(m [3][3]float64, v [3]float64) [3]float64` | ApplyMatrix3 multiplies the 3x3 matrix m by the column vector v and returns the resulting 3-vector. |
| `func ChartOuterQuad(t CheckerType, patchPx, gapPx int) [4]cv.Point` | ChartOuterQuad returns the four outer corners of the patch array in a chart rendered by `RenderChart` with the same parameters, ordered top-left,… |
| `func ChromaticAdaptation(xyz, srcWhite, dstWhite [3]float64, method AdaptationMethod) [3]float64` | ChromaticAdaptation adapts a single CIE XYZ color from the source white point to the destination white point using the given `AdaptationMethod`. |
| `func DeltaE2000(lab1, lab2 [3]float64) float64` | DeltaE2000 returns the CIEDE2000 color difference between two CIE L*a*b* colors with all parametric weights kL=kC=kH=1. |
| `func DeltaE2000RGB(a, b [3]uint8) float64` | DeltaE2000RGB returns the CIEDE2000 difference between two 8-bit sRGB colors, converting both to Lab (D65) first — the perceptual analogue of… |
| `func DeltaE2000Weighted(lab1, lab2 [3]float64, kL, kC, kH float64) float64` | DeltaE2000Weighted returns the CIEDE2000 color difference with explicit parametric weighting factors kL, kC and kH (all 1 for the default metric). |
| `func DeltaE76(lab1, lab2 [3]float64) float64` | DeltaE76 returns the CIE76 color difference between two CIE L*a*b* colors: the plain Euclidean distance sqrt(dL^2 + da^2 + db^2). |
| `func DeltaE94(lab1, lab2 [3]float64) float64` | DeltaE94 returns the CIE94 color difference between two CIE L*a*b* colors using the graphic-arts weighting (kL=kC=kH=1, K1=0.045, K2=0.015). |
| `func DeltaE94Textiles(lab1, lab2 [3]float64) float64` | DeltaE94Textiles returns the CIE94 color difference under the textile application weighting (kL=2, K1=0.048, K2=0.014). |
| `func DeltaECMC(lab1, lab2 [3]float64, l, c float64) float64` | DeltaECMC returns the CMC l:c color difference between two CIE L*a*b* colors. |
| `func DeltaECMCAcceptability(lab1, lab2 [3]float64) float64` | DeltaECMCAcceptability returns the CMC(2:1) color difference, the common pass/fail acceptability tolerance used in industry. |
| `func DeltaECMCPerceptibility(lab1, lab2 [3]float64) float64` | DeltaECMCPerceptibility returns the CMC(1:1) color difference, the threshold of perceptible difference. |
| `func DeltaERGB(a, b [3]uint8) float64` | DeltaERGB returns the CIE76 Delta E between two 8-bit sRGB colors by converting both to Lab first. |
| `func DigitalSGCols() int` | DigitalSGCols returns the number of patch columns in the ColorChecker Digital SG chart (14). |
| `func DigitalSGNumPatches() int` | DigitalSGNumPatches returns the number of patches in the ColorChecker Digital SG chart (140). |
| `func DigitalSGOuterQuad(patchPx, gapPx int) [4]cv.Point` | DigitalSGOuterQuad returns the four outer corners of the 140-patch array in an image rendered by `RenderSGChart` with the same parameters, ordered… |
| `func DigitalSGReferenceLab() [][3]float64` | DigitalSGReferenceLab returns the DigitalSG reference CIE L*a*b* (D65) values in grid order. |
| `func DigitalSGReferenceRGB() [][3]float64` | DigitalSGReferenceRGB returns the DigitalSG reference sRGB values as float64 triples in the 0..255 range and grid order, ready to feed to… |
| `func DigitalSGRows() int` | DigitalSGRows returns the number of patch rows in the ColorChecker Digital SG chart (10). |
| `func GammaCompress(c, gamma float64) float64` | GammaCompress applies a pure power-law encoding (component^(1/gamma)), the inverse of `GammaExpand`. |
| `func GammaExpand(c, gamma float64) float64` | GammaExpand applies a pure power-law decoding: it raises a nonnegative component to the given gamma (mapping a gamma-encoded value to linear light),… |
| `func LChToLab(lch [3]float64) [3]float64` | LChToLab converts cylindrical CIE LCh(ab) (L*, C*, hue in degrees) back to Cartesian CIE L*a*b*. |
| `func LabToLCh(lab [3]float64) [3]float64` | LabToLCh converts CIE L*a*b* to cylindrical CIE LCh(ab): lightness L*, chroma C* = sqrt(a*^2 + b*^2) and hue angle h in degrees within [0,360). |
| `func LabToRGB(lab, white [3]float64) [3]uint8` | LabToRGB converts a CIE L*a*b* color under the given white point to an 8-bit sRGB triple. |
| `func LabToXYZ(lab, white [3]float64) [3]float64` | LabToXYZ converts a CIE L*a*b* color back to CIE XYZ under the given reference white point. |
| `func LinearRGBToXYZ(r, g, b float64) [3]float64` | LinearRGBToXYZ converts a linear-light sRGB triple (each component in [0,1], sRGB primaries, D65) to CIE XYZ with Y normalised so a diffuse white… |
| `func LinearToSRGB(c float64) float64` | LinearToSRGB compresses a single linear-light component in [0,1] back to sRGB, the inverse of `SRGBToLinear`. |
| `func LuminanceWeights(reference [][3]float64, power float64) []float64` | LuminanceWeights returns a per-sample weight vector proportional to each reference color's relative luminance raised to the given power, suitable for… |
| `func MeanDeltaE(a, b [][3]float64) float64` | MeanDeltaE returns the mean CIE76 Delta E between two equal-length lists of sRGB colors (0..255). |
| `func RGBToLab(r, g, b uint8) [3]float64` | RGBToLab converts an 8-bit sRGB color (D65) to CIE L*a*b*. |
| `func RGBToXYZ(r, g, b uint8) [3]float64` | RGBToXYZ converts an 8-bit sRGB color to CIE XYZ (D65), with Y normalised so that a diffuse white maps to Y=1. |
| `func ReferenceLab(t CheckerType) [][3]float64` | ReferenceLab returns the reference CIE L*a*b* values of the given chart in grid order. |
| `func ReferenceRGB(t CheckerType) [][3]float64` | ReferenceRGB returns the reference sRGB values of the given chart as float64 triples in the 0..255 range and grid order, ready to feed to `TrainCCM`… |
| `func RenderChart(t CheckerType, patchPx, gapPx int) *cv.Mat` | RenderChart synthesizes a canonical image of the given chart. |
| `func RenderSGChart(patchPx, gapPx int) *cv.Mat` | RenderSGChart synthesizes a canonical image of the ColorChecker Digital SG chart, drawing each of the 140 patches as a solid patchPx-by-patchPx… |
| `func SRGBToLinear(c float64) float64` | SRGBToLinear expands a single sRGB component in the range [0,1] to linear-light, inverting the sRGB transfer function (a near-2.2 gamma with a small… |
| `func XYYToXYZ(xyY [3]float64) [3]float64` | XYYToXYZ converts CIE xyY (chromaticity x,y and luminance Y) back to CIE XYZ. |
| `func XYZToLab(xyz, white [3]float64) [3]float64` | XYZToLab converts a CIE XYZ color to CIE L*a*b* under an explicit reference white point (for example `WhiteD65`, `WhiteD50` or `WhiteA`). |
| `func XYZToRGB(xyz [3]float64) [3]uint8` | XYZToRGB converts a CIE XYZ color (D65) to an 8-bit sRGB triple, applying the sRGB companding curve and saturating out-of-gamut results to [0,255]. |
| `func XYZToxyY(xyz [3]float64) [3]float64` | XYZToxyY converts CIE XYZ to CIE xyY: the (x,y) chromaticity coordinates plus the luminance Y. For a black input (X=Y=Z=0) the chromaticity is… |

### Types

| Type | What it is |
| --- | --- |
| `AdaptationMethod` | AdaptationMethod selects the cone-response model used by `ChromaticAdaptation` and `AdaptationMatrix` when converting a color from one white point to… |
| `CCM` | CCM is a fitted color-correction model mapping a device's measured colors toward a chart's reference colors. |
| `CCMConfig` | CCMConfig configures how a `CCM` is fitted and applied. |
| `CCMModelType` | CCMModelType selects the feature expansion of a `ColorCorrectionModel`. |
| `CCMType` | CCMType selects the algebraic shape of a `CCM` color-correction model. |
| `CChecker` | CChecker is a detected chart: the four image points that frame it and the measured sRGB color of every patch in grid order. |
| `CCheckerDetector` | CCheckerDetector locates a color chart of a fixed `CheckerType` in an image and samples its patches. |
| `CCheckerSG` | CCheckerSG is a sampled ColorChecker Digital SG chart: the four image points framing the 140-patch array and the measured sRGB color of every patch… |
| `CheckerType` | CheckerType selects which reference chart the detector and color model use. |
| `ColorCorrectionConfig` | ColorCorrectionConfig configures how a `ColorCorrectionModel` is fitted. |
| `ColorCorrectionModel` | ColorCorrectionModel is a fitted color-correction model mapping a device's measured colors toward a chart's reference colors. |
| `DetectorParameters` | DetectorParameters bundles the tuning knobs of the classical `CCheckerDetector` into a single, validatable configuration object, mirroring OpenCV's… |
| `Linearization` | Linearization selects how sRGB samples are decoded to a working space before the color matrix is fitted and applied. |
| `Patch` | Patch is a single reference color of a chart, holding its conventional name, its 8-bit sRGB value and the CIE L*a*b* (D65) value derived from that… |
| `PatchColorReport` | PatchColorReport describes one patch after correction: the measured, corrected and reference colors in both sRGB (0..255) and CIE L*a*b* (D65), with… |

<details>
<summary><code>CCM</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func TrainCCM(measured, reference [][3]float64, cfg CCMConfig) (*CCM, error)` | TrainCCM fits a color-correction model that maps the measured colors to the reference colors by linear least squares (solving the normal equations). |
| `func (m *CCM) Apply(img *cv.Mat) *cv.Mat` | Apply returns a color-corrected copy of a three-channel RGB image, running every pixel through the model. |
| `func (m *CCM) ApplyRGB(rgb [3]float64) [3]float64` | ApplyRGB color-corrects a single sRGB color given in the 0..255 range and returns the corrected color in the same range (unclamped to integers but… |
| `func (m *CCM) Matrix() [][3]float64` | Matrix returns a copy of the fitted coefficient matrix as numTerms rows of three columns (one column per output channel). |
| `func (m *CCM) MeanError(measured, reference [][3]float64) float64` | MeanError returns the mean CIE76 Delta E between the model's correction of the measured colors and the reference colors — the residual error after… |
| `func (m *CCM) Type() CCMType` | Type returns the model's shape. |

</details>

<details>
<summary><code>CChecker</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (c *CChecker) MaxError() float64` | MaxError returns the largest of `CChecker.PatchErrors`. |
| `func (c *CChecker) MeanError() float64` | MeanError returns the mean of `CChecker.PatchErrors`. |
| `func (c *CChecker) MeasuredRGB() [][3]float64` | MeasuredRGB returns a copy of the measured per-patch colors in grid order. |
| `func (c *CChecker) PatchErrors() []float64` | PatchErrors returns the CIE76 Delta E between each measured patch color and its reference, in grid order. |
| `func (c *CChecker) Reference() []Patch` | Reference returns the reference patches for this checker's chart type. |

</details>

<details>
<summary><code>CCheckerDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewCCheckerDetector(t CheckerType) *CCheckerDetector` | NewCCheckerDetector returns a detector for the given chart with default parameters. |
| `func (d *CCheckerDetector) Detect(img *cv.Mat) (*CChecker, bool)` | Detect locates the chart automatically and samples its patches. |
| `func (d *CCheckerDetector) DetectWithHint(img *cv.Mat, outerQuad [4]cv.Point) (*CChecker, bool)` | DetectWithHint samples the chart directly from the four outer corners of its patch array, supplied in top-left, top-right, bottom-right, bottom-left… |

</details>

<details>
<summary><code>CCheckerSG</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func SampleSGWithHint(img *cv.Mat, outerQuad [4]cv.Point) (*CCheckerSG, bool)` | SampleSGWithHint samples a ColorChecker Digital SG chart directly from the four outer corners of its patch array, supplied top-left, top-right,… |
| `func (c *CCheckerSG) MaxError() float64` | MaxError returns the largest of `CCheckerSG.PatchErrors`. |
| `func (c *CCheckerSG) MeanError() float64` | MeanError returns the mean of `CCheckerSG.PatchErrors`. |
| `func (c *CCheckerSG) MeasuredRGB() [][3]float64` | MeasuredRGB returns a copy of the measured per-patch colors in grid order. |
| `func (c *CCheckerSG) PatchErrors() []float64` | PatchErrors returns the CIE76 Delta E between each measured patch color and its DigitalSG reference, in grid order. |
| `func (c *CCheckerSG) Reference() []Patch` | Reference returns the DigitalSG reference patches. |

</details>

<details>
<summary><code>CheckerType</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (t CheckerType) Cols() int` | Cols returns the number of patch columns in the chart's grid (6 for both supported charts). |
| `func (t CheckerType) NumPatches() int` | NumPatches returns the number of patches in the chart (24 for both supported charts). |
| `func (t CheckerType) Rows() int` | Rows returns the number of patch rows in the chart's grid (4 for both supported charts). |
| `func (t CheckerType) String() string` | String returns the chart's name. |

</details>

<details>
<summary><code>ColorCorrectionModel</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func TrainColorCorrection(measured, reference [][3]float64, cfg ColorCorrectionConfig) (*ColorCorrectionModel, error)` | TrainColorCorrection fits a `ColorCorrectionModel` mapping the measured colors to the reference colors by (optionally weighted) linear least squares. |
| `func (m *ColorCorrectionModel) Infer(img *cv.Mat) *cv.Mat` | Infer returns a color-corrected copy of a three-channel RGB image, running every pixel through the model with output clamped to the 0..255 range. |
| `func (m *ColorCorrectionModel) InferRGB(rgb [3]float64) [3]float64` | InferRGB color-corrects a single sRGB color given in the 0..255 range and returns the corrected color in the same range, saturated to [0,255]. |
| `func (m *ColorCorrectionModel) Matrix() [][3]float64` | Matrix returns a copy of the fitted coefficient matrix, numTerms rows by three columns (one per output channel), in the term order documented for… |
| `func (m *ColorCorrectionModel) MeanDeltaE2000(measured, reference [][3]float64) float64` | MeanDeltaE2000 returns the mean CIEDE2000 residual between the model's correction of the measured colors and the reference colors — the… |
| `func (m *ColorCorrectionModel) MeanDeltaE76(measured, reference [][3]float64) float64` | MeanDeltaE76 returns the mean CIE76 residual between the model's correction of the measured colors and the reference colors. |
| `func (m *ColorCorrectionModel) Report(measured, reference [][3]float64) []PatchColorReport` | Report returns a per-patch `PatchColorReport` for every measured/reference pair, in input order — the detailed per-patch Lab and Delta E breakdown… |
| `func (m *ColorCorrectionModel) Type() CCMModelType` | Type returns the model's feature expansion. |
| `func (m *ColorCorrectionModel) WhiteBalanceGains() [3]float64` | WhiteBalanceGains returns the per-channel gains applied before the matrix (1,1,1 when white balancing is disabled). |

</details>

<details>
<summary><code>DetectorParameters</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDetectorParameters() *DetectorParameters` | NewDetectorParameters returns a DetectorParameters populated with the same defaults `NewCCheckerDetector` uses: they suit charts that occupy a good… |
| `func (p *DetectorParameters) NewDetector(t CheckerType) *CCheckerDetector` | NewDetector builds a `CCheckerDetector` for the given chart type using these parameters. |
| `func (p *DetectorParameters) Validate() error` | Validate reports whether the parameters are internally consistent: positive fractions, a minimum area below the maximum, and both area fractions… |

</details>

<details>
<summary><code>Patch</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DigitalSGReference() []Patch` | DigitalSGReference returns the 140 reference patches of the ColorChecker Digital SG chart in row-major grid order (patch index = row*14 + col). |
| `func ReferenceChart(t CheckerType) []Patch` | ReferenceChart returns the reference patches of the given chart in row-major grid order (patch index = row*Cols + col). |

</details>

### Variables

`WhiteD65`, `WhiteD50`, `WhiteA`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/mcc).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
