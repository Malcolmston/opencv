# xphoto — standard-library-only port of a useful subset of OpenCV's xphoto contrib module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/xphoto.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/xphoto)

Package xphoto is a standard-library-only port of a useful subset of OpenCV's
xphoto contrib module, built on top of the root cv package
(github.com/malcolmston/opencv). It collects "extra" photo-processing
algorithms that sit alongside the mainline photo module: automatic white
balance, per-channel gain application, exemplar/shift-based inpainting, oil
painting stylization and BM3D denoising. It also provides several classical
colour-constancy estimators (shades-of-gray, white-patch, gray-edge), gamma
correction, dark-channel-prior haze removal, frequency-selective
reconstruction (FSR) inpainting and Durand bilateral tone mapping.

Every function operates on the root package's `cv.Mat`, a dense row-major
matrix of 8-bit unsigned samples. Three-channel data is treated as RGB,
matching Go's image package and the root cv conventions, not OpenCV's native
BGR. Where a function is channel-agnostic it says so; where it requires a
specific channel count it validates and panics on mismatch, mirroring the root
package's fail-fast style.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/xphoto"
```

## Usage

This is the package's own `ExampleApplyChannelGains`, so it compiles and its output is
asserted on every `go test ./xphoto/`.

```go
src := solid(1, 1, 100, 100, 100)
	out := xphoto.ApplyChannelGains(src, 1.2, 1.0, 0.5)
	px := out.AtPixel(0, 0)
	fmt.Println(px[0], px[1], px[2])
```

```
120 100 50
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ApplyChannelGains(src *cv.Mat, gainR, gainG, gainB float64) *cv.Mat` | ApplyChannelGains multiplies the red, green and blue channels of src by gainR, gainG and gainB respectively, rounding and clamping each result into… |
| `func AutoWhiteBalance(src *cv.Mat) *cv.Mat` | AutoWhiteBalance white-balances src with no tuning required. |
| `func Bm3dDenoising(src *cv.Mat, h float64) *cv.Mat` | Bm3dDenoising denoises src with a single-step (hard-threshold) BM3D filter, approximating cv::xphoto::bm3dDenoising's first stage (BM3D_STEP1). |
| `func Bm3dDenoisingStep2(noisy, basic *cv.Mat, h float64) *cv.Mat` | Bm3dDenoisingStep2 runs only the empirical-Wiener second stage of BM3D (BM3D_STEP2), porting the refinement half of cv::xphoto::bm3dDenoising. |
| `func Bm3dDenoisingTwoStep(src *cv.Mat, h float64) *cv.Mat` | Bm3dDenoisingTwoStep denoises src with the full two-step BM3D pipeline, porting cv::xphoto::bm3dDenoising's default BM3D_STEPALL mode. |
| `func DctDenoising(src *cv.Mat, sigma float64, psize int) *cv.Mat` | DctDenoising denoises src with sliding-window DCT hard-thresholding, porting cv::xphoto::dctDenoising. |
| `func Dehaze(src *cv.Mat) *cv.Mat` | Dehaze removes haze from src with the dark channel prior and default parameters. |
| `func GammaCorrection(src *cv.Mat, gamma float64) *cv.Mat` | GammaCorrection applies a power-law (gamma) transfer function to every sample of src, porting cv::xphoto::gammaCorrection. |
| `func GammaLUT(gamma float64) [256]uint8` | GammaLUT returns the 256-entry lookup table used by `GammaCorrection` for the given gamma. |
| `func GrayEdgeWB(src *cv.Mat, p float64) *cv.Mat` | GrayEdgeWB white-balances src with the first-order gray-edge assumption (van de Weijer et al.): the average reflectance of scene *edges* is… |
| `func Inpaint(src, mask *cv.Mat) *cv.Mat` | Inpaint fills the masked region of src with plausible content using a SHIFTMAP-style exemplar search, approximating cv::xphoto::inpaint. |
| `func InpaintFSR(src, mask *cv.Mat, mode FSRMode) *cv.Mat` | InpaintFSR fills the masked region of src by Frequency-Selective Reconstruction, porting the FSR variant of cv::xphoto::inpaint (INPAINT_FSR_BEST /… |
| `func InpaintWithRadius(src, mask *cv.Mat, radius int) *cv.Mat` | InpaintWithRadius is `Inpaint` with an explicit exemplar patch radius. |
| `func Oilpainting(src *cv.Mat, size, dynRatio int) *cv.Mat` | Oilpainting applies an oil-painting stylization, porting cv::xphoto::oilPainting. |
| `func OilpaintingColorSpace(src *cv.Mat, size, dynRatio int, mode OilIntensity) *cv.Mat` | OilpaintingColorSpace is the parametrised oil-painting stylizer matching OpenCV's oilPainting(src, dst, size, dynRatio, colorSpace) overload: it… |
| `func ShadesOfGray(src *cv.Mat, p float64) *cv.Mat` | ShadesOfGray white-balances src using the general "shades of gray" colour constancy method (Finlayson & Trezzi, 2004): it estimates the scene… |
| `func WhitePatchWB(src *cv.Mat) *cv.Mat` | WhitePatchWB white-balances src with the white-patch (max-RGB / Retinex) assumption: the brightest value found in each channel is assumed to be the… |

### Types

| Type | What it is |
| --- | --- |
| `DarkChannelDehazer` | DarkChannelDehazer removes haze/fog from a single RGB image using He, Sun and Tang's dark channel prior. |
| `FSRMode` | FSRMode selects the quality/speed trade-off of `InpaintFSR`, mirroring OpenCV's INPAINT_FSR_BEST and INPAINT_FSR_FAST inpainting flags. |
| `GrayworldWB` | GrayworldWB implements the gray-world white-balance assumption: the average colour of a scene is achromatic (gray), so each channel is scaled by… |
| `LearningBasedWB` | LearningBasedWB approximates OpenCV's learning-based white balance. |
| `OilIntensity` | OilIntensity selects how the per-pixel intensity used for oil-painting bucketing is derived from a colour pixel. |
| `SimpleWB` | SimpleWB implements a simple per-channel white balance: each channel is stretched independently so that, after discarding the darkest and brightest P… |
| `TonemapDurand` | TonemapDurand implements Durand and Dorsey's fast bilateral-filtering tone mapping operator, porting cv::xphoto::TonemapDurand (historically… |
| `WhiteBalancer` | WhiteBalancer is the common interface implemented by every white-balance algorithm in this package, mirroring OpenCV's cv::xphoto::WhiteBalancer. |

<details>
<summary><code>DarkChannelDehazer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDarkChannelDehazer() *DarkChannelDehazer` | NewDarkChannelDehazer returns a DarkChannelDehazer with the parameters recommended by He et al.: patch radius 3, omega 0.95, transmission floor 0.1,… |
| `func (d *DarkChannelDehazer) Apply(src *cv.Mat) *cv.Mat` | Apply runs the dark channel prior dehazing on src and returns the recovered haze-free image. |

</details>

<details>
<summary><code>GrayworldWB</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateGrayworldWB() *GrayworldWB` | CreateGrayworldWB returns a new `GrayworldWB` with OpenCV's default parameters, mirroring cv::xphoto::createGrayworldWB. |
| `func NewGrayworldWB() *GrayworldWB` | NewGrayworldWB returns a GrayworldWB with a default saturation threshold of 0.98 (OpenCV's default). |
| `func (g *GrayworldWB) BalanceWhite(src *cv.Mat) *cv.Mat` | BalanceWhite scales each channel of src so their means match. |
| `func (g *GrayworldWB) GetSaturationThreshold() float64` | GetSaturationThreshold returns the saturation threshold. |
| `func (g *GrayworldWB) SetSaturationThreshold(v float64)` | SetSaturationThreshold sets the (max-min)/max saturation above which a pixel is excluded from the gray-world statistics. |

</details>

<details>
<summary><code>LearningBasedWB</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateLearningBasedWB() *LearningBasedWB` | CreateLearningBasedWB returns a new `LearningBasedWB` with OpenCV's default parameters, mirroring cv::xphoto::createLearningBasedWB. |
| `func NewLearningBasedWB() *LearningBasedWB` | NewLearningBasedWB returns a LearningBasedWB with defaults matching OpenCV's public API (range 255, saturation threshold ~0.98, 64 histogram bins). |
| `func (l *LearningBasedWB) BalanceWhite(src *cv.Mat) *cv.Mat` | BalanceWhite estimates the illuminant of src and applies neutralising gains. |
| `func (l *LearningBasedWB) GetHistBinNum() int` | GetHistBinNum returns the chromaticity quantisation resolution. |
| `func (l *LearningBasedWB) GetRangeMaxVal() float64` | GetRangeMaxVal returns the maximum sample value. |
| `func (l *LearningBasedWB) GetSaturationThreshold() float64` | GetSaturationThreshold returns the saturation threshold fraction. |
| `func (l *LearningBasedWB) SetHistBinNum(v int)` | SetHistBinNum sets the internal chromaticity quantisation resolution. |
| `func (l *LearningBasedWB) SetRangeMaxVal(v float64)` | SetRangeMaxVal sets the maximum sample value (255 for 8-bit data). |
| `func (l *LearningBasedWB) SetSaturationThreshold(v float64)` | SetSaturationThreshold sets the fraction of RangeMaxVal above which a pixel's maximum channel is treated as clipped and excluded. |

</details>

<details>
<summary><code>SimpleWB</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateSimpleWB() *SimpleWB` | CreateSimpleWB returns a new `SimpleWB` with OpenCV's default parameters. |
| `func NewSimpleWB() *SimpleWB` | NewSimpleWB returns a SimpleWB with OpenCV's default parameters: input and output ranges of [0,255] and a 2% clip on each tail. |
| `func (s *SimpleWB) BalanceWhite(src *cv.Mat) *cv.Mat` | BalanceWhite stretches each channel of src independently. |
| `func (s *SimpleWB) GetInputMax() float64` | GetInputMax returns the upper bound of the input range. |
| `func (s *SimpleWB) GetInputMin() float64` | GetInputMin returns the lower bound of the input range. |
| `func (s *SimpleWB) GetOutputMax() float64` | GetOutputMax returns the upper bound of the output range. |
| `func (s *SimpleWB) GetOutputMin() float64` | GetOutputMin returns the lower bound of the output range. |
| `func (s *SimpleWB) GetP() float64` | GetP returns the tail-clip percentage. |
| `func (s *SimpleWB) SetInputMax(v float64)` | SetInputMax sets the upper bound of the input range. |
| `func (s *SimpleWB) SetInputMin(v float64)` | SetInputMin sets the lower bound of the input range considered when building each channel histogram. |
| `func (s *SimpleWB) SetOutputMax(v float64)` | SetOutputMax sets the upper bound of the output range. |
| `func (s *SimpleWB) SetOutputMin(v float64)` | SetOutputMin sets the lower bound of the output range each channel is stretched into. |
| `func (s *SimpleWB) SetP(v float64)` | SetP sets the tail-clip percentage (0..50) discarded from each channel before computing its stretch bounds. |

</details>

<details>
<summary><code>TonemapDurand</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTonemapDurand() *TonemapDurand` | NewTonemapDurand returns a TonemapDurand with OpenCV's default parameters: gamma 1.0, contrast 4.0, saturation 1.0, sigmaSpace 2.0 and sigmaColor 2.0. |
| `func (t *TonemapDurand) Process(src *cv.Mat) *cv.Mat` | Process tone-maps src and returns a new image of the same shape. |

</details>

### Constants

`DctDenoisingDefaultPatch`, `InpaintDefaultRadius`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/xphoto).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
