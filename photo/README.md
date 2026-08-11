# photo — standard-library-only computational-photography toolkit built on top of the root cv

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/photo.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/photo)

Package photo is a standard-library-only computational-photography toolkit
built on top of the root cv package (github.com/malcolmston/opencv). It ports
a useful subset of OpenCV's photo module: edge-aware denoising (single- and
multi-frame, plus total-variation), region inpainting, edge-preserving
smoothing, detail enhancement, stylization, pencil-sketch and cartoon
rendering, oil painting, contrast-preserving decolorization, tone and
white-balance correction, and Poisson image editing (seamless cloning, colour
change, illumination change and texture flattening).

The package operates on the root package's `cv.Mat`, a dense row-major matrix
of 8-bit unsigned samples. Single-channel (grayscale) and three-channel (RGB)
images are the common cases; where a function is channel-agnostic it says so.
Three-channel data is treated as RGB, matching Go's image package and the root
cv conventions, not OpenCV's native BGR.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/photo"
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func Cartoonify(img *cv.Mat, sigmaS, sigmaR float64) *cv.Mat` | Cartoonify gives img a cartoon look by combining edge-preserving colour smoothing with bold dark outlines. |
| `func ColorChange(src, mask *cv.Mat, redMul, greenMul, blueMul float64) *cv.Mat` | ColorChange multiplies the colour of the masked region of src by per-channel factors while keeping its texture, using Poisson image editing. |
| `func Decolor(img *cv.Mat) (gray *cv.Mat, boost *cv.Mat)` | Decolor converts a three-channel RGB image to grayscale in a contrast-preserving way and also returns a colour-boosted version of the input. |
| `func DenoiseTVL1(observations []*cv.Mat, lambda float64, niters int) *cv.Mat` | DenoiseTVL1 denoises a set of noisy observations of the same scene using the total-variation L1 (TV-L1) model, minimising the total variation of the… |
| `func DetailEnhance(img *cv.Mat, sigmaS, sigmaR float64) *cv.Mat` | DetailEnhance increases local contrast and texture while preserving edges. |
| `func DomainTransformFilter(img *cv.Mat, flags EdgePreservingFlag, sigmaS, sigmaR float64) *cv.Mat` | DomainTransformFilter is an edge-preserving smoothing filter based on the domain transform of Gastal and Oliveira (2011), the algorithm OpenCV uses… |
| `func EdgePreservingFilter(img *cv.Mat, flags EdgePreservingFlag, sigmaS, sigmaR float64) *cv.Mat` | EdgePreservingFilter smooths img while keeping strong edges crisp. |
| `func FastNlMeansDenoising(img *cv.Mat, h float64, templateWin, searchWin int) *cv.Mat` | FastNlMeansDenoising removes noise with the non-local means algorithm: each output sample is a weighted average of the samples in a… |
| `func FastNlMeansDenoisingColored(img *cv.Mat, h, hColor float64, templateWin, searchWin int) *cv.Mat` | FastNlMeansDenoisingColored denoises a three-channel RGB image, decoupling luminance from colour. |
| `func FastNlMeansDenoisingColoredMulti(imgs []*cv.Mat, imgToDenoiseIndex, temporalWindowSize int, h, hColor float64, templateWin, searchWin int) *cv.Mat` | FastNlMeansDenoisingColoredMulti is the colour, multi-frame denoiser: it converts every frame to Y'CrCb, denoises the target frame's luma plane… |
| `func FastNlMeansDenoisingMulti(imgs []*cv.Mat, imgToDenoiseIndex, temporalWindowSize int, h float64, templateWin, searchWin int) *cv.Mat` | FastNlMeansDenoisingMulti denoises the frame imgs`imgToDenoiseIndex` using a temporal window of temporalWindowSize consecutive frames centred on it. |
| `func GammaCorrection(img *cv.Mat, gamma float64) *cv.Mat` | GammaCorrection applies a power-law (gamma) tone curve to img: each sample s is remapped to 255*(s/255)^gamma. |
| `func GrayWorldWhiteBalance(img *cv.Mat) *cv.Mat` | GrayWorldWhiteBalance corrects a colour cast under the gray-world assumption: the average colour of a scene should be neutral gray. |
| `func HistogramStretch(img *cv.Mat) *cv.Mat` | HistogramStretch performs per-channel linear contrast stretching (normalisation): for every channel the darkest sample is mapped to 0 and the… |
| `func IlluminationChange(src, mask *cv.Mat, alpha, beta float64) *cv.Mat` | IlluminationChange locally relights the masked region of src by compressing its gradient field, following the Poisson local-illumination-change edit… |
| `func Inpaint(img, mask *cv.Mat, radius float64, method InpaintMethod) *cv.Mat` | Inpaint reconstructs the pixels of img marked by mask from the surrounding, unmasked pixels. |
| `func OilPainting(img *cv.Mat, size, dynRatio int) *cv.Mat` | OilPainting applies an oil-painting stylisation, the algorithm from OpenCV's xphoto module. |
| `func PencilSketch(img *cv.Mat, sigmaS, sigmaR, shadeFactor float64) (gray *cv.Mat, color *cv.Mat)` | PencilSketch renders img as a pencil drawing, returning both a grayscale sketch and a colour-tinted sketch, mirroring OpenCV's pencilSketch. |
| `func SeamlessClone(src, dst, mask *cv.Mat, center image.Point, mode SeamlessCloneMode) *cv.Mat` | SeamlessClone blends the masked part of src into dst using Poisson image editing, returning a new image (dst is not modified). |
| `func SimpleWhiteBalance(img *cv.Mat, pLow, pHigh float64) *cv.Mat` | SimpleWhiteBalance white-balances img by stretching each channel between robust percentiles, the method of OpenCV's SimpleWB. |
| `func Stylization(img *cv.Mat, sigmaS, sigmaR float64) *cv.Mat` | Stylization gives img a smoothed, cartoon-like look: it edge-preservingly smooths the image and then darkens it in proportion to the local gradient… |
| `func TextureFlattening(src, mask *cv.Mat, lowThreshold, highThreshold float64, kernelSize int) *cv.Mat` | TextureFlattening washes out fine texture inside the masked region of src while preserving its major edges, by importing only the source gradients… |
| `func UnsharpMask(img *cv.Mat, ksize int, sigma, amount float64) *cv.Mat` | UnsharpMask sharpens img with the classic unsharp-masking technique: it forms a Gaussian-blurred copy (the "unsharp" mask), takes the high-frequency… |

### Types

| Type | What it is |
| --- | --- |
| `EdgePreservingFlag` | EdgePreservingFlag selects the edge-preserving filter variant. |
| `InpaintMethod` | InpaintMethod selects the region-filling algorithm used by `Inpaint`. |
| `SeamlessCloneMode` | SeamlessCloneMode selects how the source guidance gradients are chosen in `SeamlessClone`. |

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/photo).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
