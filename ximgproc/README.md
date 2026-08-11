# ximgproc — from-scratch, standard-library-only port of a useful subset of OpenCV's contrib module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/ximgproc.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/ximgproc)

Package ximgproc is a from-scratch, standard-library-only port of a useful
subset of OpenCV's contrib module ximgproc ("extended image processing"). It
is built entirely on top of the root package github.com/malcolmston/opencv
(imported here as cv) and the Go standard library; it uses no cgo, no
third-party code and it does not import any of the sibling cv/* subpackages.

The module collects edge-preserving smoothers, morphological skeletons, local
(adaptive) binarization methods and a superpixel segmenter — tools that live
outside the core OpenCV modules but are staples of practical image processing
pipelines. Every routine consumes and produces the central `cv.Mat` type
(row-major, channel-interleaved, 8-bit samples), so results compose directly
with the filters, colour conversions and thresholds in the root package.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/ximgproc"
```

## Usage

This is the package's own `ExampleAnisotropicDiffusion`, so it compiles and its output is
asserted on every `go test ./ximgproc/`.

```go
src := cv.NewMat(10, 10, 1)
	for i := range src.Data {
		src.Data[i] = 128
	}
	out := ximgproc.AnisotropicDiffusion(src, 0.2, 20, 5)
	fmt.Println(out.Data[55])
```

```
128
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AdaptiveManifoldFilter(src *cv.Mat, sigmaSpace, sigmaColor float64) *cv.Mat` | AdaptiveManifoldFilter applies an adaptive-manifold edge-preserving filter to src and returns a new Mat of the same shape. |
| `func AnisotropicDiffusion(img *cv.Mat, alpha, k float64, iters int) *cv.Mat` | AnisotropicDiffusion applies Perona–Malik anisotropic diffusion to img and returns a new Mat of the same shape. |
| `func BilateralTextureFilter(src *cv.Mat, radius, iters int) *cv.Mat` | BilateralTextureFilter applies the bilateral texture filter of Cho, Lee, Kim and Lee ("Bilateral Texture Filtering", 2014) to src and returns a new… |
| `func CovarianceEstimation(img *cv.Mat, windowRows, windowCols int) [][]float64` | CovarianceEstimation estimates the covariance matrix of the local windows of img using the sliding-window formulation of OpenCV's… |
| `func DTFilter(src, guide *cv.Mat, sigmaSpatial, sigmaColor float64, mode DTMode, iters int) *cv.Mat` | DTFilter applies the domain-transform edge-preserving filter to src, guided by the edges of guide, and returns a new Mat of the same shape as src. |
| `func FastGlobalSmootherFilter(src, guide *cv.Mat, lambda, sigmaColor float64, iters int) *cv.Mat` | FastGlobalSmootherFilter applies the fast global smoother of Min, Choi, Lu, Xu and Wang ("Fast Global Image Smoothing Based on Weighted Least… |
| `func GradientDericheX(img *cv.Mat, alpha float64) *cv.FloatMat` | GradientDericheX computes the horizontal (x-direction) image gradient of img with Deriche's recursive filter (Deriche, "Using Canny's criteria to… |
| `func GradientDericheY(img *cv.Mat, alpha float64) *cv.FloatMat` | GradientDericheY computes the vertical (y-direction) image gradient of img with Deriche's recursive filter and returns it as a signed `cv.FloatMat`. |
| `func GradientPaillouX(img *cv.Mat, alpha, omega float64) *cv.FloatMat` | GradientPaillouX computes the horizontal (x-direction) image gradient of img with the Paillou recursive filter (Philippe Paillou, "Detecting step… |
| `func GradientPaillouY(img *cv.Mat, alpha, omega float64) *cv.FloatMat` | GradientPaillouY computes the vertical (y-direction) image gradient of img with the Paillou recursive filter and returns it as a signed `cv.FloatMat`. |
| `func GuidedFilter(src, guide *cv.Mat, radius int, eps float64) *cv.Mat` | GuidedFilter applies the edge-preserving guided filter of He, Sun and Tang ("Guided Image Filtering", 2013) to src using guide as the guidance image,… |
| `func JointBilateralFilter(joint, src *cv.Mat, d int, sigmaColor, sigmaSpace float64) *cv.Mat` | JointBilateralFilter applies a cross/joint bilateral filter: the spatial and range weights are taken from the guidance image joint, but the weighted… |
| `func NiBlackThreshold(img *cv.Mat, k float64, blockSize, variant int) *cv.Mat` | NiBlackThreshold binarizes a single-channel image with a threshold computed independently for each pixel from the mean and standard deviation of its… |
| `func RollingGuidanceFilter(src *cv.Mat, d int, sigmaColor, sigmaSpace float64, iters int) *cv.Mat` | RollingGuidanceFilter applies the rolling-guidance filter of Zhang, Shen, Xu and Jia ("Rolling Guidance Filter", 2014) to src and returns a new Mat… |
| `func StructuredEdgeDetectionLite(img *cv.Mat) *cv.FloatMat` | StructuredEdgeDetectionLite estimates a per-pixel edge probability map for img and returns it as a `cv.FloatMat` with values in [0,1]. |
| `func SuperpixelLSC(img *cv.Mat, regionSize int, ratio float64) (labels *cv.Mat, n int)` | SuperpixelLSC segments img into superpixels with Linear Spectral Clustering (Li and Chen, "Superpixel Segmentation using Linear Spectral Clustering",… |
| `func SuperpixelSLIC(img *cv.Mat, regionSize int, ruler float64) (labels *cv.Mat, n int)` | SuperpixelSLIC segments img into compact, roughly equally sized superpixels using Simple Linear Iterative Clustering (Achanta et al., 2012) and… |
| `func Thinning(binary *cv.Mat) *cv.Mat` | Thinning reduces a binary image to a one-pixel-wide skeleton using the Zhang–Suen iterative thinning algorithm ("A fast parallel algorithm for… |
| `func WeightedMedianFilter(src, guide *cv.Mat, radius int, sigma float64) *cv.Mat` | WeightedMedianFilter applies a guided weighted-median filter to src and returns a new Mat of the same shape. |

### Types

| Type | What it is |
| --- | --- |
| `Affine` | Affine is a 2×3 affine transform in row-major form. |
| `Box` | Box is an axis-aligned object-proposal rectangle with a confidence Score. |
| `DTMode` | DTMode selects the filtering primitive used by `DTFilter` once the guidance image has been mapped through the edge-aware domain transform of Gastal… |
| `LineSeg` | LineSeg is a detected straight line segment with floating-point endpoints (x1,y1)–(x2,y2) in image pixel coordinates. |
| `NiBlackVariant` | NiBlackVariant selects the local-threshold formula used by `NiBlackThreshold`. |

<details>
<summary><code>Affine</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func PeiLinNormalization(img *cv.Mat) Affine` | PeiLinNormalization computes the Pei–Lin normalization transform of img: the 2×3 affine matrix that maps the image into a canonical frame in which… |
| `func (m Affine) Apply(x, y float64) (float64, float64)` | Apply transforms the point (x, y) by the affine matrix and returns (x', y'). |

</details>

<details>
<summary><code>Box</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func EdgeBoxes(edges *cv.FloatMat, maxBoxes int) []Box` | EdgeBoxes generates object bounding-box proposals from an edge-response map, following the objectness idea of Zitnick and Dollár ("Edge Boxes:… |

</details>

<details>
<summary><code>LineSeg</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FastLineDetector(img *cv.Mat) []LineSeg` | FastLineDetector detects straight line segments in img and returns them as a slice of `LineSeg`. |
| `func (s LineSeg) Length() float64` | Length returns the Euclidean length of the segment in pixels. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/ximgproc).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
