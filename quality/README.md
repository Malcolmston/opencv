# quality — from-scratch, standard-library-only port of a useful subset of OpenCV's quality module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/quality.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/quality)

Package quality is a from-scratch, standard-library-only port of a useful
subset of OpenCV's quality module: objective image-quality assessment (IQA)
metrics.

The package sits on top of the root module github.com/malcolmston/opencv
(imported as cv) and the Go standard library (only math). It uses no cgo and
no third-party dependencies, and it does not import any of the other cv/*
subpackages. Every metric operates on the package's central image type,
`cv.Mat` (8-bit unsigned samples, one or three channels).

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/quality"
```

## Usage

This is the package's own `ExampleNewQualitySSIM`, so it compiles and its output is
asserted on every `go test ./quality/`.

```go
ref := ramp(32, 32)
	q := quality.NewQualitySSIM(ref)
	score := q.Compute(ref.Clone())
	fmt.Printf("%.3f\n", score[0])
```

```
1.000
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func BrisqueScore(img *cv.Mat) float64` | BrisqueScore returns a lightweight, no-reference BRISQUE-style heuristic for img: the variance of its mean-subtracted contrast-normalised (MSCN)… |
| `func CWSSIM(a, b *cv.Mat) float64` | CWSSIM returns the complex-wavelet structural similarity index between a and b (Wang & Simoncelli 2005). |
| `func EdgePreservationRatio(a, b *cv.Mat) float64` | EdgePreservationRatio returns the edge-preservation index of candidate b relative to reference a: the Pearson correlation coefficient between their… |
| `func Entropy(img *cv.Mat) float64` | Entropy returns the Shannon entropy, in bits, of the luminance histogram of img. |
| `func EntropyDiff(a, b *cv.Mat) float64` | EntropyDiff returns the absolute difference between the luminance `Entropy` of a and that of b. |
| `func FSIM(a, b *cv.Mat) float64` | FSIM returns the feature-similarity index between reference a and candidate b (Zhang et al. |
| `func FSIMc(a, b *cv.Mat) float64` | FSIMc returns the colour feature-similarity index between reference a and candidate b: `FSIM` augmented with a chrominance-similarity term computed… |
| `func GMSD(a, b *cv.Mat) (float64, *cv.Mat)` | GMSD returns the gradient magnitude similarity deviation between a and b and the per-pixel gradient-magnitude-similarity (GMS) map. |
| `func IWSSIM(a, b *cv.Mat) float64` | IWSSIM returns the information-content-weighted multi-scale structural similarity index between a and b (Wang & Li 2011). |
| `func LaplacianVariance(img *cv.Mat) float64` | LaplacianVariance returns the variance of the Laplacian response of img, the classic passive auto-focus / sharpness measure: a sharply focused image… |
| `func MAE(a, b *cv.Mat) []float64` | MAE returns the mean absolute error between a and b, one value per channel. |
| `func MSE(a, b *cv.Mat) []float64` | MSE returns the mean squared error between a and b, one value per channel. |
| `func MSSSIM(a, b *cv.Mat) float64` | MSSSIM returns the multi-scale structural similarity index between a and b. |
| `func NIQE(img *cv.Mat) float64` | NIQE returns the Natural Image Quality Evaluator score of img (Mittal et al. |
| `func NRMSE(a, b *cv.Mat) []float64` | NRMSE returns the normalised root-mean-squared error between a and b, one value per channel. |
| `func PIQE(img *cv.Mat) float64` | PIQE returns the Perception-based Image Quality Evaluator score of img (Venkatanath et al. |
| `func PSNR(a, b *cv.Mat) float64` | PSNR returns the peak signal-to-noise ratio between a and b in decibels, pooling the error across every channel. |
| `func RMSE(a, b *cv.Mat) []float64` | RMSE returns the root-mean-squared error between a and b, one value per channel. |
| `func SNR(a, b *cv.Mat) float64` | SNR returns the signal-to-noise ratio between reference a and candidate b in decibels, pooling over every channel. |
| `func SSIM(a, b *cv.Mat) (float64, *cv.Mat)` | SSIM returns the mean structural similarity index between a and b and the per-pixel SSIM map. |
| `func SSIMMap(a, b *cv.Mat) *cv.Mat` | SSIMMap returns the per-pixel structural-similarity map of a versus b as a single-channel `cv.Mat`, with each SSIM value clamped to [0, 1] and scaled… |
| `func Sharpness(img *cv.Mat) float64` | Sharpness is an alias for `LaplacianVariance`: it reports the variance of the Laplacian response, higher meaning sharper. |
| `func Tenengrad(img *cv.Mat) float64` | Tenengrad returns the Tenengrad focus measure of img: the mean squared Sobel gradient magnitude (gx²+gy²) over all pixels. |
| `func UQI(a, b *cv.Mat) float64` | UQI returns the universal quality index between a and b (Wang & Bovik 2002), the predecessor of SSIM. |
| `func VIF(a, b *cv.Mat) float64` | VIF returns a subband-domain approximation of visual information fidelity between reference a and candidate b. |
| `func VIFP(a, b *cv.Mat) float64` | VIFP returns the pixel-domain visual information fidelity between reference a and candidate b (Sheikh & Bovik 2006). |
| `func VSI(a, b *cv.Mat) float64` | VSI returns the visual-saliency-induced index between reference a and candidate b (Zhang et al. |

### Types

| Type | What it is |
| --- | --- |
| `QualityBase` | QualityBase is the common interface of the object-form full-reference metrics, mirroring OpenCV's cv::quality::QualityBase. |

<details>
<summary><code>QualityBase</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewQualityFSIM(ref *cv.Mat) QualityBase` | NewQualityFSIM returns a `QualityBase` whose Compute reports the `FSIM` against ref (as a one-element slice) and whose QualityMap is the per-pixel… |
| `func NewQualityFSIMc(ref *cv.Mat) QualityBase` | NewQualityFSIMc returns a `QualityBase` whose Compute reports the colour `FSIMc` against ref (as a one-element slice) and whose QualityMap is the… |
| `func NewQualityGMSD(ref *cv.Mat) QualityBase` | NewQualityGMSD returns a `QualityBase` whose Compute reports the `GMSD` against ref (as a one-element slice) and whose QualityMap is the per-pixel… |
| `func NewQualityMSE(ref *cv.Mat) QualityBase` | NewQualityMSE returns a `QualityBase` whose Compute reports the per-channel `MSE` against ref and whose QualityMap is the pooled squared-error image. |
| `func NewQualityPSNR(ref *cv.Mat) QualityBase` | NewQualityPSNR returns a `QualityBase` whose Compute reports the pooled `PSNR` against ref (as a one-element slice) and whose QualityMap is the… |
| `func NewQualityRMSE(ref *cv.Mat) QualityBase` | NewQualityRMSE returns a `QualityBase` whose Compute reports the per-channel `RMSE` against ref and whose QualityMap is the pooled squared-error… |
| `func NewQualitySSIM(ref *cv.Mat) QualityBase` | NewQualitySSIM returns a `QualityBase` whose Compute reports the mean `SSIM` against ref (as a one-element slice) and whose QualityMap is the… |
| `func NewQualityVIFP(ref *cv.Mat) QualityBase` | NewQualityVIFP returns a `QualityBase` whose Compute reports the `VIFP` against ref (as a one-element slice) and whose QualityMap is the per-pixel… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/quality).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
