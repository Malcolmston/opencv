# superres — pure-Go, standard-library-only toolkit for image super-resolution

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/superres.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/superres)

Package superres is a pure-Go, standard-library-only toolkit for image
super-resolution, high-quality interpolation and the supporting building
blocks those techniques need. It is a subpackage of the OpenCV port
github.com/malcolmston/opencv (imported here as cv) and operates on the
library's central `cv.Mat` type — a dense, row-major, channel-interleaved
matrix of 8-bit samples — rather than defining any competing image type.

Nothing in this package uses cgo, a GPU, trained model files or any
third-party dependency: every algorithm is a classical, deterministic,
weight-free method implemented against the Go standard library (math, sort and
the cv core). Given the same input, every function returns the same output on
every run and platform.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/superres"
```

## Usage

This is the package's own `ExampleEstimateShift`, so it compiles and its output is
asserted on every `go test ./superres/`.

```go
ref := checkerboard(32, 8)
	moved := superres.Shift(ref, 0.4, -0.3)

	s := superres.EstimateShift(ref, moved)
	fmt.Printf("dx=%.1f dy=%.1f\n", s.Dx, s.Dy)
```

```
dx=0.4 dy=-0.3
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AdaptiveUnsharpMask(src *cv.Mat, sigma, amount, edgeScale float64) *cv.Mat` | AdaptiveUnsharpMask is like `UnsharpMask` but scales the sharpening amount by the local gradient magnitude, so strong edges are sharpened more than… |
| `func BackProjectionSR(low *cv.Mat, scale, iterations int) *cv.Mat` | BackProjectionSR is a convenience wrapper that runs `IterativeBackProjection` with `DefaultBackProjectionParams` for the given integer scale and… |
| `func BicubicResize(src *cv.Mat, width, height int) *cv.Mat` | BicubicResize resizes src to width×height using the Catmull-Rom bicubic kernel, the recommended general-purpose interpolator for photographic… |
| `func BilinearResize(src *cv.Mat, width, height int) *cv.Mat` | BilinearResize resizes src to width×height by bilinear interpolation. |
| `func BoxDownscale(src *cv.Mat, factor int) *cv.Mat` | BoxDownscale shrinks src by an integer factor using box (average) pooling: each output sample is the mean of a factor×factor block of source samples. |
| `func EdgeDirectedResize(src *cv.Mat, width, height int) *cv.Mat` | EdgeDirectedResize upscales src to an arbitrary width×height using edge- directed interpolation: it applies `NEDIDouble` until both dimensions reach… |
| `func ExampleFreeSR(low *cv.Mat, scale, iterations int) *cv.Mat` | ExampleFreeSR performs single-image, example-free super-resolution at an integer scale. |
| `func FuseAverage(frames []*cv.Mat, shifts []Shift2D) *cv.Mat` | FuseAverage aligns each frame back onto the reference grid by its (negated) shift and averages them, producing a denoised image at the original… |
| `func FuseMedian(frames []*cv.Mat, shifts []Shift2D) *cv.Mat` | FuseMedian is like `FuseAverage` but combines the aligned frames by their per-sample median instead of the mean, which rejects outliers (moving… |
| `func GaussianBlur(src *cv.Mat, sigma float64) *cv.Mat` | GaussianBlur convolves src with a separable Gaussian of standard deviation sigma, replicating the border. |
| `func GaussianDownscale(src *cv.Mat, factor int, sigma float64) *cv.Mat` | GaussianDownscale shrinks src by an integer factor after a Gaussian anti-aliasing blur, the correct way to downsample without introducing aliasing. |
| `func GaussianKernel1D(sigma float64) []float64` | GaussianKernel1D returns a normalised 1-D Gaussian kernel for the given standard deviation sigma. |
| `func GradientProfileSharpen(src *cv.Mat, strength float64) *cv.Mat` | GradientProfileSharpen sharpens the edges of an (already upscaled) image by steepening its gradient profile, the core idea behind… |
| `func IntegerShift(src *cv.Mat, dx, dy int) *cv.Mat` | IntegerShift translates src by a whole number of pixels (dx, dy) with border replication, a fast exact special case of `Shift`. |
| `func IterativeBackProjection(low *cv.Mat, params BackProjectionParams) *cv.Mat` | IterativeBackProjection performs Irani-Peleg iterative back-projection super-resolution on a single low-resolution image. |
| `func LanczosResize(src *cv.Mat, width, height, a int) *cv.Mat` | LanczosResize resizes src to width×height using an a-lobe Lanczos kernel. |
| `func MAE(a, b *cv.Mat) float64` | MAE returns the mean absolute error between two images of identical shape, averaged over all samples. |
| `func MSE(a, b *cv.Mat) float64` | MSE returns the mean squared error between two images of identical shape, averaged over all samples (rows·cols·channels). |
| `func MitchellResize(src *cv.Mat, width, height int) *cv.Mat` | MitchellResize resizes src to width×height using the Mitchell-Netravali kernel (b = c = 1/3), a balanced compromise between sharpness and ringing. |
| `func MultiFrameSR(frames []*cv.Mat, scale int) *cv.Mat` | MultiFrameSR is the recommended end-to-end multi-frame super-resolution pipeline: it registers the frames with `RegisterFrames`, fuses them onto a… |
| `func NEDI(src *cv.Mat, scale int) *cv.Mat` | NEDI upscales src by an integer power-of-two scale (2, 4, 8, …) by applying `NEDIDouble` repeatedly. |
| `func NEDIDouble(src *cv.Mat) *cv.Mat` | NEDIDouble upscales src by exactly 2× using New Edge-Directed Interpolation (Li & Orchard). |
| `func NearestResize(src *cv.Mat, width, height int) *cv.Mat` | NearestResize resizes src to width×height by nearest-neighbour sampling. |
| `func PSNR(a, b *cv.Mat) float64` | PSNR returns the peak signal-to-noise ratio in decibels between two 8-bit images of identical shape (peak value 255). |
| `func ProgressiveUpscale(src *cv.Mat, scale float64, stepSharpen float64) *cv.Mat` | ProgressiveUpscale enlarges src to the target integer scale in small geometric steps rather than a single jump, sharpening after each step. |
| `func Resize(src *cv.Mat, width, height int, interp Interpolation) *cv.Mat` | Resize resizes src to width×height using the named interpolation. |
| `func ResizeKernel(src *cv.Mat, width, height int, k ResampleKernel) *cv.Mat` | ResizeKernel resizes src to width×height using the given separable kernel. |
| `func ResizeScale(src *cv.Mat, factor float64, interp Interpolation) *cv.Mat` | ResizeScale resizes src by an isotropic floating-point factor using the named interpolation. |
| `func SSIM(a, b *cv.Mat, window int) float64` | SSIM returns the mean structural similarity index between two grayscale or multi-channel images of identical shape. |
| `func SharpenLaplacian(src *cv.Mat, strength float64) *cv.Mat` | SharpenLaplacian sharpens src by subtracting a scaled discrete Laplacian: out = src − strength·∇²src, using the 4-neighbour Laplacian. |
| `func Shift(src *cv.Mat, dx, dy float64) *cv.Mat` | Shift translates src by the sub-pixel offset (dx, dy) using bilinear resampling with border replication. |
| `func ShiftAndAddSR(frames []*cv.Mat, shifts []Shift2D, scale int) *cv.Mat` | ShiftAndAddSR fuses several low-resolution frames of a static scene into a single high-resolution image by the shift-and-add method: it splats every… |
| `func Sinc(x float64) float64` | Sinc returns the normalised sinc function sin(pi*x)/(pi*x), with Sinc(0)=1. |
| `func SplineResize(src *cv.Mat, width, height int) *cv.Mat` | SplineResize resizes src to width×height using the smooth cubic B-spline kernel, giving a ring-free (slightly soft) result. |
| `func UnsharpMask(src *cv.Mat, sigma, amount, threshold float64) *cv.Mat` | UnsharpMask sharpens src with the classic unsharp-mask operator: the output is src + amount·(src − Gaussian(src, sigma)), applied per channel. |
| `func UpscaleAndSharpen(src *cv.Mat, width, height int, interp Interpolation, sigma, amount float64) *cv.Mat` | UpscaleAndSharpen enlarges src to width×height with the named interpolation and then applies an unsharp mask, the usual remedy for the softness that… |

### Types

| Type | What it is |
| --- | --- |
| `BackProjectionParams` | BackProjectionParams configures `IterativeBackProjection`. |
| `Interpolation` | Interpolation names a resize method for the convenience `Resize` entry point. |
| `ResampleKernel` | ResampleKernel describes a separable 1-D reconstruction filter used by the resize routines. |
| `Shift2D` | Shift2D is a sub-pixel translation in image coordinates, measured in pixels. |

<details>
<summary><code>BackProjectionParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultBackProjectionParams(scale int) BackProjectionParams` | DefaultBackProjectionParams returns default back-projection parameters for the given integer scale: eight iterations, a step size of 1, and a… |

</details>

<details>
<summary><code>ResampleKernel</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BSplineKernel() ResampleKernel` | BSplineKernel returns the cubic B-spline kernel (radius two). |
| `func CatmullRomKernel() ResampleKernel` | CatmullRomKernel returns the Catmull-Rom cubic (CubicKernel with a = -0.5), the standard interpolating bicubic used for photographic upscaling. |
| `func CubicKernel(a float64) ResampleKernel` | CubicKernel returns the parametric cubic convolution kernel of Keys with free parameter a (radius two). |
| `func LanczosKernel(a int) ResampleKernel` | LanczosKernel returns the Lanczos windowed-sinc kernel with lobe count a (radius a). |
| `func LinearKernel() ResampleKernel` | LinearKernel returns the linear (triangle / tent) kernel of radius one, producing bilinear interpolation when applied separably. |
| `func MitchellKernel(b, c float64) ResampleKernel` | MitchellKernel returns the Mitchell-Netravali cubic with parameters b and c (radius two). |
| `func NearestKernel() ResampleKernel` | NearestKernel returns the nearest-neighbour kernel: a unit box of radius one half. |

</details>

<details>
<summary><code>Shift2D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func EstimateShift(ref, moved *cv.Mat) Shift2D` | EstimateShift estimates the sub-pixel translation that best maps ref onto moved, assuming the two images differ by a small global shift (the classic… |
| `func EstimateShiftRefine(ref, moved *cv.Mat, iterations int) Shift2D` | EstimateShiftRefine estimates a larger sub-pixel shift by iterated Lucas-Kanade: it repeatedly estimates the residual shift with `EstimateShift` and… |
| `func PhaseCorrelateShift(ref, moved *cv.Mat) Shift2D` | PhaseCorrelateShift estimates the integer-plus-sub-pixel translation between ref and moved by phase correlation: it computes the cross-power spectrum… |
| `func RegisterFrames(frames []*cv.Mat) []Shift2D` | RegisterFrames estimates the sub-pixel translation of each frame relative to the first (reference) frame using iterated Lucas-Kanade… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/superres).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
