# filters2 — standard-library-only collection of advanced image filtering

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/filters2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/filters2)

Package filters2 is a standard-library-only collection of advanced image
filtering, denoising and multi-scale analysis routines built on top of the
parent package's `github.com/malcolmston/opencv.Mat` image type.

- Edge-preserving smoothing and denoising: bilateral and joint (cross)
bilateral filtering, the guided filter of He et al., non-local means,
Perona-Malik anisotropic diffusion and the Kuwahara filter. - Rank filters:
plain and adaptive median, min/max, midpoint and alpha-trimmed-mean filters. -
Sharpening: unsharp masking and high-boost filtering. - Multi-scale band-pass
analysis: the Laplacian-of-Gaussian, the difference-of-Gaussians and
Marr-Hildreth zero-crossing edges. - Oriented analysis: Gabor kernels and
Gabor filter banks, and the Freeman-Adelson steerable first- and
second-derivative-of-Gaussian filters.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/filters2"
```

## Usage

This is the package's own `ExampleBilateralFilter`, so it compiles and its output is
asserted on every `go test ./filters2/`.

```go
img := cv.NewMat(9, 9, 1)
	for y := 0; y < 9; y++ {
		for x := 5; x < 9; x++ {
			img.Set(y, x, 0, 200)
		}
	}
	out := filters2.BilateralFilter(img, 5, 20, 3)
	fmt.Println("dark side:", out.At(4, 2, 0), "bright side:", out.At(4, 7, 0))
```

```
dark side: 0 bright side: 200
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AdaptiveMedianFilter(src *cv.Mat, maxSize int) *cv.Mat` | AdaptiveMedianFilter applies the adaptive median filter, which grows its window from 3×3 up to maxSize×maxSize (maxSize must be a positive odd… |
| `func AlphaTrimmedMeanFilter(src *cv.Mat, size, trim int) *cv.Mat` | AlphaTrimmedMeanFilter replaces every sample with the mean of its neighbourhood of the given odd size after discarding the trim smallest and trim… |
| `func AnisotropicDiffusion(src *cv.Mat, iterations int, kappa, lambda float64, option DiffusionOption) *cv.Mat` | AnisotropicDiffusion applies iterations of Perona-Malik anisotropic diffusion, an edge-preserving smoothing that damps intra-region gradients while… |
| `func BilateralFilter(src *cv.Mat, d int, sigmaColor, sigmaSpace float64) *cv.Mat` | BilateralFilter applies an edge-preserving bilateral filter. |
| `func DifferenceOfGaussiansKernel(size int, sigma1, sigma2 float64) [][]float64` | DifferenceOfGaussiansKernel returns a size×size difference-of-Gaussians kernel, the normalised Gaussian of standard deviation sigma1 minus the… |
| `func GaborEnergy(src *cv.Mat, size int, bank []GaborParams) *cv.Mat` | GaborEnergy applies every filter in a bank to a single-channel image and returns, per pixel, the maximum magnitude response across the bank, rescaled… |
| `func GaborKernel(size int, p GaborParams) [][]float64` | GaborKernel returns a size×size real Gabor kernel for the given parameters. |
| `func GaussianBlur(src *cv.Mat, sigma float64) *cv.Mat` | GaussianBlur returns a Gaussian-blurred copy of src using a separable kernel of standard deviation sigma (the truncation radius is chosen from sigma)… |
| `func GaussianDerivativeKernel(size int, sigma float64, dx, dy int) [][]float64` | GaussianDerivativeKernel returns a size×size sampled partial derivative of a 2-D Gaussian of the given standard deviation: the (dx-th, dy-th) mixed… |
| `func GaussianKernel1D(radius int, sigma float64) []float64` | GaussianKernel1D returns a normalised 1-D Gaussian kernel of length 2*radius+1 with the given standard deviation, so its samples sum to one. |
| `func GaussianKernel2D(size int, sigma float64) [][]float64` | GaussianKernel2D returns a normalised 2-D Gaussian kernel of size size×size (size must be a positive odd integer) with the given standard deviation,… |
| `func GuidedFilter(src, guide *cv.Mat, radius int, eps float64) *cv.Mat` | GuidedFilter applies the guided filter of He, Sun and Tang: an edge-preserving smoothing of src whose edges follow the single-channel guide image. |
| `func HighBoostFilter(src *cv.Mat, sigma, boost float64) *cv.Mat` | HighBoostFilter applies high-boost (high-frequency-emphasis) filtering: out = boost*src - blur, where blur is a Gaussian low-pass of standard… |
| `func JointBilateralFilter(src, guide *cv.Mat, d int, sigmaColor, sigmaSpace float64) *cv.Mat` | JointBilateralFilter applies a cross (joint) bilateral filter: the spatial term and the averaging act on src, but the range weight is computed from… |
| `func KuwaharaFilter(src *cv.Mat, size int) *cv.Mat` | KuwaharaFilter applies the Kuwahara edge-preserving smoothing filter. |
| `func LaplacianOfGaussianKernel(size int, sigma float64) [][]float64` | LaplacianOfGaussianKernel returns a size×size sampled Laplacian-of-Gaussian (Mexican-hat) kernel of the given standard deviation. |
| `func LaplacianSharpen(src *cv.Mat, amount float64) *cv.Mat` | LaplacianSharpen sharpens src by subtracting a scaled discrete Laplacian, out = src - amount*Laplacian(src), which emphasises edges and fine detail. |
| `func MarrHildrethEdges(src *cv.Mat, sigma, threshold float64) *cv.Mat` | MarrHildrethEdges detects edges by the Marr-Hildreth method: it computes the Laplacian of Gaussian of a single-channel image and marks the zero… |
| `func MaxFilter(src *cv.Mat, size int) *cv.Mat` | MaxFilter (dilation by a square element) replaces every sample with the maximum of its neighbourhood of the given odd size, processing each channel… |
| `func MedianFilter(src *cv.Mat, size int) *cv.Mat` | MedianFilter replaces every sample with the median of its square neighbourhood of the given odd size, an effective remedy for impulse… |
| `func MidpointFilter(src *cv.Mat, size int) *cv.Mat` | MidpointFilter replaces every sample with the midpoint (min+max)/2 of its neighbourhood of the given odd size, which suppresses Gaussian and uniform… |
| `func MinFilter(src *cv.Mat, size int) *cv.Mat` | MinFilter (erosion by a square element) replaces every sample with the minimum of its neighbourhood of the given odd size, processing each channel… |
| `func NonLocalMeans(src *cv.Mat, h float64, templateRadius, searchRadius int) *cv.Mat` | NonLocalMeans denoises src with the non-local means algorithm of Buades, Coll and Morel. |
| `func UnsharpMask(src *cv.Mat, sigma, amount, threshold float64) *cv.Mat` | UnsharpMask sharpens src by adding a fraction of the difference between the image and a Gaussian-blurred version of it: out = src + amount*(src-blur). |

### Types

| Type | What it is |
| --- | --- |
| `DiffusionOption` | DiffusionOption selects the conduction (edge-stopping) function used by `AnisotropicDiffusion`. |
| `FloatImage` | FloatImage is a dense, single-channel raster of float64 samples in row-major order. |
| `GaborParams` | GaborParams describes a 2-D Gabor filter. |
| `SteerableG1` | SteerableG1 holds the two basis responses of a first-derivative-of-Gaussian steerable filter (Freeman and Adelson): Gx is the response to the x… |
| `SteerableG2` | SteerableG2 holds the three basis responses of a second-derivative-of- Gaussian steerable filter (Freeman and Adelson): Gxx, Gxy and Gyy are the… |

<details>
<summary><code>FloatImage</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Convolve(src *FloatImage, kernel [][]float64) *FloatImage` | Convolve correlates src with a row-major kernel anchored at its centre and returns a new FloatImage of the same size, using edge replication at the… |
| `func ConvolveMat(src *cv.Mat, kernel [][]float64) *FloatImage` | ConvolveMat correlates a single-channel `cv.Mat` with a row-major kernel anchored at its centre and returns the unclamped, signed result as a… |
| `func DifferenceOfGaussians(src *cv.Mat, sigma1, sigma2 float64, size int) *FloatImage` | DifferenceOfGaussians convolves a single-channel image with the difference of two Gaussians of standard deviations sigma1 and sigma2 (with sigma1 <… |
| `func GaborBankResponse(src *cv.Mat, size int, bank []GaborParams) []*FloatImage` | GaborBankResponse applies every filter in a bank to a single-channel image and returns the per-filter magnitude responses, in bank order. |
| `func GaborFilter(src *cv.Mat, size int, p GaborParams) *FloatImage` | GaborFilter convolves a single-channel image with the real (even) Gabor kernel described by p at the given odd kernel size and returns the signed… |
| `func GaborMagnitude(src *cv.Mat, size int, p GaborParams) *FloatImage` | GaborMagnitude convolves a single-channel image with the even (cosine) and odd (sine) Gabor kernels described by p and returns the magnitude of the… |
| `func LaplacianOfGaussian(src *cv.Mat, sigma float64, size int) *FloatImage` | LaplacianOfGaussian convolves a single-channel image with a Laplacian-of-Gaussian kernel of the given standard deviation and returns the signed… |
| `func Magnitude(a, b *FloatImage) *FloatImage` | Magnitude returns the pointwise Euclidean magnitude sqrt(a^2+b^2) of two equally sized responses, for example the real and imaginary parts of a… |
| `func MatToFloatImage(src *cv.Mat) *FloatImage` | MatToFloatImage converts a single-channel `cv.Mat` to a FloatImage, copying each sample as a float64. |
| `func NewFloatImage(rows, cols int) *FloatImage` | NewFloatImage allocates a zero-filled FloatImage with the given dimensions. |
| `func (f *FloatImage) Abs() *FloatImage` | Abs returns a new FloatImage holding the absolute value of every sample. |
| `func (f *FloatImage) Add(other *FloatImage) *FloatImage` | Add returns the pointwise sum f+other. |
| `func (f *FloatImage) At(y, x int) float64` | At returns the sample at row y, column x. |
| `func (f *FloatImage) Clone() *FloatImage` | Clone returns an independent deep copy of the image. |
| `func (f *FloatImage) Mean() float64` | Mean returns the arithmetic mean of all samples. |
| `func (f *FloatImage) MinMax() (min, max float64)` | MinMax returns the smallest and largest sample values. |
| `func (f *FloatImage) Normalize() *cv.Mat` | Normalize linearly rescales the sample range [min,max] onto [0,255] and returns the result as an 8-bit single-channel `cv.Mat`. |
| `func (f *FloatImage) Scale(s float64) *FloatImage` | Scale returns a new FloatImage with every sample multiplied by s. |
| `func (f *FloatImage) Set(y, x int, value float64)` | Set stores value at row y, column x. |
| `func (f *FloatImage) Sub(other *FloatImage) *FloatImage` | Sub returns the pointwise difference f-other. |
| `func (f *FloatImage) ToMat() *cv.Mat` | ToMat converts the image to an 8-bit single-channel `cv.Mat` by rounding and clamping each sample into the [0,255] range, without rescaling. |

</details>

<details>
<summary><code>GaborParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func GaborBank(numOrientations int, lambdas []float64, sigma, gamma, psi float64) []GaborParams` | GaborBank builds a bank of `GaborParams` sweeping numOrientations equally spaced orientations over [0,pi) for each supplied wavelength in lambdas,… |

</details>

<details>
<summary><code>SteerableG1</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func SteerG1(src *cv.Mat, sigma float64, size int) *SteerableG1` | SteerG1 convolves a single-channel image with the two first-derivative-of- Gaussian basis filters of the given standard deviation and returns their… |
| `func (s *SteerableG1) DominantOrientation() *FloatImage` | DominantOrientation returns, per pixel, the orientation in radians (in (-pi,pi]) that maximises the first-derivative response, namely atan2(Gy,Gx). |
| `func (s *SteerableG1) OrientationEnergy() *FloatImage` | OrientationEnergy returns the orientation-independent gradient energy sqrt(Gx^2+Gy^2) of the first-derivative basis, the largest response obtainable… |
| `func (s *SteerableG1) Steer(theta float64) *FloatImage` | Steer returns the first-derivative-of-Gaussian response oriented at theta radians, computed as cos(theta)*Gx + sin(theta)*Gy. |

</details>

<details>
<summary><code>SteerableG2</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func SteerG2(src *cv.Mat, sigma float64, size int) *SteerableG2` | SteerG2 convolves a single-channel image with the three second-derivative-of- Gaussian basis filters of the given standard deviation and returns… |
| `func (s *SteerableG2) DominantOrientation() *FloatImage` | DominantOrientation returns, per pixel, the orientation in radians that maximises the magnitude of the steered second-derivative response. |
| `func (s *SteerableG2) OrientationEnergy() *FloatImage` | OrientationEnergy returns, per pixel, the maximum squared second-derivative response over all orientations. |
| `func (s *SteerableG2) Steer(theta float64) *FloatImage` | Steer returns the second-derivative-of-Gaussian response oriented at theta radians, computed with the Freeman-Adelson interpolation cos^2(theta)*Gxx… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/filters2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
