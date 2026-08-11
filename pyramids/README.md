# pyramids — implements image pyramids and scale-space machinery on top of the parent cv module's

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/pyramids.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/pyramids)

Package pyramids implements image pyramids and scale-space machinery on top of
the parent cv module's image types.

The package builds directly on the parent package's `cv.Mat` (dense 8-bit
images) and `cv.FloatMat` (single-channel float64 grids). Rather than
introducing an incompatible image type, every routine here consumes and
produces those two types so results interoperate with the rest of the module.
Because pyramid arithmetic — Laplacian differences, wavelet detail
coefficients, difference-of-Gaussian responses — routinely goes negative and
must round-trip losslessly, the internal representation is the float64
`cv.FloatMat`. Convert an 8-bit image to that domain with `GrayFloat` or
`ChannelFloat`, operate, then convert back with `FloatToMat` (clamping) or
`FloatToMatNormalized` (contrast-stretched for visualisation).

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/pyramids"
```

## Usage

This is the package's own `ExampleBlendLaplacianMat`, so it compiles and its output is
asserted on every `go test ./pyramids/`.

```go
const n = 32
	left := cv.NewMat(n, n, 1)
	right := cv.NewMat(n, n, 1)
	mask := cv.NewMat(n, n, 1)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			left.Data[y*n+x] = 60
			right.Data[y*n+x] = 200
			if x < n/2 {
				mask.Data[y*n+x] = 255
			}
		}
	}

	out := pyramids.BlendLaplacianMat(left, right, mask, 4)

	fmt.Println("left edge near 60:", absDiff8(out.At(16, 1, 0), 60) <= 5)
	fmt.Println("right edge near 200:", absDiff8(out.At(16, n-2, 0), 200) <= 5)
	fmt.Println("monotonic across seam:", out.At(16, 12, 0) < out.At(16, 20, 0))
```

```
left edge near 60: true
right edge near 200: true
monotonic across seam: true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AbsFloat(f *cv.FloatMat) *cv.FloatMat` | AbsFloat returns a new grid holding the absolute value of every element of f. |
| `func AddFloat(a, b *cv.FloatMat) *cv.FloatMat` | AddFloat returns the element-wise sum a+b of two equally sized grids. |
| `func AddScaledFloat(a *cv.FloatMat, s float64, b *cv.FloatMat) *cv.FloatMat` | AddScaledFloat returns a + s*b for two equally sized grids, a fused multiply-add convenient for blending and reconstruction. |
| `func AlphaBlendFloat(a, b, mask *cv.FloatMat) *cv.FloatMat` | AlphaBlendFloat linearly blends two equally sized grids under a per-pixel weight: out = a*mask + b*(1-mask). |
| `func BinomialKernel() []float64` | BinomialKernel returns the normalised 5-tap binomial kernel [1 4 6 4 1]/16, the standard separable approximation to a Gaussian used for pyramid… |
| `func BlendLaplacian(a, b, mask *cv.FloatMat, levels int) *cv.FloatMat` | BlendLaplacian performs seamless multi-resolution blending of two single-channel images a and b under a soft mask, following Burt and Adelson's… |
| `func BlendLaplacianMat(a, b, mask *cv.Mat, levels int) *cv.Mat` | BlendLaplacianMat performs multi-resolution blending of two 8-bit images channel by channel and returns an 8-bit result. |
| `func ChannelFloat(m *cv.Mat, c int) *cv.FloatMat` | ChannelFloat extracts channel c of an 8-bit `cv.Mat` as a float64 grid, keeping the 0..255 scale. |
| `func CloneFloat(f *cv.FloatMat) *cv.FloatMat` | CloneFloat returns an independent deep copy of a float64 grid. |
| `func Convolve2D(f *cv.FloatMat, kernel *cv.FloatMat) *cv.FloatMat` | Convolve2D convolves f with an arbitrary 2-D kernel anchored at its centre, replicating the border, and returns a new grid. |
| `func ConvolveSeparable(f *cv.FloatMat, kx, ky []float64) *cv.FloatMat` | ConvolveSeparable convolves f with the separable kernel formed by the outer product of the horizontal kernel kx and the vertical kernel ky,… |
| `func DifferenceOfGaussians(f *cv.FloatMat, sigma1, sigma2 float64) *cv.FloatMat` | DifferenceOfGaussians returns the difference of two Gaussian-blurred versions of f, blur(sigma1) - blur(sigma2). |
| `func FloatToMat(f *cv.FloatMat) *cv.Mat` | FloatToMat converts a single-channel float64 grid to an 8-bit grayscale `cv.Mat`, rounding and clamping each value to the 0..255 range. |
| `func FloatToMatNormalized(f *cv.FloatMat) *cv.Mat` | FloatToMatNormalized converts a float64 grid to an 8-bit grayscale `cv.Mat` after linearly rescaling its actual minimum to 0 and maximum to 255. |
| `func GaussianBlurFloat(f *cv.FloatMat, sigma float64) *cv.FloatMat` | GaussianBlurFloat blurs f with a separable Gaussian of standard deviation sigma, replicating the border, and returns a new grid. |
| `func GaussianDerivativeKernel(sigma float64) []float64` | GaussianDerivativeKernel returns a 1-D first-derivative-of-Gaussian kernel with standard deviation sigma, sampled over radius ceil(3*sigma). |
| `func GaussianKernel(sigma float64) []float64` | GaussianKernel returns a normalised 1-D Gaussian kernel with standard deviation sigma. |
| `func GaussianSecondDerivativeKernel(sigma float64) []float64` | GaussianSecondDerivativeKernel returns a 1-D second-derivative-of-Gaussian kernel with standard deviation sigma over radius ceil(3*sigma). |
| `func GrayFloat(m *cv.Mat) *cv.FloatMat` | GrayFloat converts an 8-bit `cv.Mat` to a single-channel float64 grid. |
| `func HaarInverse(h *HaarBands) *cv.FloatMat` | HaarInverse reconstructs the image from one level of Haar sub-bands, the exact inverse of `HaarForward`. |
| `func LaplacianOfGaussianKernel(sigma float64) *cv.FloatMat` | LaplacianOfGaussianKernel returns a square, odd-sized 2-D Laplacian-of- Gaussian (Mexican-hat) kernel with standard deviation sigma. |
| `func MergeFloat(planes []*cv.FloatMat) *cv.Mat` | MergeFloat combines equally sized float64 grids into an interleaved 8-bit `cv.Mat`, one grid per channel, rounding and clamping each value to 0..255. |
| `func MultiBandBlendMat(images, masks []*cv.Mat, levels int) *cv.Mat` | MultiBandBlendMat blends an arbitrary number of 8-bit source images into a single 8-bit mosaic using per-source soft masks and Laplacian-pyramid… |
| `func NormalizeFloat(f *cv.FloatMat, lo, hi float64) *cv.FloatMat` | NormalizeFloat linearly rescales f so its minimum maps to lo and its maximum to hi, returning a new grid. |
| `func OrientationEnergy(gx, gy *cv.FloatMat) *cv.FloatMat` | OrientationEnergy returns the gradient magnitude sqrt(gx^2 + gy^2) of the two first-derivative basis responses, a rotation-invariant measure of local… |
| `func OrientationMap(gx, gy *cv.FloatMat) *cv.FloatMat` | OrientationMap returns the per-pixel gradient orientation atan2(gy, gx) in radians (range (-pi, pi]) from the two first-derivative basis responses. |
| `func PyrDownFloat(f *cv.FloatMat) *cv.FloatMat` | PyrDownFloat blurs f with the separable binomial kernel and drops every other row and column, halving each dimension (rounding up). |
| `func PyrUpFloat(f *cv.FloatMat, dstRows, dstCols int) *cv.FloatMat` | PyrUpFloat upsamples f to the given destination size by injecting its samples at even coordinates of a zero grid and smoothing with the binomial… |
| `func ScaleFloat(f *cv.FloatMat, s float64) *cv.FloatMat` | ScaleFloat returns a new grid with every element of f multiplied by s. |
| `func SplitFloat(m *cv.Mat) []*cv.FloatMat` | SplitFloat separates every channel of an 8-bit `cv.Mat` into its own float64 grid, returning them in channel order. |
| `func SteerG1(gx, gy *cv.FloatMat, theta float64) *cv.FloatMat` | SteerG1 evaluates the first-derivative steerable filter at orientation theta (radians, measured from the +x axis) from the basis responses produced… |
| `func SteerG2(gxx, gxy, gyy *cv.FloatMat, theta float64) *cv.FloatMat` | SteerG2 evaluates the second-derivative steerable filter at orientation theta (radians) from the basis responses of `SteerableBasisG2` using the… |
| `func SteerableBasisG1(f *cv.FloatMat, sigma float64) (gx, gy *cv.FloatMat)` | SteerableBasisG1 computes the two separable first-derivative-of-Gaussian basis responses of f at scale sigma: gx is the derivative along x (smoothed… |
| `func SteerableBasisG2(f *cv.FloatMat, sigma float64) (gxx, gxy, gyy *cv.FloatMat)` | SteerableBasisG2 computes the three separable second-derivative-of-Gaussian basis responses of f at scale sigma: gxx (second derivative along x), gxy… |
| `func SubtractFloat(a, b *cv.FloatMat) *cv.FloatMat` | SubtractFloat returns the element-wise difference a-b of two equally sized grids. |

### Types

| Type | What it is |
| --- | --- |
| `DoGScaleSpace` | DoGScaleSpace is a difference-of-Gaussian stack together with the standard deviation associated with each difference image, suitable for… |
| `GaussianPyramid` | GaussianPyramid is a sequence of increasingly coarse, blurred and down-sampled versions of an image. |
| `HaarBands` | HaarBands holds one level of a 2-D orthonormal Haar wavelet transform: the LL approximation (low-low) and the three detail sub-bands LH (horizontal… |
| `Keypoint` | Keypoint is a scale-space blob detected in a difference-of-Gaussian stack. |
| `LaplacianPyramid` | LaplacianPyramid is a band-pass decomposition of an image. |
| `ScaleSpace` | ScaleSpace is a stack of Gaussian-blurred versions of one image, all at the original resolution, with geometrically increasing standard deviations. |
| `SteerableLevel` | SteerableLevel holds the oriented band-pass responses at one octave of a steerable pyramid: one image per analysis orientation, plus the orientations… |
| `SteerablePyramid` | SteerablePyramid is a multi-scale, multi-orientation decomposition. |
| `WaveletPyramid` | WaveletPyramid is a multi-level Haar wavelet decomposition: the detail sub-bands at each level plus the coarsest approximation. |

<details>
<summary><code>DoGScaleSpace</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BuildDoGScaleSpace(f *cv.FloatMat, numDoG int, sigma0, k float64) *DoGScaleSpace` | BuildDoGScaleSpace builds a difference-of-Gaussian stack from a Gaussian scale space of numDoG+1 blur levels, so it contains numDoG DoG images. |
| `func (d *DoGScaleSpace) NumImages() int` | NumImages returns the number of DoG images in the stack. |

</details>

<details>
<summary><code>GaussianPyramid</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BuildGaussianPyramid(f *cv.FloatMat, levels int) *GaussianPyramid` | BuildGaussianPyramid constructs a Gaussian pyramid of the requested number of levels from f. |
| `func (p *GaussianPyramid) Base() *cv.FloatMat` | Base returns the finest (full-resolution) level of the pyramid. |
| `func (p *GaussianPyramid) Coarsest() *cv.FloatMat` | Coarsest returns the smallest (last) level of the pyramid. |
| `func (p *GaussianPyramid) Level(i int) *cv.FloatMat` | Level returns the pyramid image at index i (0 is finest). |
| `func (p *GaussianPyramid) NumLevels() int` | NumLevels returns the number of levels in the pyramid. |

</details>

<details>
<summary><code>HaarBands</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func HaarForward(f *cv.FloatMat) *HaarBands` | HaarForward applies one level of the 2-D orthonormal Haar wavelet transform to f, returning the four half-size sub-bands. |

</details>

<details>
<summary><code>Keypoint</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DetectDoGExtrema(d *DoGScaleSpace, threshold float64) []Keypoint` | DetectDoGExtrema finds blobs as local extrema of the difference-of-Gaussian stack. |

</details>

<details>
<summary><code>LaplacianPyramid</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BuildLaplacianPyramid(f *cv.FloatMat, levels int) *LaplacianPyramid` | BuildLaplacianPyramid constructs a Laplacian pyramid with the requested number of band levels plus a residual base. |
| `func (lp *LaplacianPyramid) Band(i int) *cv.FloatMat` | Band returns the band-pass image at index i (0 is finest). |
| `func (lp *LaplacianPyramid) NumBands() int` | NumBands returns the number of band-pass levels (excluding the base). |
| `func (lp *LaplacianPyramid) Reconstruct() *cv.FloatMat` | Reconstruct rebuilds the original image from the Laplacian pyramid by repeatedly expanding the running coarse image and adding the next finer band. |

</details>

<details>
<summary><code>ScaleSpace</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BuildScaleSpace(f *cv.FloatMat, numLevels int, sigma0, k float64) *ScaleSpace` | BuildScaleSpace produces a Gaussian scale space of numLevels images. |
| `func (s *ScaleSpace) DoG() []*cv.FloatMat` | DoG returns the difference-of-Gaussian stack derived from adjacent scale levels: element i is Level(i+1) - Level(i). |
| `func (s *ScaleSpace) Level(i int) *cv.FloatMat` | Level returns the blurred image at index i. |
| `func (s *ScaleSpace) NumLevels() int` | NumLevels returns the number of levels in the scale space. |

</details>

<details>
<summary><code>SteerablePyramid</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BuildSteerablePyramid(f *cv.FloatMat, levels, orientations int) *SteerablePyramid` | BuildSteerablePyramid decomposes f into levels octaves, each analysed into the given number of orientations equally spaced over [0, pi). |
| `func (sp *SteerablePyramid) NumLevels() int` | NumLevels returns the number of octaves in the steerable pyramid. |
| `func (sp *SteerablePyramid) Orientations() int` | Orientations returns the number of orientation bands per level (0 if the pyramid is empty). |

</details>

<details>
<summary><code>WaveletPyramid</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BuildWaveletPyramid(f *cv.FloatMat, levels int) *WaveletPyramid` | BuildWaveletPyramid applies the Haar transform recursively, feeding each level's LL approximation into the next, for levels iterations (or until the… |
| `func (wp *WaveletPyramid) NumLevels() int` | NumLevels returns the number of wavelet decomposition levels. |
| `func (wp *WaveletPyramid) Reconstruct() *cv.FloatMat` | Reconstruct rebuilds the original image by inverting each Haar level from coarsest to finest, cropping any edge-replication padding introduced for… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/pyramids).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
