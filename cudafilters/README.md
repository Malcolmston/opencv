# cudafilters — pure-Go, CPU-backed, API-compatible mirror of OpenCV's cudafilters module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/cudafilters.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/cudafilters)

Package cudafilters is a pure-Go, CPU-backed, API-compatible mirror of
OpenCV's cudafilters module. It reproduces the shape of that module — a
`Filter` object interface obtained from Create* factory functions and applied
with `Filter.Apply`, together with a `GpuMat` device-matrix wrapper and a
no-op `Stream` — but performs every computation on the CPU using the
standard-library-only OpenCV port github.com/malcolmston/opencv (imported here
as cv). It uses no cgo, no CUDA, no GPU and no third-party code.

There is NO GPU acceleration here. OpenCV's real cudafilters offloads
convolution and morphology to an NVIDIA GPU; this package instead delegates to
the CPU filter primitives in the root package (`cv.Filter2D`,
`cv.GaussianBlur`, `cv.Sobel`, `cv.Erode`, `cv.MorphologyEx`, and friends).
The `GpuMat` type does not allocate device memory — it holds an ordinary
`cv.Mat`; Upload and Download are deep copies that model the host⇆device
transfer boundary without any real transfer. The `Stream` type is a no-op
placeholder so that call sites written against the CUDA API compile and run
unchanged. The value of this package is source compatibility and identical
numerical results to the corresponding root-package call, not speed.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/cudafilters"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./cudafilters/`.

```go
src := cudafilters.GpuMatFromMat(ramp(5, 5))
	out := cudafilters.GaussianBlur(src, image.Pt(3, 3), 1.0, 1.0).Download()

	fmt.Printf("%dx%d\n", out.Rows, out.Cols)
```

```
5x5
```

## Exported surface

### Types

| Type | What it is |
| --- | --- |
| `BorderType` | BorderType names a pixel-extrapolation method, mirroring OpenCV's border constants for API compatibility. |
| `Filter` | Filter is the object interface OpenCV's cudafilters exposes: a reusable filter created by one of the Create* factories and applied to a `GpuMat` with… |
| `GpuMat` | GpuMat is a CPU-backed stand-in for OpenCV's cv::cuda::GpuMat. |
| `MorphOp` | MorphOp selects the operation performed by `CreateMorphologyFilter`, mirroring OpenCV's MorphTypes. |
| `Stream` | Stream is a no-op stand-in for OpenCV's cv::cuda::Stream. |

<details>
<summary><code>Filter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateBlackHatFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter` | CreateBlackHatFilter is a shortcut for `CreateMorphologyFilter` with `MorphBlackhat`. |
| `func CreateBlurFilter(ksize image.Point) Filter` | CreateBlurFilter is an alias for `CreateBoxFilter` with the centred anchor and default border, matching OpenCV's blur convenience. |
| `func CreateBoxFilter(ksize image.Point, anchor image.Point, borderType BorderType) Filter` | CreateBoxFilter returns a `Filter` that smooths with a normalised (averaging) box of size ksize, mirroring cv::cuda::createBoxFilter. |
| `func CreateBoxMaxFilter(ksize image.Point, anchor image.Point, borderType BorderType) Filter` | CreateBoxMaxFilter returns a `Filter` replacing each sample with the maximum over a ksize rectangular window, mirroring cv::cuda::createBoxMaxFilter. |
| `func CreateBoxMinFilter(ksize image.Point, anchor image.Point, borderType BorderType) Filter` | CreateBoxMinFilter returns a `Filter` replacing each sample with the minimum over a ksize rectangular window, mirroring cv::cuda::createBoxMinFilter. |
| `func CreateCloseFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter` | CreateCloseFilter is a shortcut for `CreateMorphologyFilter` with `MorphClose`. |
| `func CreateColumnSumFilter(ksize int) Filter` | CreateColumnSumFilter returns a `Filter` computing, for every pixel, the sum of its ksize vertical neighbours, mirroring… |
| `func CreateDerivFilter(dx, dy, ksize int, scale, delta float64, borderType BorderType) Filter` | CreateDerivFilter returns a general derivative `Filter` of orders (dx, dy), mirroring cv::cuda::createDerivFilter. |
| `func CreateDilateFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter` | CreateDilateFilter is a shortcut for `CreateMorphologyFilter` with `MorphDilate`. |
| `func CreateErodeFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter` | CreateErodeFilter is a shortcut for `CreateMorphologyFilter` with `MorphErode`. |
| `func CreateGaussianFilter(ksize image.Point, sigma1, sigma2 float64, borderType BorderType) Filter` | CreateGaussianFilter returns a `Filter` that blurs with a separable Gaussian of size ksize, mirroring cv::cuda::createGaussianFilter. |
| `func CreateLaplacianFilter(ksize int, scale, delta float64, borderType BorderType) Filter` | CreateLaplacianFilter returns a `Filter` computing the Laplacian with aperture ksize (1 or 3), mirroring cv::cuda::createLaplacianFilter. |
| `func CreateLinearFilter(kernel cv.Kernel, delta float64, anchor image.Point, borderType BorderType) Filter` | CreateLinearFilter returns a `Filter` that convolves with an arbitrary 2-D kernel, mirroring cv::cuda::createLinearFilter (the GPU analogue of… |
| `func CreateMedianFilter(windowSize int) Filter` | CreateMedianFilter returns a `Filter` replacing each sample with the median of its windowSize×windowSize neighbourhood, mirroring the intent of… |
| `func CreateMorphologyFilter(op MorphOp, kernel *cv.Mat, anchor image.Point, iterations int) Filter` | CreateMorphologyFilter returns a `Filter` performing the morphological operation op over the structuring element kernel, mirroring cv::cuda::… |
| `func CreateMorphologyGradientFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter` | CreateMorphologyGradientFilter is a shortcut for `CreateMorphologyFilter` with `MorphGradient`. |
| `func CreateOpenFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter` | CreateOpenFilter is a shortcut for `CreateMorphologyFilter` with `MorphOpen`. |
| `func CreateRowSumFilter(ksize int) Filter` | CreateRowSumFilter returns a `Filter` computing, for every pixel, the sum of its ksize horizontal neighbours, mirroring cv::cuda::createRowSumFilter. |
| `func CreateScharrFilter(dx, dy int, scale, delta float64, borderType BorderType) Filter` | CreateScharrFilter returns a `Filter` computing the 3×3 Scharr derivative, mirroring cv::cuda::createScharrFilter. |
| `func CreateSeparableLinearFilter(rowKernel, columnKernel []float64, delta float64, anchor image.Point, borderType BorderType) Filter` | CreateSeparableLinearFilter returns a `Filter` that applies a separable linear filter: convolution with the 1-D horizontal kernel rowKernel followed… |
| `func CreateSobelFilter(dx, dy, ksize int, scale, delta float64, borderType BorderType) Filter` | CreateSobelFilter returns a `Filter` computing the Sobel derivative of orders (dx, dy) with aperture ksize (1 or 3), mirroring… |
| `func CreateTopHatFilter(kernel *cv.Mat, anchor image.Point, iterations int) Filter` | CreateTopHatFilter is a shortcut for `CreateMorphologyFilter` with `MorphTophat`. |

</details>

<details>
<summary><code>GpuMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Blur(src *GpuMat, ksize image.Point, stream ...*Stream) *GpuMat` | Blur smooths src with a normalised box of the given square odd ksize. |
| `func BoxFilter(src *GpuMat, ksize image.Point, stream ...*Stream) *GpuMat` | BoxFilter smooths src with a normalised box of the given square odd ksize. |
| `func BoxMax(src *GpuMat, ksize image.Point, stream ...*Stream) *GpuMat` | BoxMax replaces each sample with the maximum over a ksize rectangle. |
| `func BoxMin(src *GpuMat, ksize image.Point, stream ...*Stream) *GpuMat` | BoxMin replaces each sample with the minimum over a ksize rectangle. |
| `func ColumnSum(src *GpuMat, ksize int, stream ...*Stream) *GpuMat` | ColumnSum computes the running vertical sum over ksize neighbours. |
| `func Dilate(src *GpuMat, kernel *cv.Mat, iterations int, stream ...*Stream) *GpuMat` | Dilate applies a morphological dilation with the given structuring element. |
| `func Erode(src *GpuMat, kernel *cv.Mat, iterations int, stream ...*Stream) *GpuMat` | Erode applies a morphological erosion with the given structuring element. |
| `func Filter2D(src *GpuMat, kernel cv.Kernel, delta float64, stream ...*Stream) *GpuMat` | Filter2D convolves src with an arbitrary kernel plus delta. |
| `func GaussianBlur(src *GpuMat, ksize image.Point, sigma1, sigma2 float64, stream ...*Stream) *GpuMat` | GaussianBlur blurs src with a separable Gaussian of the given ksize and standard deviations. |
| `func GpuMatFromMat(m *cv.Mat) *GpuMat` | GpuMatFromMat returns a `GpuMat` whose contents are a deep copy of m. |
| `func Laplacian(src *GpuMat, ksize int, scale, delta float64, stream ...*Stream) *GpuMat` | Laplacian computes the Laplacian with aperture ksize. |
| `func MedianBlur(src *GpuMat, windowSize int, stream ...*Stream) *GpuMat` | MedianBlur replaces each sample with the median of its windowSize×windowSize neighbourhood. |
| `func MorphologyEx(src *GpuMat, op MorphOp, kernel *cv.Mat, iterations int, stream ...*Stream) *GpuMat` | MorphologyEx applies the compound morphological operation op with the given structuring element. |
| `func NewGpuMat() *GpuMat` | NewGpuMat returns an empty `GpuMat` holding no matrix. |
| `func RowSum(src *GpuMat, ksize int, stream ...*Stream) *GpuMat` | RowSum computes the running horizontal sum over ksize neighbours. |
| `func Scharr(src *GpuMat, dx, dy int, scale, delta float64, stream ...*Stream) *GpuMat` | Scharr computes the 3×3 Scharr derivative of orders (dx, dy). |
| `func SepFilter2D(src *GpuMat, rowKernel, columnKernel []float64, delta float64, stream ...*Stream) *GpuMat` | SepFilter2D applies a separable linear filter (horizontal rowKernel then vertical columnKernel) plus delta. |
| `func Sobel(src *GpuMat, dx, dy, ksize int, scale, delta float64, stream ...*Stream) *GpuMat` | Sobel computes the Sobel derivative of orders (dx, dy) with aperture ksize. |
| `func (g *GpuMat) Channels() int` | Channels returns the number of samples per pixel, or 0 when empty. |
| `func (g *GpuMat) Clone() *GpuMat` | Clone returns a deep copy of the GpuMat with its own backing storage. |
| `func (g *GpuMat) Download() *cv.Mat` | Download returns a deep copy of the GpuMat's contents as a `cv.Mat`, modelling a device-to-host transfer. |
| `func (g *GpuMat) Empty() bool` | Empty reports whether the GpuMat holds no usable matrix. |
| `func (g *GpuMat) Release()` | Release frees the GpuMat's contents, leaving it empty. |
| `func (g *GpuMat) Size() (rows, cols int)` | Size returns the matrix dimensions as (rows, cols), or (0, 0) when empty. |
| `func (g *GpuMat) Upload(m *cv.Mat)` | Upload copies m into the GpuMat, modelling a host-to-device transfer. |

</details>

<details>
<summary><code>Stream</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStream() *Stream` | NewStream returns a ready-to-use `Stream`. |
| `func (s *Stream) Release()` | Release does nothing; a `Stream` owns no resources. |
| `func (s *Stream) WaitForCompletion()` | WaitForCompletion returns immediately: all work in this package is synchronous, so there is never anything to wait for. |

</details>

### Variables

`AnchorCenter`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/cudafilters).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
