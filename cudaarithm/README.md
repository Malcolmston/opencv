# cudaarithm — CPU-backed, API-compatible mirror of OpenCV's cudaarithm module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/cudaarithm.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/cudaarithm)

Package cudaarithm is a CPU-backed, API-compatible mirror of OpenCV's
cudaarithm module. It provides the shape of OpenCV's CUDA per-element and
matrix-reduction API — a `GpuMat` device-matrix type, a no-op async
`Stream`, and a large family of element-wise, mathematical, channel and
reduction operations — implemented entirely in pure Go on top of the root
module github.com/malcolmston/opencv (imported as cv) and the Go standard
library.

- `GpuMat` is a thin wrapper around a *cv.Mat. "Uploading" and "downloading"
(`GpuMat.Upload`, `GpuMat.Download`, `NewGpuMat`) copy the matrix rather than
moving it across a PCIe bus, because host and device memory are the same
memory here. The copy semantics are preserved (the returned/stored Mat does
not alias the caller's), which keeps behaviour predictable, but there is no
acceleration. - `Stream` is a no-op. Every operation accepts an optional
trailing stream argument for source compatibility, but work is always
performed synchronously before the call returns, so `Stream.WaitForCompletion`
has nothing to wait for.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/cudaarithm"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./cudaarithm/`.

```go
a := cv.NewMat(1, 4, 1)
	copy(a.Data, []uint8{10, 90, 130, 250})
	b := cv.NewMat(1, 4, 1)
	copy(b.Data, []uint8{5, 30, 40, 10})

	sum := cudaarithm.Add(cudaarithm.NewGpuMat(a), cudaarithm.NewGpuMat(b))
	mask, _ := cudaarithm.Threshold(sum, 120, 255, cv.ThreshBinary)

	fmt.Println(ints(sum.Download().Data))
	fmt.Println(ints(mask.Download().Data))
```

```
[15 120 170 255]
[0 0 255 255]
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AbsSum(src *GpuMat, _ ...*Stream) []float64` | AbsSum returns the per-channel sum of absolute sample values, mirroring cv::cuda::absSum. |
| `func CountNonZero(src *GpuMat, _ ...*Stream) int` | CountNonZero returns the number of samples that are not zero, mirroring cv::cuda::countNonZero. |
| `func Mean(src *GpuMat, _ ...*Stream) []float64` | Mean returns the per-channel arithmetic mean of the samples, mirroring cv::cuda::mean. |
| `func MeanStdDev(src *GpuMat, _ ...*Stream) (mean, stddev []float64)` | MeanStdDev returns the per-channel mean and population standard deviation of the samples, mirroring cv::cuda::meanStdDev. |
| `func MinMax(src *GpuMat, _ ...*Stream) (minVal, maxVal float64)` | MinMax returns the minimum and maximum sample value of a single-channel GpuMat, mirroring cv::cuda::minMax. |
| `func MinMaxLoc(src *GpuMat, _ ...*Stream) (minVal, maxVal float64, minX, minY, maxX, maxY int)` | MinMaxLoc returns the minimum and maximum sample values of a single-channel GpuMat together with the (x, y) location of the first occurrence of each,… |
| `func Norm(src *GpuMat, normType NormType, _ ...*Stream) float64` | Norm returns the norm of src selected by normType, mirroring cv::cuda::norm. |
| `func SqrSum(src *GpuMat, _ ...*Stream) []float64` | SqrSum returns the per-channel sum of squared sample values, mirroring cv::cuda::sqrSum. |
| `func Sum(src *GpuMat, _ ...*Stream) []float64` | Sum returns the per-channel sum of all samples, mirroring cv::cuda::sum. |

### Types

| Type | What it is |
| --- | --- |
| `CmpOp` | CmpOp selects the relation applied by `Compare`. |
| `ComplexMat` | ComplexMat is a dense, row-major complex matrix used by `DFT`, `IDFT` and `MulSpectrums`. |
| `GpuMat` | GpuMat is a CPU-backed stand-in for OpenCV's cv::cuda::GpuMat. |
| `MatType` | MatType encodes an OpenCV matrix type. |
| `NormType` | NormType selects the norm used by `Norm` and `Normalize`. |
| `Stream` | Stream is a CPU-backed stand-in for cv::cuda::Stream. |

<details>
<summary><code>ComplexMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DFT(src *GpuMat, _ ...*Stream) *ComplexMat` | DFT computes the forward two-dimensional discrete Fourier transform of a single-channel GpuMat and returns the complex spectrum, mirroring… |
| `func IDFTComplex(spec *ComplexMat, _ ...*Stream) *ComplexMat` | IDFTComplex computes the inverse two-dimensional DFT and returns the full complex result (normalised by 1/(rows*cols)) without rounding to bytes. |
| `func MulSpectrums(a, b *ComplexMat, conjB bool, _ ...*Stream) *ComplexMat` | MulSpectrums multiplies two spectra element-wise, mirroring cv::cuda::mulSpectrums. |
| `func NewComplexMat(rows, cols int) *ComplexMat` | NewComplexMat allocates a zero-filled ComplexMat. |
| `func (c *ComplexMat) At(y, x int) (re, im float64)` | At returns the real and imaginary parts at row y, column x. |

</details>

<details>
<summary><code>GpuMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Abs(src *GpuMat, _ ...*Stream) *GpuMat` | Abs returns \|src\| per sample. |
| `func AbsDiff(a, b *GpuMat, _ ...*Stream) *GpuMat` | AbsDiff returns the per-sample absolute difference \|a-b\|, delegating to `cv.AbsDiff`. |
| `func Add(a, b *GpuMat, _ ...*Stream) *GpuMat` | Add returns the per-sample saturating sum a+b as a new GpuMat. |
| `func AddWeighted(a *GpuMat, alpha float64, b *GpuMat, beta, gamma float64, _ ...*Stream) *GpuMat` | AddWeighted returns the per-sample weighted sum alpha*a + beta*b + gamma, rounded and saturated, delegating to `cv.AddWeighted`. |
| `func BitwiseAnd(a, b *GpuMat, _ ...*Stream) *GpuMat` | BitwiseAnd returns the per-sample bitwise AND of a and b, delegating to `cv.BitwiseAnd`. |
| `func BitwiseNot(src *GpuMat, _ ...*Stream) *GpuMat` | BitwiseNot returns the per-sample bitwise complement (255-value) of src, delegating to `cv.BitwiseNot`. |
| `func BitwiseOr(a, b *GpuMat, _ ...*Stream) *GpuMat` | BitwiseOr returns the per-sample bitwise OR of a and b, delegating to `cv.BitwiseOr`. |
| `func BitwiseXor(a, b *GpuMat, _ ...*Stream) *GpuMat` | BitwiseXor returns the per-sample bitwise XOR of a and b, delegating to `cv.BitwiseXor`. |
| `func CartToPolar(x, y *GpuMat, angleInDegrees bool, _ ...*Stream) (magnitude, angle *GpuMat)` | CartToPolar converts the Cartesian vectors (x, y) to polar form, returning the magnitude and angle GpuMats, mirroring cv::cuda::cartToPolar. |
| `func Compare(a, b *GpuMat, op CmpOp, _ ...*Stream) *GpuMat` | Compare compares a and b per sample and returns a mask that is 255 where the relation op holds and 0 elsewhere, matching cv::cuda::compare. |
| `func Divide(a, b *GpuMat, scale float64, _ ...*Stream) *GpuMat` | Divide returns the per-sample quotient scale*a/b (0 where b is 0), rounded and saturated, delegating to `cv.Divide`. |
| `func Exp(src *GpuMat, _ ...*Stream) *GpuMat` | Exp returns e raised to each sample, rounded and saturated, mirroring cv::cuda::exp. |
| `func Flip(src *GpuMat, code cv.FlipCode, _ ...*Stream) *GpuMat` | Flip mirrors src along the axis chosen by code, mirroring cv::cuda::flip. |
| `func Gemm(a, b *GpuMat, alpha float64, c *GpuMat, beta float64, _ ...*Stream) *GpuMat` | Gemm computes the saturating general matrix product alpha*A*B + beta*C and returns it as a single-channel GpuMat, mirroring cv::cuda::gemm (without… |
| `func IDFT(spec *ComplexMat, _ ...*Stream) *GpuMat` | IDFT computes the inverse two-dimensional DFT of a spectrum and returns the real part as a rounded, saturated single-channel GpuMat, mirroring… |
| `func LUT(src *GpuMat, table []uint8, _ ...*Stream) *GpuMat` | LUT remaps every sample through a 256-entry lookup table, mirroring cv::cuda::LUT (the OpenCV CUDA LookUpTable). |
| `func Log(src *GpuMat, _ ...*Stream) *GpuMat` | Log returns the natural logarithm of each sample, rounded and saturated, mirroring cv::cuda::log. |
| `func Magnitude(x, y *GpuMat, _ ...*Stream) *GpuMat` | Magnitude returns the per-sample vector magnitude sqrt(x^2 + y^2), rounded and saturated, mirroring cv::cuda::magnitude. |
| `func Max(a, b *GpuMat, _ ...*Stream) *GpuMat` | Max returns the per-sample maximum of a and b, delegating to `cv.Max`. |
| `func Merge(planes []*GpuMat, _ ...*Stream) *GpuMat` | Merge interleaves single-channel GpuMats into one multi-channel GpuMat, mirroring cv::cuda::merge. |
| `func Min(a, b *GpuMat, _ ...*Stream) *GpuMat` | Min returns the per-sample minimum of a and b, delegating to `cv.Min`. |
| `func Multiply(a, b *GpuMat, scale float64, _ ...*Stream) *GpuMat` | Multiply returns the per-sample product a*b*scale, rounded and saturated, delegating to `cv.Multiply`. |
| `func NewGpuMat(src *cv.Mat) *GpuMat` | NewGpuMat returns a GpuMat holding a copy of src, mimicking an upload of a host matrix to the device. |
| `func Normalize(src *GpuMat, alpha, beta float64, normType NormType, _ ...*Stream) *GpuMat` | Normalize rescales src according to normType and returns a new GpuMat, mirroring cv::cuda::normalize. |
| `func Phase(x, y *GpuMat, angleInDegrees bool, _ ...*Stream) *GpuMat` | Phase returns the per-sample orientation atan2(y, x) of the vectors, mirroring cv::cuda::phase. |
| `func PolarToCart(magnitude, angle *GpuMat, angleInDegrees bool, _ ...*Stream) (x, y *GpuMat)` | PolarToCart converts the polar vectors (magnitude, angle) back to Cartesian form, returning the x and y GpuMats, mirroring cv::cuda::polarToCart. |
| `func Pow(src *GpuMat, power float64, _ ...*Stream) *GpuMat` | Pow raises each sample of src to power and returns the rounded, saturated result, mirroring cv::cuda::pow. |
| `func Split(src *GpuMat, _ ...*Stream) []*GpuMat` | Split separates a multi-channel GpuMat into one single-channel GpuMat per channel, in channel order, mirroring cv::cuda::split. |
| `func Sqrt(src *GpuMat, _ ...*Stream) *GpuMat` | Sqrt returns the square root of each sample, rounded and saturated, mirroring cv::cuda::sqrt. |
| `func Subtract(a, b *GpuMat, _ ...*Stream) *GpuMat` | Subtract returns the per-sample saturating difference a-b (floored at 0), delegating to `cv.Subtract`. |
| `func Threshold(src *GpuMat, thresh, maxval float64, typ cv.ThresholdType, _ ...*Stream) (*GpuMat, float64)` | Threshold applies a fixed-level threshold to a single-channel GpuMat and returns the thresholded result together with the threshold used, delegating… |
| `func Transpose(src *GpuMat, _ ...*Stream) *GpuMat` | Transpose swaps rows and columns, returning a new GpuMat of shape Cols×Rows, mirroring cv::cuda::transpose. |
| `func (g *GpuMat) Channels() int` | Channels returns the number of channels, or 0 when empty. |
| `func (g *GpuMat) Clone() *GpuMat` | Clone returns a deep copy of the GpuMat with its own backing storage. |
| `func (g *GpuMat) Download() *cv.Mat` | Download returns a copy of the GpuMat's contents as a host `cv.Mat`, the analogue of cv::cuda::GpuMat::download. |
| `func (g *GpuMat) Empty() bool` | Empty reports whether the GpuMat holds no samples. |
| `func (g *GpuMat) Mat() *cv.Mat` | Mat returns the wrapped `cv.Mat` without copying, for read-only interop with the root package. |
| `func (g *GpuMat) Release()` | Release frees the GpuMat's contents, leaving it empty. |
| `func (g *GpuMat) Size() (rows, cols int)` | Size returns the matrix dimensions as (rows, cols), or (0, 0) when empty. |
| `func (g *GpuMat) Type() MatType` | Type returns the `MatType` of the matrix. |
| `func (g *GpuMat) Upload(src *cv.Mat)` | Upload copies src into the GpuMat, replacing any previous contents. |

</details>

<details>
<summary><code>MatType</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MakeType(channels int) MatType` | MakeType returns the CV_8U MatType for the given channel count, mirroring OpenCV's CV_8UC(n). |
| `func (t MatType) Channels() int` | Channels returns the number of channels encoded by the type. |
| `func (t MatType) String() string` | String renders the type as OpenCV's CV_8UC<n> spelling. |

</details>

<details>
<summary><code>Stream</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStream() *Stream` | NewStream returns a ready-to-use no-op Stream. |
| `func (s *Stream) WaitForCompletion()` | WaitForCompletion returns immediately. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/cudaarithm).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
