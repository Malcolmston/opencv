# cudastereo — standard-library-only, CPU-backed mirror of OpenCV's cudastereo module (cv::cuda stereo

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/cudastereo.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/cudastereo)

Package cudastereo is a standard-library-only, CPU-backed mirror of OpenCV's
cudastereo module (cv::cuda stereo correspondence). It reproduces that
module's public API — a `GpuMat` device-matrix wrapper, a no-op `Stream`,
and the StereoBM / StereoBeliefPropagation / StereoConstantSpaceBP / StereoSGM
matchers plus the DisparityBilateralFilter, reprojectImageTo3D and
drawColorDisp helpers — so code written against the CUDA module ports across
with only cosmetic changes.

There is no GPU here and no CUDA. This package is pure Go: it imports only the
Go standard library, the root cv package (github.com/malcolmston/opencv,
aliased cv) and the sibling github.com/malcolmston/opencv/stereo package.
There is no cgo, no device memory and no kernel launch. `GpuMat` is a thin
wrapper around a host `github.com/malcolmston/opencv.Mat`; Upload and Download
are ordinary deep copies; `Stream` is an inert placeholder whose
WaitForCompletion returns immediately because every operation already ran
synchronously on the CPU. Passing a nil `Stream` is always valid.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/cudastereo"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./cudastereo/`.

```go
left, right := makePair(64, 24, 8)
	l := cudastereo.NewGpuMatFromMat(left)
	r := cudastereo.NewGpuMatFromMat(right)

	bm := cudastereo.CreateStereoBM(16, 7)
	disp := bm.Compute(l, r, nil).Download()

	fmt.Println("disparity:", disp.Data[12*disp.Cols+50])
```

```
disparity: 8
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ReprojectImageTo3D(disparity *GpuMat, Q [4][4]float64, stream *Stream) [][3]float64` | ReprojectImageTo3D maps a disparity `GpuMat` to 3-D coordinates through the 4×4 reprojection matrix Q, mirroring cv::cuda::reprojectImageTo3D. It… |

### Types

| Type | What it is |
| --- | --- |
| `DisparityBilateralFilter` | DisparityBilateralFilter is a CPU-backed mirror of cv::cuda::DisparityBilateralFilter: an edge-preserving refinement of a disparity map guided by the… |
| `GpuMat` | GpuMat is a CPU-backed stand-in for cv::cuda::GpuMat: the device-resident matrix type of OpenCV's CUDA modules. |
| `SGMMode` | SGMMode selects the aggregation-path layout of `StereoSGM`, mirroring the CUDA module's MODE_HH4 / MODE_HH flags. |
| `StereoBM` | StereoBM is the CPU-backed mirror of cv::cuda::StereoBM: local block matching over a rectified pair. |
| `StereoBeliefPropagation` | StereoBeliefPropagation is a CPU-backed mirror of cv::cuda::StereoBeliefPropagation and a genuine hierarchical loopy belief-propagation stereo… |
| `StereoConstantSpaceBP` | StereoConstantSpaceBP is a CPU-backed mirror of cv::cuda::StereoConstantSpaceBP and a genuine constant-space belief propagation stereo matcher. |
| `StereoSGM` | StereoSGM is the CPU-backed mirror of cv::cuda::StereoSGM: semi-global matching over a rectified pair. |
| `Stream` | Stream is a CPU-backed placeholder for cv::cuda::Stream. |

<details>
<summary><code>DisparityBilateralFilter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateDisparityBilateralFilter(ndisp, radius, iters int) *DisparityBilateralFilter` | CreateDisparityBilateralFilter constructs a filter, mirroring cv::cuda::createDisparityBilateralFilter(ndisp, radius, iters). |
| `func (f *DisparityBilateralFilter) Apply(disparity, image *GpuMat, stream *Stream) *GpuMat` | Apply refines disparity using image as the guidance reference and returns a new single-channel 8-bit disparity map as a `GpuMat`, mirroring… |

</details>

<details>
<summary><code>GpuMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DrawColorDisp(disparity *GpuMat, ndisp int, stream *Stream) *GpuMat` | DrawColorDisp renders a disparity `GpuMat` as a three-channel RGB colour image for visualisation, mirroring cv::cuda::drawColorDisp. |
| `func NewGpuMat(rows, cols, channels int) *GpuMat` | NewGpuMat allocates a zero-filled matrix of the given shape in host memory, mirroring the cv::cuda::GpuMat(rows, cols, type) constructor. |
| `func NewGpuMatFromMat(m *cv.Mat) *GpuMat` | NewGpuMatFromMat returns a GpuMat holding an independent copy of the host matrix m, the equivalent of default-constructing a GpuMat and calling… |
| `func (g *GpuMat) Channels() int` | Channels returns the number of samples per pixel, or 0 when empty. |
| `func (g *GpuMat) Clone() *GpuMat` | Clone returns a deep copy of the matrix with its own backing storage. |
| `func (g *GpuMat) Cols() int` | Cols returns the number of columns (width), or 0 when empty. |
| `func (g *GpuMat) Download() *cv.Mat` | Download returns an independent copy of the matrix, like cv::cuda::GpuMat::download. |
| `func (g *GpuMat) Empty() bool` | Empty reports whether the GpuMat holds no samples. |
| `func (g *GpuMat) Mat() *cv.Mat` | Mat returns the underlying host matrix without copying. |
| `func (g *GpuMat) Release()` | Release drops the backing storage, leaving the GpuMat empty, like cv::cuda::GpuMat::release. |
| `func (g *GpuMat) Rows() int` | Rows returns the number of rows (height), or 0 when empty. |
| `func (g *GpuMat) Size() (rows, cols int)` | Size returns the matrix dimensions as (rows, cols). |
| `func (g *GpuMat) Upload(host *cv.Mat)` | Upload copies the matrix host into this GpuMat, replacing any previous contents, like cv::cuda::GpuMat::upload. |

</details>

<details>
<summary><code>StereoBM</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateStereoBM(numDisparities, blockSize int) *StereoBM` | CreateStereoBM constructs a StereoBM with the given search range and window size, mirroring cv::cuda::createStereoBM(numDisparities, blockSize). |
| `func (bm *StereoBM) Compute(left, right *GpuMat, stream *Stream) *GpuMat` | Compute matches left against right and returns a single-channel 8-bit disparity map as a `GpuMat`, mirroring cv::cuda::StereoBM::compute. |

</details>

<details>
<summary><code>StereoBeliefPropagation</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateStereoBeliefPropagation(ndisp, iters, levels int) *StereoBeliefPropagation` | CreateStereoBeliefPropagation constructs a matcher, mirroring cv::cuda::createStereoBeliefPropagation(ndisp, iters, levels). |
| `func (bp *StereoBeliefPropagation) Compute(left, right *GpuMat, stream *Stream) *GpuMat` | Compute matches left against right and returns a single-channel 8-bit disparity map as a `GpuMat`, mirroring… |
| `func (bp *StereoBeliefPropagation) EstimateRecommendedParams(width, height int) (ndisp, iters, levels int)` | EstimateRecommendedParams fills in disparity range, iteration count and level count appropriate for an image of the given size, mirroring… |

</details>

<details>
<summary><code>StereoConstantSpaceBP</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateStereoConstantSpaceBP(ndisp, iters, levels, nrPlane int) *StereoConstantSpaceBP` | CreateStereoConstantSpaceBP constructs a matcher, mirroring cv::cuda::createStereoConstantSpaceBP(ndisp, iters, levels, nr_plane). |
| `func (csbp *StereoConstantSpaceBP) Compute(left, right *GpuMat, stream *Stream) *GpuMat` | Compute matches left against right and returns a single-channel 8-bit disparity map as a `GpuMat`, mirroring cv::cuda::StereoConstantSpaceBP::compute. |
| `func (csbp *StereoConstantSpaceBP) EstimateRecommendedParams(width, height int) (ndisp, iters, levels, nrPlane int)` | EstimateRecommendedParams fills in disparity range, iterations, levels and plane count appropriate for an image of the given size, mirroring… |

</details>

<details>
<summary><code>StereoSGM</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateStereoSGM(minDisparity, numDisparities, p1, p2, uniquenessRatio int, mode SGMMode) *StereoSGM` | CreateStereoSGM constructs a StereoSGM, mirroring cv::cuda::createStereoSGM(minDisparity, numDisparities, P1, P2, uniquenessRatio, mode). |
| `func (s *StereoSGM) Compute(left, right *GpuMat, stream *Stream) *GpuMat` | Compute matches left against right and returns a single-channel 8-bit disparity map as a `GpuMat`, mirroring cv::cuda::StereoSGM::compute. |

</details>

<details>
<summary><code>Stream</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStream() *Stream` | NewStream returns a ready-to-use stream. |
| `func (s *Stream) WaitForCompletion()` | WaitForCompletion blocks until all work queued on the stream has finished, like cv::cuda::Stream::waitForCompletion. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/cudastereo).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
