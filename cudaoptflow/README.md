# cudaoptflow — CPU-backed, API-compatible mirror of OpenCV's cudaoptflow module (cv::cuda dense

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/cudaoptflow.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/cudaoptflow)

Package cudaoptflow is a CPU-backed, API-compatible mirror of OpenCV's
cudaoptflow module (cv::cuda dense and sparse optical flow).

This package contains NO CUDA, NO cgo and NO GPU code. It is written entirely
against the Go standard library, the root module github.com/malcolmston/opencv
(imported as cv) and the two sibling optical-flow packages
github.com/malcolmston/opencv/optflow and github.com/malcolmston/opencv/video.
The word "Gpu" in the type names is kept only so that code ported from
OpenCV's C++/Python cuda bindings reads the same way; every buffer lives in
ordinary host memory and every computation runs on the CPU. A `GpuMat` is a
thin wrapper around a *cv.Mat and a `Stream` is a no-op placeholder for
cv::cuda::Stream — there is nothing asynchronous to wait for. Nothing here
is faster than the equivalent CPU call; the value is a drop-in API surface,
not acceleration.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/cudaoptflow"
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func DecodeFlow(g *GpuMat, scale float64) *optflow.FlowField` | DecodeFlow unpacks a two-channel uint8 GpuMat produced by `EncodeFlow` back into an `optflow.FlowField`, inverting the fixed-point encoding as… |

### Types

| Type | What it is |
| --- | --- |
| `BroxOpticalFlow` | BroxOpticalFlow is the CPU-backed mirror of cv::cuda::BroxOpticalFlow. |
| `DensePyrLKOpticalFlow` | DensePyrLKOpticalFlow is the CPU-backed mirror of cv::cuda::DensePyrLKOpticalFlow: it produces a displacement for every pixel with pyramidal… |
| `FarnebackOpticalFlow` | FarnebackOpticalFlow is the CPU-backed mirror of cv::cuda::FarnebackOpticalFlow. |
| `GpuMat` | GpuMat is a CPU-backed stand-in for cv::cuda::GpuMat. |
| `NvidiaHWOpticalFlow` | NvidiaHWOpticalFlow mirrors cv::cuda::NvidiaHWOpticalFlow, which in OpenCV drives the fixed-function NVIDIA Optical Flow Accelerator (NVOFA) present… |
| `OpticalFlowDualTVL1` | OpticalFlowDualTVL1 is the CPU-backed mirror of cv::cuda::OpticalFlowDual_TVL1: the duality-based TV-L1 method of Zach, Pock & Bischof (an L1… |
| `SparsePyrLKOpticalFlow` | SparsePyrLKOpticalFlow is the CPU-backed mirror of cv::cuda::SparsePyrLKOpticalFlow: pyramidal Lucas-Kanade tracking of a sparse set of points from… |
| `Stream` | Stream is a no-op stand-in for cv::cuda::Stream. |

<details>
<summary><code>BroxOpticalFlow</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultBroxOpticalFlow() *BroxOpticalFlow` | DefaultBroxOpticalFlow returns a Brox estimator with parameters tuned for interactive CPU use. |
| `func NewBroxOpticalFlow(alpha, gamma, scaleFactor float64, inner, outer, solver int) *BroxOpticalFlow` | NewBroxOpticalFlow creates a Brox variational estimator with explicit parameters, mirroring cv::cuda::BroxOpticalFlow::create(alpha, gamma,… |
| `func (o *BroxOpticalFlow) Calc(prev, next *GpuMat, stream *Stream) *optflow.FlowField` | Calc computes a dense flow field from prev to next with the Brox variational solver, mirroring cv::cuda::BroxOpticalFlow::calc. |

</details>

<details>
<summary><code>DensePyrLKOpticalFlow</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDensePyrLKOpticalFlow(winSize, maxLevel, iters int) *DensePyrLKOpticalFlow` | NewDensePyrLKOpticalFlow creates a dense pyramidal Lucas-Kanade estimator, mirroring cv::cuda::DensePyrLKOpticalFlow::create(winSize, maxLevel,… |
| `func (o *DensePyrLKOpticalFlow) Calc(prev, next *GpuMat, stream *Stream) *optflow.FlowField` | Calc computes a dense flow field from prev to next, mirroring cv::cuda::DensePyrLKOpticalFlow::calc. |

</details>

<details>
<summary><code>FarnebackOpticalFlow</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewFarnebackOpticalFlow(winSize, searchRadius int) *FarnebackOpticalFlow` | NewFarnebackOpticalFlow creates a dense Farneback-style estimator. |
| `func (o *FarnebackOpticalFlow) Calc(prev, next *GpuMat, stream *Stream) *optflow.FlowField` | Calc computes a dense flow field from prev to next, mirroring cv::cuda::FarnebackOpticalFlow::calc. |

</details>

<details>
<summary><code>GpuMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func EncodeFlow(f *optflow.FlowField, scale float64) *GpuMat` | EncodeFlow packs a dense `optflow.FlowField` into a two-channel uint8 GpuMat, the CPU analogue of OpenCV's CV_32FC2 flow GpuMat. |
| `func FlowToGpuMat(f *optflow.FlowField) *GpuMat` | FlowToGpuMat is EncodeFlow with the `DefaultFlowScale`. |
| `func GpuMatFromMat(m *cv.Mat) *GpuMat` | GpuMatFromMat uploads a host image into a new GpuMat, mirroring GpuMat::upload / the GpuMat(Mat) constructor. |
| `func NewGpuMat(rows, cols, channels int) *GpuMat` | NewGpuMat allocates a zero-filled GpuMat with the given dimensions, mirroring GpuMat(rows, cols, type). |
| `func (g *GpuMat) Channels() int` | Channels returns the number of samples per pixel. |
| `func (g *GpuMat) Clone() *GpuMat` | Clone returns a deep copy of the GpuMat, mirroring GpuMat::clone. |
| `func (g *GpuMat) Download() *cv.Mat` | Download returns a host copy of the contents, mirroring GpuMat::download. |
| `func (g *GpuMat) Empty() bool` | Empty reports whether the GpuMat holds no samples. |
| `func (g *GpuMat) Mat() *cv.Mat` | Mat returns the underlying *cv.Mat without copying. |
| `func (g *GpuMat) Release()` | Release frees the underlying buffer, mirroring GpuMat::release. |
| `func (g *GpuMat) Size() (rows, cols int)` | Size returns the number of rows and columns, mirroring GpuMat::size. |
| `func (g *GpuMat) ToFlowField() *optflow.FlowField` | ToFlowField is DecodeFlow with the `DefaultFlowScale`, the inverse of `FlowToGpuMat`. |
| `func (g *GpuMat) Upload(m *cv.Mat)` | Upload copies a host image into the receiver, mirroring GpuMat::upload. |

</details>

<details>
<summary><code>NvidiaHWOpticalFlow</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewNvidiaHWOpticalFlow(patchRadius, searchRadius, levels int) *NvidiaHWOpticalFlow` | NewNvidiaHWOpticalFlow creates the CPU dense-flow estimator standing in for the NVIDIA hardware optical flow. |
| `func (o *NvidiaHWOpticalFlow) Calc(prev, next *GpuMat, stream *Stream) *optflow.FlowField` | Calc computes a dense flow field from prev to next, mirroring cv::cuda::NvidiaHWOpticalFlow::calc. |

</details>

<details>
<summary><code>OpticalFlowDualTVL1</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewOpticalFlowDualTVL1() *OpticalFlowDualTVL1` | NewOpticalFlowDualTVL1 creates a TV-L1 estimator with the standard parameters (see optflow.DefaultTVL1Params), mirroring… |
| `func NewOpticalFlowDualTVL1WithParams(p optflow.TVL1Params) *OpticalFlowDualTVL1` | NewOpticalFlowDualTVL1WithParams creates a TV-L1 estimator with explicit parameters, mirroring the fuller create(tau, lambda, theta, nscales, ...)… |
| `func (o *OpticalFlowDualTVL1) Calc(prev, next *GpuMat, stream *Stream) *optflow.FlowField` | Calc computes a dense flow field from prev to next, mirroring cv::cuda::OpticalFlowDual_TVL1::calc. |

</details>

<details>
<summary><code>SparsePyrLKOpticalFlow</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSparsePyrLKOpticalFlow(winSize, maxLevel, iters int) *SparsePyrLKOpticalFlow` | NewSparsePyrLKOpticalFlow creates a sparse pyramidal Lucas-Kanade tracker, mirroring cv::cuda::SparsePyrLKOpticalFlow::create(winSize, maxLevel,… |
| `func (o *SparsePyrLKOpticalFlow) Calc(prev, next *GpuMat, prevPts []cv.Point, stream *Stream) (nextPts []cv.Point, status []uint8, err []float32)` | Calc tracks each point of prevPts from prev to next with pyramidal Lucas-Kanade, mirroring cv::cuda::SparsePyrLKOpticalFlow::calc. |

</details>

<details>
<summary><code>Stream</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStream() *Stream` | NewStream returns a new no-op Stream, mirroring the cv::cuda::Stream constructor. |
| `func (s *Stream) QueryIfComplete() bool` | QueryIfComplete always reports true: because work is synchronous, any Stream is always idle. |
| `func (s *Stream) WaitForCompletion()` | WaitForCompletion returns immediately. |

</details>

### Constants

`DefaultFlowScale`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/cudaoptflow).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
