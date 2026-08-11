# cudabgsegm — CPU-backed, API-compatible mirror of OpenCV's cudabgsegm module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/cudabgsegm.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/cudabgsegm)

Package cudabgsegm is a CPU-backed, API-compatible mirror of OpenCV's
cudabgsegm module — the GPU background-segmentation subtractors that in
upstream OpenCV live under the cv::cuda namespace
(createBackgroundSubtractorMOG, createBackgroundSubtractorMOG2,
createBackgroundSubtractorGMG and createBackgroundSubtractorFGD).

There is no CUDA, no GPU and no cgo here. This package is pure Go built only
on the standard library, the root package `github.com/malcolmston/opencv`
(imported under the alias cv) and the sibling CPU package
`github.com/malcolmston/opencv/bgsegm`, to which every model delegates. The
point is source and shape compatibility: code written against the OpenCV CUDA
bgsegm API — a `GpuMat` holding the frame, a `Stream` threaded through each
call, factory functions named createBackgroundSubtractor* and the matching
getter/setter methods — ports across with minimal edits, while the actual
pixels are classified on the CPU by the `github.com/malcolmston/opencv/bgsegm`
models.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/cudabgsegm"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./cudabgsegm/`.

```go
sub := cudabgsegm.CreateBackgroundSubtractorMOG2(10, 16, false)
	stream := cudabgsegm.NewStream()

	for i := 0; i < 30; i++ {
		sub.Apply(cudabgsegm.GpuMatFromMat(fill(8, 8, 40)), -1, stream)
	}

	frame := fill(8, 8, 40)
	for y := 2; y < 5; y++ {
		for x := 3; x < 6; x++ {
			frame.Set(y, x, 0, 220)
		}
	}
	mask := sub.Apply(cudabgsegm.GpuMatFromMat(frame), -1, stream)

	fmt.Println("foreground pixels:", fg(mask))
```

```
foreground pixels: 9
```

## Exported surface

### Types

| Type | What it is |
| --- | --- |
| `BackgroundSubtractorGMG` | BackgroundSubtractorGMG is a CPU-backed mirror of the subtractor returned by OpenCV's cv::cuda::createBackgroundSubtractorGMG — the… |
| `BackgroundSubtractorMOG` | BackgroundSubtractorMOG is a CPU-backed mirror of the subtractor returned by OpenCV's cv::cuda::createBackgroundSubtractorMOG — the original… |
| `BackgroundSubtractorMOG2` | BackgroundSubtractorMOG2 is a CPU-backed mirror of the subtractor returned by OpenCV's cv::cuda::createBackgroundSubtractorMOG2 — the Zivkovic… |
| `GpuMat` | GpuMat is a CPU-backed stand-in for OpenCV's cv::cuda::GpuMat. |
| `Stream` | Stream is a CPU-backed no-op stand-in for OpenCV's cv::cuda::Stream. |

<details>
<summary><code>BackgroundSubtractorGMG</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateBackgroundSubtractorGMG(initializationFrames int, decisionThreshold float64) *BackgroundSubtractorGMG` | CreateBackgroundSubtractorGMG creates a CUDA-style GMG subtractor, mirroring cv::cuda::createBackgroundSubtractorGMG(initializationFrames,… |
| `func (b *BackgroundSubtractorGMG) Apply(frame *GpuMat, learningRate float64, stream *Stream) *GpuMat` | Apply learns from or classifies frame and returns the foreground mask wrapped in a fresh `GpuMat`. |
| `func (b *BackgroundSubtractorGMG) GetBackgroundImage(stream *Stream) *GpuMat` | GetBackgroundImage returns the model's current background estimate wrapped in a `GpuMat`. |
| `func (b *BackgroundSubtractorGMG) GetDecisionThreshold() float64` | GetDecisionThreshold returns the foreground-probability threshold above which a pixel is classified as foreground. |
| `func (b *BackgroundSubtractorGMG) GetNumFrames() int` | GetNumFrames returns the length of the initial learning period. |
| `func (b *BackgroundSubtractorGMG) SetDecisionThreshold(t float64)` | SetDecisionThreshold sets the foreground-probability threshold above which a pixel is classified as foreground. |
| `func (b *BackgroundSubtractorGMG) SetNumFrames(n int)` | SetNumFrames sets the length of the initial learning period. |

</details>

<details>
<summary><code>BackgroundSubtractorMOG</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateBackgroundSubtractorMOG(history, nmixtures int, backgroundRatio, noiseSigma float64) *BackgroundSubtractorMOG` | CreateBackgroundSubtractorMOG creates a CUDA-style MOG subtractor, mirroring cv::cuda::createBackgroundSubtractorMOG(history, nmixtures,… |
| `func (b *BackgroundSubtractorMOG) Apply(frame *GpuMat, learningRate float64, stream *Stream) *GpuMat` | Apply classifies frame against the model, updates the model and returns the foreground mask wrapped in a fresh `GpuMat`. |
| `func (b *BackgroundSubtractorMOG) GetBackgroundImage(stream *Stream) *GpuMat` | GetBackgroundImage returns the model's current background estimate wrapped in a `GpuMat`, mirroring… |
| `func (b *BackgroundSubtractorMOG) GetBackgroundRatio() float64` | GetBackgroundRatio returns the fraction of mixture weight that defines the background. |
| `func (b *BackgroundSubtractorMOG) GetHistory() int` | GetHistory returns the model's history length. |
| `func (b *BackgroundSubtractorMOG) GetNMixtures() int` | GetNMixtures returns the number of Gaussians kept per pixel. |
| `func (b *BackgroundSubtractorMOG) GetNoiseSigma() float64` | GetNoiseSigma returns the standard deviation assigned to a freshly spawned Gaussian. |
| `func (b *BackgroundSubtractorMOG) SetBackgroundRatio(r float64)` | SetBackgroundRatio sets the fraction of mixture weight that defines the background. |
| `func (b *BackgroundSubtractorMOG) SetHistory(history int)` | SetHistory sets the model's history length. |
| `func (b *BackgroundSubtractorMOG) SetNMixtures(n int)` | SetNMixtures sets the number of Gaussians kept per pixel. |
| `func (b *BackgroundSubtractorMOG) SetNoiseSigma(sigma float64)` | SetNoiseSigma sets the standard deviation assigned to a freshly spawned Gaussian. |

</details>

<details>
<summary><code>BackgroundSubtractorMOG2</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateBackgroundSubtractorMOG2(history int, varThreshold float64, detectShadows bool) *BackgroundSubtractorMOG2` | CreateBackgroundSubtractorMOG2 creates a CUDA-style MOG2 subtractor, mirroring cv::cuda::createBackgroundSubtractorMOG2(history, varThreshold,… |
| `func (b *BackgroundSubtractorMOG2) Apply(frame *GpuMat, learningRate float64, stream *Stream) *GpuMat` | Apply classifies frame against the model, updates the model and returns the foreground mask wrapped in a fresh `GpuMat`. |
| `func (b *BackgroundSubtractorMOG2) GetBackgroundImage(stream *Stream) *GpuMat` | GetBackgroundImage returns the model's current background estimate wrapped in a `GpuMat`, mirroring… |
| `func (b *BackgroundSubtractorMOG2) GetDetectShadows() bool` | GetDetectShadows reports whether shadow classification is enabled. |
| `func (b *BackgroundSubtractorMOG2) GetHistory() int` | GetHistory returns the model's history length. |
| `func (b *BackgroundSubtractorMOG2) GetVarThreshold() float64` | GetVarThreshold returns the squared-Mahalanobis matching threshold. |
| `func (b *BackgroundSubtractorMOG2) SetDetectShadows(on bool)` | SetDetectShadows enables or disables shadow classification. |
| `func (b *BackgroundSubtractorMOG2) SetHistory(history int)` | SetHistory sets the model's history length. |
| `func (b *BackgroundSubtractorMOG2) SetVarThreshold(t float64)` | SetVarThreshold sets the squared-Mahalanobis matching threshold. |

</details>

<details>
<summary><code>GpuMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func GpuMatFromMat(m *cv.Mat) *GpuMat` | GpuMatFromMat wraps an existing host matrix as a GpuMat without copying it. |
| `func NewGpuMat() *GpuMat` | NewGpuMat returns an empty GpuMat with no backing matrix. |
| `func (g *GpuMat) Channels() int` | Channels returns the channel count of the wrapped matrix, or 0 when empty. |
| `func (g *GpuMat) Clone() *GpuMat` | Clone returns a deep copy of the GpuMat with its own backing matrix, mirroring cv::cuda::GpuMat::clone. |
| `func (g *GpuMat) Download(stream *Stream) *cv.Mat` | Download returns a host copy of the wrapped matrix, standing in for cv::cuda::GpuMat::download. |
| `func (g *GpuMat) Empty() bool` | Empty reports whether the GpuMat has no samples, mirroring cv::cuda::GpuMat::empty. |
| `func (g *GpuMat) Size() (rows, cols int)` | Size returns the dimensions of the wrapped matrix as (rows, cols), or (0, 0) when the GpuMat is empty. |
| `func (g *GpuMat) Upload(m *cv.Mat, stream *Stream)` | Upload copies host matrix m into this GpuMat, standing in for cv::cuda::GpuMat::upload. |

</details>

<details>
<summary><code>Stream</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStream() *Stream` | NewStream returns a new no-op Stream, mirroring the cv::cuda::Stream constructor. |
| `func (s *Stream) QueryIfComplete() bool` | QueryIfComplete always reports true, since work on this backend is synchronous and already finished by the time any call returns. |
| `func (s *Stream) WaitForCompletion()` | WaitForCompletion returns immediately: there is no asynchronous work to wait for. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/cudabgsegm).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
