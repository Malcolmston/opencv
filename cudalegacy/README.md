# cudalegacy — CPU-backed, API-compatible mirror of OpenCV's cudalegacy module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/cudalegacy.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/cudalegacy)

Package cudalegacy is a CPU-backed, API-compatible mirror of OpenCV's
cudalegacy module — the grab-bag of "device functions" and legacy NPP/NCV
helpers that historically lived under the cv::cuda and NCV namespaces
(background subtraction, image pyramids, connected-component labelling on the
device, block-matching optical flow, motion-compensated frame interpolation,
needle-map visualisation, graph-cut segmentation and planar pose estimation).

There is NO CUDA, NO GPU and NO cgo here. Every type and function is pure Go
built only on the standard library, the root package
`github.com/malcolmston/opencv` (imported under the alias cv) and the sibling
CPU background-segmentation package `github.com/malcolmston/opencv/bgsegm`.
The goal is source and shape compatibility: code written against the OpenCV
cudalegacy API — a `GpuMat` holding pixels, a `Stream` threaded through each
call, factory functions and algorithm objects with the same names — ports
across with minimal edits, while the actual computation runs on the CPU.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/cudalegacy"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./cudalegacy/`.

```go
img := cv.NewMat(32, 32, 1)
	pyr := cudalegacy.NewImagePyramid(cudalegacy.GpuMatFromMat(img), 3, nil)

	for i := 0; i < pyr.NumLayers(); i++ {
		r, c := pyr.Layer(i).Size()
		fmt.Printf("level %d: %dx%d\n", i, r, c)
	}
```

```
level 0: 32x32
level 1: 16x16
level 2: 8x8
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func CompactPoints(points0, points1 [][2]float64, mask []uint8) (out0, out1 [][2]float64)` | CompactPoints is a CPU-backed mirror of OpenCV's cv::cuda::compactPoints. |
| `func CreateOpticalFlowNeedleMap(flow *Flow, gridStep int, scale float64, stream *Stream) (*GpuMat, []NeedleSegment)` | CreateOpticalFlowNeedleMap is a CPU-backed mirror of OpenCV's cv::cuda::createOpticalFlowNeedleMap. |
| `func LabelComponents(mask *GpuMat, stream *Stream) (labels []int32, count int)` | LabelComponents is a CPU-backed mirror of OpenCV's cv::cuda::labelComponents. |
| `func ProjectPoints(objectPoints [][3]float64, rvec, tvec [3]float64, K [3][3]float64, dist []float64) [][2]float64` | ProjectPoints is a CPU-backed mirror of the projection performed by OpenCV's cudalegacy projectPoints. |
| `func Rodrigues(rvec [3]float64) [3][3]float64` | Rodrigues converts a rotation vector (axis-angle, whose direction is the rotation axis and whose magnitude is the rotation angle in radians) into a… |
| `func SolvePnPRansac(objectPoints [][3]float64, imagePoints [][2]float64, K [3][3]float64, dist []float64, reprojThresh float64, iterations int, rng *rand.Rand) (rvec, tvec [3]float64, inliers []bool, ok bool)` | SolvePnPRansac is a CPU-backed mirror of the pose estimation performed by OpenCV's cudalegacy solvePnPRansac, specialised to a planar target: the… |

### Types

| Type | What it is |
| --- | --- |
| `BackgroundSubtractorFGD` | BackgroundSubtractorFGD is a CPU-backed mirror of the detector returned by OpenCV's cv::cuda::createBackgroundSubtractorFGD — the "foreground… |
| `BackgroundSubtractorGMG` | BackgroundSubtractorGMG is a CPU-backed mirror of the subtractor returned by OpenCV's cv::cuda::createBackgroundSubtractorGMG — the… |
| `Flow` | Flow is a dense, per-pixel motion field: for every pixel it stores a horizontal (U) and vertical (V) displacement, both as `cv.FloatMat` planes the… |
| `GpuMat` | GpuMat is a CPU-backed stand-in for OpenCV's cv::cuda::GpuMat. |
| `ImagePyramid` | ImagePyramid is a CPU-backed mirror of OpenCV's cv::cuda::ImagePyramid. |
| `NeedleSegment` | NeedleSegment is one arrow of an optical-flow needle map: a line from Start to End (in pixel coordinates) whose direction and length depict the… |
| `Stream` | Stream is a CPU-backed no-op stand-in for OpenCV's cv::cuda::Stream. |

<details>
<summary><code>BackgroundSubtractorFGD</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateBackgroundSubtractorFGD(history int) *BackgroundSubtractorFGD` | CreateBackgroundSubtractorFGD creates an FGD detector, mirroring cv::cuda::createBackgroundSubtractorFGD. |
| `func (b *BackgroundSubtractorFGD) Apply(frame *GpuMat, learningRate float64, stream *Stream) *GpuMat` | Apply learns from or classifies frame and returns the foreground mask wrapped in a fresh `GpuMat`. |
| `func (b *BackgroundSubtractorFGD) GetBackgroundImage(stream *Stream) *GpuMat` | GetBackgroundImage returns the model's current per-pixel mean as a fresh `GpuMat` the same shape as the learned frames, mirroring getBackgroundImage. |

</details>

<details>
<summary><code>BackgroundSubtractorGMG</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateBackgroundSubtractorGMG(initializationFrames int, decisionThreshold float64) *BackgroundSubtractorGMG` | CreateBackgroundSubtractorGMG creates a cudalegacy-style GMG subtractor, mirroring cv::cuda::createBackgroundSubtractorGMG(initializationFrames,… |
| `func (b *BackgroundSubtractorGMG) Apply(frame *GpuMat, learningRate float64, stream *Stream) *GpuMat` | Apply learns from or classifies frame and returns the foreground mask wrapped in a fresh `GpuMat`. |
| `func (b *BackgroundSubtractorGMG) GetBackgroundImage(stream *Stream) *GpuMat` | GetBackgroundImage returns the model's current background estimate wrapped in a `GpuMat`. |
| `func (b *BackgroundSubtractorGMG) GetDecisionThreshold() float64` | GetDecisionThreshold returns the foreground-probability threshold above which a pixel is classified as foreground. |
| `func (b *BackgroundSubtractorGMG) GetNumFrames() int` | GetNumFrames returns the length of the initial learning period. |
| `func (b *BackgroundSubtractorGMG) SetDecisionThreshold(t float64)` | SetDecisionThreshold sets the foreground-probability threshold above which a pixel is classified as foreground. |
| `func (b *BackgroundSubtractorGMG) SetNumFrames(n int)` | SetNumFrames sets the length of the initial learning period. |

</details>

<details>
<summary><code>Flow</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CalcOpticalFlowBM(prev, curr *GpuMat, blockSize, shiftSize, maxRange int, stream *Stream) *Flow` | CalcOpticalFlowBM is a CPU-backed mirror of OpenCV's cv::cuda::calcOpticalFlowBM — dense optical flow by exhaustive block matching. |
| `func NewFlow(rows, cols int) *Flow` | NewFlow allocates a zero-filled Flow of the given size. |
| `func (f *Flow) At(y, x int) (u, v float64)` | At returns the (u, v) displacement stored at row y, column x. |
| `func (f *Flow) Cols() int` | Cols returns the field width. |
| `func (f *Flow) Rows() int` | Rows returns the field height. |

</details>

<details>
<summary><code>GpuMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ConnectivityMask(image *GpuMat, loThreshold, hiThreshold float64, conn cv.Connectivity, stream *Stream) *GpuMat` | ConnectivityMask is a CPU-backed mirror of OpenCV's cv::cuda::connectivityMask. |
| `func GpuMatFromMat(m *cv.Mat) *GpuMat` | GpuMatFromMat wraps an existing host matrix as a GpuMat without copying it. |
| `func GraphCut(sourceCap, sinkCap *cv.FloatMat, lambda float64, stream *Stream) *GpuMat` | GraphCut is a CPU-backed mirror of the binary segmentation performed by OpenCV's cv::cuda::graphcut. |
| `func InterpolateFrames(frame0, frame1 *GpuMat, forward, backward *Flow, pos float64, stream *Stream) *GpuMat` | InterpolateFrames is a CPU-backed mirror of OpenCV's cv::cuda::interpolateFrames. |
| `func NewGpuMat() *GpuMat` | NewGpuMat returns an empty GpuMat with no backing matrix. |
| `func RenderLabels(labels []int32, rows, cols int) *GpuMat` | RenderLabels turns a label slice from `LabelComponents` into a viewable single-channel `GpuMat` of the given size, mapping each label to a distinct… |
| `func (g *GpuMat) Channels() int` | Channels returns the channel count of the wrapped matrix, or 0 when empty. |
| `func (g *GpuMat) Clone() *GpuMat` | Clone returns a deep copy of the GpuMat with its own backing matrix, mirroring cv::cuda::GpuMat::clone. |
| `func (g *GpuMat) Download(stream *Stream) *cv.Mat` | Download returns a host copy of the wrapped matrix, standing in for cv::cuda::GpuMat::download. |
| `func (g *GpuMat) Empty() bool` | Empty reports whether the GpuMat has no samples, mirroring cv::cuda::GpuMat::empty. |
| `func (g *GpuMat) Release()` | Release drops the reference to the wrapped matrix, leaving the GpuMat empty. |
| `func (g *GpuMat) Size() (rows, cols int)` | Size returns the dimensions of the wrapped matrix as (rows, cols), or (0, 0) when the GpuMat is empty. |
| `func (g *GpuMat) Upload(m *cv.Mat, stream *Stream)` | Upload copies host matrix m into this GpuMat, standing in for cv::cuda::GpuMat::upload. |

</details>

<details>
<summary><code>ImagePyramid</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewImagePyramid(img *GpuMat, numLayers int, stream *Stream) *ImagePyramid` | NewImagePyramid builds a pyramid of up to numLayers levels from img, mirroring the ImagePyramid constructor. |
| `func (p *ImagePyramid) GetLayer(outRows, outCols int, stream *Stream) *GpuMat` | GetLayer returns a fresh `GpuMat` holding the pyramid content resampled to outRows×outCols, mirroring cv::cuda::ImagePyramid::getLayer. |
| `func (p *ImagePyramid) Layer(i int) *GpuMat` | Layer returns a copy of stored pyramid level i (level 0 is the original), as a convenience beyond the OpenCV surface. |
| `func (p *ImagePyramid) NumLayers() int` | NumLayers returns the number of stored pyramid levels. |

</details>

<details>
<summary><code>Stream</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStream() *Stream` | NewStream returns a new no-op Stream, mirroring the cv::cuda::Stream constructor. |
| `func (s *Stream) QueryIfComplete() bool` | QueryIfComplete always reports true, since work on this backend is synchronous and already finished by the time any call returns. |
| `func (s *Stream) WaitForCompletion()` | WaitForCompletion returns immediately: there is no asynchronous work to wait for. |

</details>

### Constants

`MaskConnectEast`, `MaskConnectSouth`, `MaskConnectWest`, `MaskConnectNorth`, `MaskConnectSE`, `MaskConnectSW`, `MaskConnectNW`, `MaskConnectNE`, `SourceLabel`, `SinkLabel`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/cudalegacy).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
