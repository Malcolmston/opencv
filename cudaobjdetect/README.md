# cudaobjdetect — CPU-backed, API-compatible mirror of OpenCV's cv::cuda objdetect module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/cudaobjdetect.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/cudaobjdetect)

Package cudaobjdetect is a CPU-backed, API-compatible mirror of OpenCV's
cv::cuda objdetect module (opencv_contrib's cudaobjdetect). It offers the same
types and method shapes a caller would use with the CUDA classes — a
`GpuMat` container, a no-op `Stream`, a `HOG` descriptor/detector and a Haar
`CascadeClassifier` — but there is no GPU and no cgo: every operation runs
on the host in pure Go by delegating to the sibling
`github.com/malcolmston/opencv/objdetect` package.

- `GpuMat` wraps a host `github.com/malcolmston/opencv.Mat`. Upload and
Download copy between host buffers; they do not cross a device boundary. -
`Stream` is a no-op. All work is synchronous, so its methods do nothing and
passing nil is always valid. - `HOG.GetDefaultPeopleDetector` returns an
approximate matched-filter classifier synthesised from a prototype silhouette,
not OpenCV's trained INRIA weights (which cannot be reproduced without the
training data). - Detection results that OpenCV packs into a CV_32SC4 GpuMat
are carried out-of-band, because the root Mat is 8-bit only;
`CascadeClassifier.Convert` decodes them exactly as the CUDA API does.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/cudaobjdetect"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./cudaobjdetect/`.

```go
h := cudaobjdetect.NewDefaultHOG()
	h.SetSVMDetector(h.GetDefaultPeopleDetector())

	window := cv.NewMat(128, 64, 1)
	window.SetTo(128)
	locs, _ := h.Detect(cudaobjdetect.NewGpuMatFromMat(window), nil)

	fmt.Printf("descriptor size: %d, detections on flat window: %d\n",
		h.GetDescriptorSize(), len(locs))
```

```
descriptor size: 3780, detections on flat window: 0
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func GroupRectangles(rects []cv.Rect, groupThreshold int, eps float64) []cv.Rect` | GroupRectangles clusters overlapping detection rectangles and returns one averaged rectangle per cluster whose membership is at least groupThreshold,… |
| `func GroupRectanglesWeights(rects []cv.Rect, weights []float64, groupThreshold int, eps float64) ([]cv.Rect, []float64)` | GroupRectanglesWeights clusters rectangles like `GroupRectangles` but also carries a per-rectangle score; each surviving cluster reports the maximum… |
| `func NMSBoxes(boxes []cv.Rect, scores []float64, scoreThreshold, nmsThreshold float64) []int` | NMSBoxes performs greedy non-maximum suppression on scored detection boxes and returns the kept indices in descending-score order, delegating to… |
| `func RectIoU(a, b cv.Rect) float64` | RectIoU returns the intersection-over-union overlap of two rectangles, a value in [0,1], delegating to `objdetect.RectIoU`. |
| `func SoftNMSBoxes(boxes []cv.Rect, scores []float64, scoreThreshold, sigma float64) (indices []int, keptScores []float64)` | SoftNMSBoxes performs Gaussian Soft-NMS on scored detection boxes, returning the kept indices and their decayed scores, delegating to… |

### Types

| Type | What it is |
| --- | --- |
| `CascadeClassifier` | CascadeClassifier is a CPU-backed analogue of cv::cuda::CascadeClassifier: a Viola–Jones Haar cascade loaded from an OpenCV XML file and evaluated… |
| `GpuMat` | GpuMat is a CPU-backed stand-in for OpenCV's cv::cuda::GpuMat. |
| `HOG` | HOG is a CPU-backed analogue of cv::cuda::HOG: a Histogram of Oriented Gradients descriptor with a linear-SVM sliding-window detector (Dalal &… |
| `Size` | Size is a width/height pair in pixels used to configure detector geometry. |
| `Stream` | Stream is a CPU-backed no-op analogue of cv::cuda::Stream. |

<details>
<summary><code>CascadeClassifier</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func LoadCascadeFromString(xmlData string) (*CascadeClassifier, error)` | LoadCascadeFromString parses an OpenCV Haar cascade from an in-memory XML string, the analogue of loading a cascade from a cv::FileStorage. |
| `func NewCascadeClassifier(filename string) (*CascadeClassifier, error)` | NewCascadeClassifier loads and parses an OpenCV Haar cascade XML file from path, the analogue of cv::cuda::CascadeClassifier::create(filename). |
| `func (c *CascadeClassifier) Convert(gpuObjects *GpuMat) []cv.Rect` | Convert decodes the packed detection `GpuMat` produced by `CascadeClassifier.DetectMultiScale` into a slice of rectangles, the analogue of… |
| `func (c *CascadeClassifier) DetectMultiScale(img *GpuMat, stream *Stream) *GpuMat` | DetectMultiScale scans img and returns a `GpuMat` carrying the detected object rectangles, the analogue of… |
| `func (c *CascadeClassifier) GetClassifierSize() (w, h int)` | GetClassifierSize returns the base detection window (width, height) declared by the loaded cascade, the analogue of… |
| `func (c *CascadeClassifier) GetFindLargestObject() bool` | GetFindLargestObject reports whether only the largest detection is returned. |
| `func (c *CascadeClassifier) GetMaxObjectSize() Size` | GetMaxObjectSize returns the configured maximum object size. |
| `func (c *CascadeClassifier) GetMinNeighbors() int` | GetMinNeighbors returns the configured minimum neighbour count. |
| `func (c *CascadeClassifier) GetMinObjectSize() Size` | GetMinObjectSize returns the configured minimum object size. |
| `func (c *CascadeClassifier) GetScaleFactor() float64` | GetScaleFactor returns the configured scale factor. |
| `func (c *CascadeClassifier) SetFindLargestObject(on bool)` | SetFindLargestObject toggles returning only the single largest detection. |
| `func (c *CascadeClassifier) SetMaxObjectSize(s Size)` | SetMaxObjectSize sets the largest object to detect; detections larger in either dimension are discarded. |
| `func (c *CascadeClassifier) SetMinNeighbors(n int)` | SetMinNeighbors sets the minimum number of overlapping detections a grouped result must have; 0 disables grouping and returns every raw hit. |
| `func (c *CascadeClassifier) SetMinObjectSize(s Size)` | SetMinObjectSize sets the smallest object to detect; detections smaller in either dimension are discarded. |
| `func (c *CascadeClassifier) SetScaleFactor(s float64)` | SetScaleFactor sets the ratio between successive window sizes; values <= 1 are treated as the default 1.1. |

</details>

<details>
<summary><code>GpuMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGpuMat(rows, cols, channels int) *GpuMat` | NewGpuMat allocates a zero-filled GpuMat of the given dimensions, mirroring the GpuMat(rows, cols, type) constructor. |
| `func NewGpuMatFromMat(m *cv.Mat) *GpuMat` | NewGpuMatFromMat returns a GpuMat that shares m's backing storage (no copy), the analogue of wrapping an existing device allocation. |
| `func (g *GpuMat) Channels() int` | Channels returns the number of samples per pixel, or 0 for a GpuMat with no image data. |
| `func (g *GpuMat) Clone() *GpuMat` | Clone returns a deep copy of the GpuMat, including any carried detection rectangles, with its own backing storage. |
| `func (g *GpuMat) Download() *cv.Mat` | Download returns a host `cv.Mat` holding an independent copy of the GpuMat's pixels, the analogue of cv::cuda::GpuMat::download. |
| `func (g *GpuMat) Empty() bool` | Empty reports whether the GpuMat holds no image samples. |
| `func (g *GpuMat) Mat() *cv.Mat` | Mat returns the GpuMat's underlying host `cv.Mat` without copying. |
| `func (g *GpuMat) Size() (rows, cols int)` | Size returns the image dimensions as (rows, cols). |
| `func (g *GpuMat) Upload(m *cv.Mat)` | Upload copies host data from m into the GpuMat, the analogue of cv::cuda::GpuMat::upload. |

</details>

<details>
<summary><code>HOG</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDefaultHOG() *HOG` | NewDefaultHOG returns a HOG configured with the canonical Dalal–Triggs geometry (64×128 window, 16×16 blocks stepping by 8×8, 8×8 cells, 9… |
| `func NewHOG(winSize, blockSize, blockStride, cellSize Size, nbins int) *HOG` | NewHOG creates a HOG detector with the given geometry, mirroring cv::cuda::HOG::create(win_size, block_size, block_stride, cell_size, nbins). |
| `func (h *HOG) Compute(img *GpuMat, stream *Stream) []float64` | Compute returns the HOG descriptor for a single window taken from the top-left corner of img, the analogue of cv::cuda::HOG::compute. |
| `func (h *HOG) Detect(img *GpuMat, stream *Stream) (locations []cv.Point, confidences []float64)` | Detect runs the SVM detector at a single scale and returns the top-left corner of every window whose score meets the hit threshold, together with the… |
| `func (h *HOG) DetectMultiScale(img *GpuMat, stream *Stream) (locations []cv.Rect, confidences []float64)` | DetectMultiScale slides the detector over a downscaling image pyramid and returns the detected object rectangles together with their scores, the… |
| `func (h *HOG) GetBlockHistogramSize() int` | GetBlockHistogramSize returns the number of values in a single block's histogram (cells-per-block × nbins), the analogue of… |
| `func (h *HOG) GetDefaultPeopleDetector() []float64` | GetDefaultPeopleDetector returns an approximate linear-SVM weight vector for upright-person detection sized to this detector's geometry, the analogue… |
| `func (h *HOG) GetDescriptorSize() int` | GetDescriptorSize returns the length of the descriptor vector produced for a single window, the analogue of cv::cuda::HOG::getDescriptorSize. |
| `func (h *HOG) GetGammaCorrection() bool` | GetGammaCorrection reports whether gamma correction is enabled. |
| `func (h *HOG) GetGroupThreshold() int` | GetGroupThreshold returns the grouping threshold. |
| `func (h *HOG) GetHitThreshold() float64` | GetHitThreshold returns the current hit threshold. |
| `func (h *HOG) GetL2HysThreshold() float64` | GetL2HysThreshold returns the L2-Hys clipping threshold. |
| `func (h *HOG) GetNumLevels() int` | GetNumLevels returns the configured maximum number of pyramid levels. |
| `func (h *HOG) GetSVMDetector() []float64` | GetSVMDetector returns a copy of the currently set SVM coefficients, or nil if none has been set. |
| `func (h *HOG) GetScaleFactor() float64` | GetScaleFactor returns the pyramid scale factor. |
| `func (h *HOG) GetWinSigma() float64` | GetWinSigma returns the configured window sigma. |
| `func (h *HOG) GetWinStride() Size` | GetWinStride returns the configured window stride. |
| `func (h *HOG) SetGammaCorrection(on bool)` | SetGammaCorrection toggles input gamma correction (stored for compatibility). |
| `func (h *HOG) SetGroupThreshold(n int)` | SetGroupThreshold sets the minimum cluster size kept when grouping overlapping detections; 0 disables grouping. |
| `func (h *HOG) SetHitThreshold(t float64)` | SetHitThreshold sets the minimum SVM score a window must reach to be reported. |
| `func (h *HOG) SetL2HysThreshold(t float64)` | SetL2HysThreshold sets the L2-Hys clipping threshold (stored for compatibility). |
| `func (h *HOG) SetNumLevels(n int)` | SetNumLevels sets the maximum number of pyramid levels (advisory). |
| `func (h *HOG) SetSVMDetector(detector []float64)` | SetSVMDetector sets the linear-SVM coefficients used by `HOG.Detect` and `HOG.DetectMultiScale`, the analogue of cv::cuda::HOG::setSVMDetector. |
| `func (h *HOG) SetScaleFactor(s float64)` | SetScaleFactor sets the pyramid ratio between successive levels; it must be greater than 1 and panics otherwise. |
| `func (h *HOG) SetWinSigma(s float64)` | SetWinSigma sets the Gaussian window weighting sigma (stored for compatibility; a negative value means "auto"). |
| `func (h *HOG) SetWinStride(s Size)` | SetWinStride sets the window step (advisory; detection steps by BlockStride). |
| `func (h *HOG) WinSize() Size` | WinSize returns the detection window in pixels. |

</details>

<details>
<summary><code>Stream</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStream() *Stream` | NewStream returns a new no-op Stream. |
| `func (s *Stream) QueryIfComplete() bool` | QueryIfComplete always reports true, since work is never deferred. |
| `func (s *Stream) WaitForCompletion()` | WaitForCompletion returns immediately: all work in this package is executed synchronously, so there is never anything outstanding to wait for. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/cudaobjdetect).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
