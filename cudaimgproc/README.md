# cudaimgproc — pure-Go, standard-library-only mirror of OpenCV's cudaimgproc module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/cudaimgproc.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/cudaimgproc)

Package cudaimgproc is a pure-Go, standard-library-only mirror of OpenCV's
cudaimgproc module. It presents the same API shape as the CUDA image
processing routines — a `GpuMat` device-matrix type, no-op `Stream` objects
and the familiar create*/Detect/Compute/Apply/Match algorithm objects — but
every operation runs on the CPU.

There is NO GPU here. Despite the "cuda" name and the GpuMat/Stream
vocabulary, this package contains no cgo, no CUDA kernels and no device
memory. A `GpuMat` is a thin wrapper around a host-resident `cv.Mat`; every
function computes its result on the CPU by delegating to the root package
github.com/malcolmston/opencv (imported here as cv). The device-flavoured
surface exists so that code written against OpenCV's cuda modules — which
upload a Mat, run a sequence of GpuMat operations, then download the result
— can be ported to pure Go with minimal edits. Because the maths is
identical to the root package, results are deterministic and byte-for-byte
comparable with the equivalent cv call; the `Stream` type is a placeholder
that schedules nothing.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/cudaimgproc"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./cudaimgproc/`.

```go
img := cv.NewMat(4, 4, 3)
	for i := range img.Data {
		img.Data[i] = uint8(i * 5)
	}

	var d cudaimgproc.GpuMat
	d.Upload(img)
	gray := cudaimgproc.CvtColor(d, cv.ColorRGB2Gray)
	out := cudaimgproc.EqualizeHist(gray).Download()

	fmt.Printf("%dx%dx%d\n", out.Rows, out.Cols, out.Channels)
```

```
4x4x1
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func CalcHist(src GpuMat, streams ...Stream) []int` | CalcHist computes a 256-bin intensity histogram of a single-channel GpuMat, mirroring cuda::calcHist. |
| `func HistEven(src GpuMat, histSize, lowerLevel, upperLevel int, streams ...Stream) []int` | HistEven computes a histogram with histSize equally spaced bins spanning the half-open range [lowerLevel, upperLevel), mirroring cuda::histEven. |
| `func HistRange(src GpuMat, levels []int, streams ...Stream) []int` | HistRange computes a histogram whose bins are delimited by the ascending boundaries in levels, mirroring cuda::histRange. |

### Types

| Type | What it is |
| --- | --- |
| `AlphaCompType` | AlphaCompType selects the Porter–Duff compositing operator used by `AlphaComp`, matching OpenCV's ALPHA_* constants. |
| `BayerCode` | BayerCode selects the Bayer colour-filter-array layout for `DemosaicBayer` and `CvtColorBayer`. |
| `CLAHE` | CLAHE is a CPU-backed Contrast-Limited Adaptive Histogram Equalisation algorithm object, mirroring cv::cuda::CLAHE. |
| `CannyEdgeDetector` | CannyEdgeDetector is a CPU-backed Canny edge detector, mirroring cv::cuda::CannyEdgeDetector. |
| `CornernessCriteria` | CornernessCriteria is the common interface of the CPU-backed corner response operators `HarrisCorner` and `MinEigenValCorner`, mirroring… |
| `GpuMat` | GpuMat is a CPU-backed analogue of OpenCV's cuda::GpuMat. |
| `HarrisCorner` | HarrisCorner is a CPU-backed Harris corner-response operator, mirroring the object returned by cv::cuda::createHarrisCorner. |
| `HoughCirclesDetector` | HoughCirclesDetector is a CPU-backed Hough-gradient circle detector, mirroring cv::cuda::HoughCirclesDetector. |
| `HoughLinesDetector` | HoughLinesDetector is a CPU-backed standard Hough line detector, mirroring cv::cuda::HoughLinesDetector. |
| `HoughSegmentDetector` | HoughSegmentDetector is a CPU-backed probabilistic Hough line-segment detector, mirroring cv::cuda::HoughSegmentDetector. |
| `MinEigenValCorner` | MinEigenValCorner is a CPU-backed Shi–Tomasi corner-response operator (the minimum eigenvalue of the structure tensor), mirroring the object… |
| `Stream` | Stream is a CPU no-op stand-in for OpenCV's cuda::Stream. |
| `TemplateMatching` | TemplateMatching is a CPU-backed template matcher, mirroring cv::cuda::TemplateMatching. |

<details>
<summary><code>CLAHE</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateCLAHE(clipLimit float64, tileGridSize int) *CLAHE` | CreateCLAHE returns a `CLAHE` with the given clip limit and a square tileGridSize×tileGridSize tiling, mirroring cuda::createCLAHE. |
| `func (c *CLAHE) Apply(src GpuMat, streams ...Stream) GpuMat` | Apply runs CLAHE on a single-channel GpuMat and returns the enhanced image. |
| `func (c *CLAHE) GetClipLimit() float64` | GetClipLimit returns the current clip limit. |
| `func (c *CLAHE) GetTilesGridSize() int` | GetTilesGridSize returns the current tile-grid size. |
| `func (c *CLAHE) SetClipLimit(clipLimit float64)` | SetClipLimit updates the contrast clip limit used by subsequent Apply calls. |
| `func (c *CLAHE) SetTilesGridSize(size int)` | SetTilesGridSize updates the square tile-grid size used by subsequent Apply calls. |

</details>

<details>
<summary><code>CannyEdgeDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateCannyEdgeDetector(lowThresh, highThresh float64) *CannyEdgeDetector` | CreateCannyEdgeDetector returns a `CannyEdgeDetector` with the given hysteresis thresholds, mirroring cuda::createCannyEdgeDetector. |
| `func (d *CannyEdgeDetector) Detect(src GpuMat, streams ...Stream) GpuMat` | Detect runs the Canny pipeline on a single-channel GpuMat and returns a binary edge map (edges are 255, background 0). |
| `func (d *CannyEdgeDetector) SetHighThreshold(v float64)` | SetHighThreshold updates the high hysteresis threshold. |
| `func (d *CannyEdgeDetector) SetLowThreshold(v float64)` | SetLowThreshold updates the low hysteresis threshold. |

</details>

<details>
<summary><code>GpuMat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func AlphaComp(img1, img2 GpuMat, alphaOp AlphaCompType, streams ...Stream) GpuMat` | AlphaComp composites two 4-channel (RGBA) images using the selected Porter–Duff operator, mirroring cuda::alphaComp. |
| `func BilateralFilter(src GpuMat, d int, sigmaColor, sigmaSpace float64, streams ...Stream) GpuMat` | BilateralFilter applies an edge-preserving bilateral filter, mirroring cuda::bilateralFilter. |
| `func Blend(img1, img2 GpuMat, weight1, weight2 float64, streams ...Stream) GpuMat` | Blend returns the constant-weight linear combination weight1·img1 + weight2·img2, saturating to [0,255], mirroring the scalar form of… |
| `func BlendLinear(img1, img2, weights1, weights2 GpuMat, streams ...Stream) GpuMat` | BlendLinear blends two images using per-pixel weight maps, mirroring cuda::blendLinear: for each pixel the output is (w1·img1 + w2·img2)/(w1 + w2),… |
| `func CalcBackProject(src GpuMat, hist []int, streams ...Stream) GpuMat` | CalcBackProject back-projects a 256-bin histogram onto a single-channel GpuMat, mirroring cuda::calcBackProject: each output pixel is the histogram… |
| `func CvtColor(src GpuMat, code cv.ColorConversionCode, streams ...Stream) GpuMat` | CvtColor converts src between colour spaces on the CPU and returns a new GpuMat, mirroring cuda::cvtColor. |
| `func CvtColorBayer(src GpuMat, code BayerCode, streams ...Stream) GpuMat` | CvtColorBayer is an alias for `DemosaicBayer` kept for parity with call sites that use the cvtColor spelling for Bayer conversions. |
| `func DemosaicBayer(src GpuMat, code BayerCode, streams ...Stream) GpuMat` | DemosaicBayer reconstructs a 3-channel RGB image from a single-channel Bayer mosaic using bilinear interpolation, mirroring cuda::demosaicing for the… |
| `func EqualizeHist(src GpuMat, streams ...Stream) GpuMat` | EqualizeHist performs global histogram equalisation on a single-channel GpuMat, mirroring cuda::equalizeHist, and returns the equalised image. |
| `func GammaCorrection(src GpuMat, forward bool, streams ...Stream) GpuMat` | GammaCorrection applies sRGB gamma (de)correction, matching cuda::gammaCorrection. |
| `func MeanShiftFiltering(src GpuMat, sp int, sr float64, maxIter int, eps float64, streams ...Stream) GpuMat` | MeanShiftFiltering performs mean-shift filtering on a colour image, mirroring cuda::meanShiftFiltering. |
| `func MeanShiftProc(src GpuMat, sp int, sr float64, maxIter int, eps float64, streams ...Stream) (dstr, dstsp GpuMat)` | MeanShiftProc performs mean-shift filtering and additionally returns the per-pixel converged spatial coordinates, mirroring cuda::meanShiftProc. |
| `func MeanShiftSegmentation(src GpuMat, sp int, sr float64, minsize, maxIter int, eps float64, streams ...Stream) GpuMat` | MeanShiftSegmentation segments a colour image with the mean-shift procedure, mirroring cuda::meanShiftSegmentation. |
| `func NewGpuMat() GpuMat` | NewGpuMat returns an empty `GpuMat`. |
| `func NewGpuMatWithSize(rows, cols, channels int) GpuMat` | NewGpuMatWithSize returns a zero-filled `GpuMat` of the given dimensions, mirroring the sized cuda::GpuMat constructor. |
| `func SwapChannels(src GpuMat, dstOrder []int, streams ...Stream) GpuMat` | SwapChannels reorders the channels of src in place-like fashion, returning a new GpuMat whose channel c is taken from source channel dstOrder`c`. |
| `func (g GpuMat) Channels() int` | Channels returns the number of channels of the wrapped image, or 0 when empty. |
| `func (g GpuMat) Clone() GpuMat` | Clone returns a deep copy of the GpuMat with its own backing storage. |
| `func (g GpuMat) Download(streams ...Stream) *cv.Mat` | Download copies this GpuMat back to a fresh host `cv.Mat`, mirroring the device-to-host transfer. |
| `func (g GpuMat) Empty() bool` | Empty reports whether the GpuMat holds no image. |
| `func (g *GpuMat) Release()` | Release frees the GpuMat's storage, leaving it empty. |
| `func (g GpuMat) Size() (rows, cols int)` | Size returns the matrix dimensions as (rows, cols). |
| `func (g *GpuMat) Upload(src *cv.Mat, streams ...Stream)` | Upload copies a host `cv.Mat` into this GpuMat, replacing any previous contents. |

</details>

<details>
<summary><code>HarrisCorner</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateHarrisCorner(blockSize, ksize int, k float64) *HarrisCorner` | CreateHarrisCorner returns a `HarrisCorner` operator, mirroring cuda::createHarrisCorner. |
| `func (h *HarrisCorner) Compute(src GpuMat, streams ...Stream) *cv.FloatMat` | Compute returns the Harris response R = det(M) - k·trace(M)² for each pixel of a single-channel GpuMat. |

</details>

<details>
<summary><code>HoughCirclesDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateHoughCirclesDetector(minDist, cannyThreshold, votesThreshold float64, minRadius, maxRadius int) *HoughCirclesDetector` | CreateHoughCirclesDetector returns a `HoughCirclesDetector`, mirroring cuda::createHoughCirclesDetector. |
| `func (d *HoughCirclesDetector) Detect(src GpuMat, streams ...Stream) []cv.HoughCircle` | Detect finds circles in a single-channel image. |

</details>

<details>
<summary><code>HoughLinesDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateHoughLinesDetector(rho, theta float64, threshold int) *HoughLinesDetector` | CreateHoughLinesDetector returns a `HoughLinesDetector` with the given accumulator resolution and vote threshold, mirroring… |
| `func (d *HoughLinesDetector) Detect(src GpuMat, streams ...Stream) []cv.PolarLine` | Detect finds lines in a binary edge image and returns them in Hesse normal form (rho, theta), sorted by descending vote count. |

</details>

<details>
<summary><code>HoughSegmentDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateHoughSegmentDetector(rho, theta float64, threshold, minLineLength, maxLineGap int) *HoughSegmentDetector` | CreateHoughSegmentDetector returns a `HoughSegmentDetector` with the given accumulator resolution, vote threshold and segment constraints, mirroring… |
| `func (d *HoughSegmentDetector) Detect(src GpuMat, streams ...Stream) []cv.LineSegment` | Detect finds line segments in a binary edge image. |

</details>

<details>
<summary><code>MinEigenValCorner</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateMinEigenValCorner(blockSize, ksize int) *MinEigenValCorner` | CreateMinEigenValCorner returns a `MinEigenValCorner` operator, mirroring cuda::createMinEigenValCorner. |
| `func (mvc *MinEigenValCorner) Compute(src GpuMat, streams ...Stream) *cv.FloatMat` | Compute returns the smaller eigenvalue of the windowed structure tensor for each pixel of a single-channel GpuMat — the Shi–Tomasi corner measure. |

</details>

<details>
<summary><code>Stream</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStream() Stream` | NewStream returns a ready, non-null `Stream`. |
| `func (s Stream) Empty() bool` | Empty reports whether this is the null stream. |
| `func (s Stream) QueryIfComplete() bool` | QueryIfComplete always reports true, since there is never pending work. |
| `func (s Stream) WaitForCompletion()` | WaitForCompletion returns immediately: CPU work has already finished by the time any function that accepted this stream returned. |

</details>

<details>
<summary><code>TemplateMatching</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CreateTemplateMatching(mode cv.TemplateMatchMode) *TemplateMatching` | CreateTemplateMatching returns a `TemplateMatching` configured with the given similarity mode, mirroring cuda::createTemplateMatching. |
| `func (t *TemplateMatching) Match(src, templ GpuMat, streams ...Stream) *cv.FloatMat` | Match slides templ over src and returns a `cv.FloatMat` of similarity scores of shape (src.Rows-templ.Rows+1)×(src.Cols-templ.Cols+1); use… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/cudaimgproc).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
