# objdetect — provides classic, learning-free object-detection tooling for the parent

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/objdetect.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/objdetect)

Package objdetect provides classic, learning-free object-detection tooling for
the parent `github.com/malcolmston/opencv` package: a Histogram of Oriented
Gradients (HOG) descriptor with a linear-SVM sliding-window detector, a
Viola–Jones Haar cascade classifier that reads OpenCV's XML cascade files,
and a QR-code finder-pattern locator. It mirrors a useful subset of OpenCV's
cv::objdetect module using only the Go standard library (including
encoding/xml) plus the root cv package.

Everything here operates on the root `cv.Mat` type and returns results in the
root's `cv.Rect` and `cv.Point` geometry types. Images may be single-channel
(grayscale) or three-channel; multi-channel input is reduced to luma (0.299R +
0.587G + 0.114B, matching the root package's RGB convention) before analysis.
No cgo, no third-party dependencies.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/objdetect"
```

## Usage

This is the package's own `ExampleCascadeClassifier_LoadFromString`, so it compiles and its output is
asserted on every `go test ./objdetect/`.

```go
const xml = `<?xml version="1.0"?>
<opencv_storage>
<cascade>
  <featureType>HAAR</featureType>
  <height>8</height>
  <width>8</width>
  <stages><_>
    <stageThreshold>0.</stageThreshold>
    <weakClassifiers><_>
      <internalNodes>0 -1 0 0.5</internalNodes>
      <leafValues>-1. 1.</leafValues>
    </_></weakClassifiers>
  </_></stages>
  <features><_>
    <rects><_>0 0 8 4 1.</_><_>0 4 8 4 -1.</_></rects>
    <tilted>0</tilted>
  </_></features>
</cascade>
</opencv_storage>`
	var clf objdetect.CascadeClassifier
	if err := clf.LoadFromString(xml); err != nil {
		fmt.Println("error:", err)
		return
	}
	w, h := clf.WindowSize()
	fmt.Printf("%dx%d\n", w, h)
```

```
8x8
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ComputeLBP(img *cv.Mat) *cv.Mat` | ComputeLBP returns the Local Binary Pattern code image of img (reduced to luma). |
| `func GroupRectangles(rects []cv.Rect, minNeighbors int, eps float64) []cv.Rect` | GroupRectangles clusters similar rectangles and returns one averaged rectangle per cluster whose membership is at least minNeighbors, following the… |
| `func GroupRectanglesWeights(rects []cv.Rect, weights []float64, minNeighbors int, eps float64) ([]cv.Rect, []float64)` | GroupRectanglesWeights clusters rectangles like `GroupRectangles` but also carries a per-rectangle score. |
| `func NMSBoxes(boxes []cv.Rect, scores []float64, scoreThreshold, nmsThreshold float64) []int` | NMSBoxes performs greedy non-maximum suppression on a set of scored detection boxes, mirroring OpenCV's cv::dnn::NMSBoxes. |
| `func RectIoU(a, b cv.Rect) float64` | RectIoU returns the intersection-over-union overlap of two rectangles, a value in [0,1]. |
| `func SoftNMSBoxes(boxes []cv.Rect, scores []float64, scoreThreshold, sigma float64) (indices []int, keptScores []float64)` | SoftNMSBoxes performs Gaussian Soft-NMS (Bodla et al., 2017). |

### Types

| Type | What it is |
| --- | --- |
| `CascadeClassifier` | CascadeClassifier is a Viola–Jones Haar-feature cascade loaded from an OpenCV XML file. |
| `DetectionBasedTracker` | DetectionBasedTracker follows detected objects across a sequence of frames, giving each a stable identity, in the spirit of OpenCV's… |
| `Detector` | Detector is anything that can return object rectangles for an image. |
| `HOGDescriptor` | HOGDescriptor computes a Histogram of Oriented Gradients feature vector and runs a linear-SVM sliding-window detector, following Dalal & Triggs… |
| `IntegralImage` | IntegralImage is a summed-area table of an image's luma and squared luma. |
| `LBPCascadeClassifier` | LBPCascadeClassifier is a Local Binary Pattern cascade loaded from an OpenCV XML file (featureType LBP), the LBP analogue of the Haar… |
| `QRCodeDetector` | QRCodeDetector locates the finder patterns of a QR code. |
| `Size` | Size is a width/height pair in pixels, the analogue of OpenCV's cv::Size used to configure the HOG geometry. |
| `SoftCascade` | SoftCascade is a soft-cascade detector (Bourdev & Brandt, 2005): a single additive ensemble of `SoftStump`s evaluated with a running score that is… |
| `SoftFeature` | SoftFeature is one Haar-like feature: a small set of `WeightedRect`s whose weighted area sums are combined into a single scalar feature value. |
| `SoftStump` | SoftStump is a depth-1 decision stump over a `SoftFeature`. |
| `TrackedObject` | TrackedObject is one object followed across frames by a `DetectionBasedTracker`. |
| `WeightedRect` | WeightedRect is a rectangle with a signed weight, the building block of a Haar feature: the feature value is the weighted sum of the image intensity… |

<details>
<summary><code>CascadeClassifier</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (c *CascadeClassifier) DetectMultiScale(img *cv.Mat) []cv.Rect` | DetectMultiScale scans img at increasing window sizes and returns the rectangles where the cascade fires. |
| `func (c *CascadeClassifier) Load(path string) error` | Load reads and parses an OpenCV Haar cascade XML file from path. |
| `func (c *CascadeClassifier) LoadFromString(xmlData string) error` | LoadFromString parses an OpenCV Haar cascade from an in-memory XML string. |
| `func (c *CascadeClassifier) Loaded() bool` | Loaded reports whether a cascade has been successfully loaded. |
| `func (c *CascadeClassifier) ToSoftCascade() *SoftCascade` | ToSoftCascade converts a loaded Haar `CascadeClassifier` into an equivalent `SoftCascade`. |
| `func (c *CascadeClassifier) WindowSize() (w, h int)` | WindowSize returns the base detection window (width, height) declared by the loaded cascade. |

</details>

<details>
<summary><code>DetectionBasedTracker</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (t *DetectionBasedTracker) Objects() []TrackedObject` | Objects returns a snapshot of every live track, ordered by ID. |
| `func (t *DetectionBasedTracker) Process(img *cv.Mat)` | Process runs the detector on img and updates the tracker's state for one frame. |
| `func (t *DetectionBasedTracker) Reset()` | Reset clears all tracks and identity counters, returning the tracker to its initial state while keeping its configuration. |
| `func (t *DetectionBasedTracker) Visible() []TrackedObject` | Visible returns only the tracks matched in the most recent frame (TimeSinceUpdate == 0), ordered by ID. |

</details>

<details>
<summary><code>HOGDescriptor</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewHOGDescriptor() *HOGDescriptor` | NewHOGDescriptor returns the canonical OpenCV/Dalal–Triggs configuration: a 64×128 window, 16×16 blocks stepping by 8×8, 8×8 cells and 9… |
| `func (h *HOGDescriptor) Compute(img *cv.Mat) []float64` | Compute returns the HOG descriptor for a single window taken from the top-left corner of img. |
| `func (h *HOGDescriptor) ComputeGradient(img *cv.Mat) (mag, angle []float64, width, height int)` | ComputeGradient returns the per-pixel gradient magnitude and unsigned orientation of img (reduced to luma), the analogue of OpenCV's… |
| `func (h *HOGDescriptor) DefaultPeopleDetector() []float64` | DefaultPeopleDetector returns an approximate linear-SVM weight vector for upright-person detection, the standard-library analogue of OpenCV's… |
| `func (h *HOGDescriptor) DescriptorSize() int` | DescriptorSize returns the length of the vector produced by `HOGDescriptor.Compute` for this geometry. |
| `func (h *HOGDescriptor) DetectMultiScale(img *cv.Mat, svmWeights []float64, hitThreshold, scale float64) []cv.Rect` | DetectMultiScale slides the detection window over a downscaling image pyramid and returns the windows whose linear-SVM score meets hitThreshold,… |
| `func (h *HOGDescriptor) DetectMultiScaleWeights(img *cv.Mat, svmWeights []float64, hitThreshold, scale float64) ([]cv.Rect, []float64)` | DetectMultiScaleWeights is `HOGDescriptor.DetectMultiScale` that also reports the linear-SVM score of every returned window. |

</details>

<details>
<summary><code>IntegralImage</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewIntegralImage(img *cv.Mat) *IntegralImage` | NewIntegralImage builds the summed-area tables for img (reduced to luma). |
| `func (ii *IntegralImage) SqSum(x, y, w, h int) float64` | SqSum returns the total squared luma over the rectangle [x, x+w) × [y, y+h). |
| `func (ii *IntegralImage) Sum(x, y, w, h int) float64` | Sum returns the total luma over the rectangle [x, x+w) × [y, y+h). |

</details>

<details>
<summary><code>LBPCascadeClassifier</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (c *LBPCascadeClassifier) DetectMultiScale(img *cv.Mat) []cv.Rect` | DetectMultiScale scans img at growing window sizes and returns the rectangles where the LBP cascade fires. |
| `func (c *LBPCascadeClassifier) Load(path string) error` | Load reads and parses an OpenCV LBP cascade XML file from path. |
| `func (c *LBPCascadeClassifier) LoadFromString(xmlData string) error` | LoadFromString parses an OpenCV LBP cascade from an in-memory XML string. |
| `func (c *LBPCascadeClassifier) Loaded() bool` | Loaded reports whether a cascade has been successfully loaded. |
| `func (c *LBPCascadeClassifier) WindowSize() (w, h int)` | WindowSize returns the base detection window (width, height) declared by the loaded cascade. |

</details>

<details>
<summary><code>QRCodeDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewQRCodeDetector() *QRCodeDetector` | NewQRCodeDetector returns a detector with the default binarisation threshold. |
| `func (d *QRCodeDetector) Detect(img *cv.Mat) (corners []cv.Point, found bool)` | Detect scans img for QR finder patterns and returns the centres of the (up to three) strongest patterns. |
| `func (d *QRCodeDetector) DetectFinderPatterns(img *cv.Mat) []cv.Point` | DetectFinderPatterns scans img and returns the centres of every QR finder pattern it can locate, ordered strongest-first (by the number of supporting… |
| `func (d *QRCodeDetector) DetectMulti(img *cv.Mat) (quads [][]cv.Point, found bool)` | DetectMulti locates every QR symbol in img and returns one quadrilateral per symbol, the multi-code analogue of OpenCV's QRCodeDetector::detectMulti. |

</details>

<details>
<summary><code>SoftCascade</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (s *SoftCascade) DetectMultiScale(img *cv.Mat) []cv.Rect` | DetectMultiScale scans img at growing window sizes and returns every window the soft cascade accepts. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/objdetect).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
