# saliency2 — standard-library-only toolkit of visual saliency and attention algorithms built on top

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/saliency2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/saliency2)

Package saliency2 is a standard-library-only toolkit of visual saliency and
attention algorithms built on top of the parent cv package's `cv.Mat` image
type.

Saliency estimation predicts where a human observer's gaze is drawn in an
image: bright, novel, high-contrast or moving regions "pop out" from their
surroundings. This package collects the classic static and dynamic saliency
models, plus an objectness proposal generator and a set of reusable
post-processing operators, all implemented from scratch against the Go
standard library (image, math, math/cmplx). There is no cgo, no third-party
dependency and no GPU requirement; every routine is deterministic and
CPU-only.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/saliency2"
```

## Usage

This is the package's own `ExampleCenterPrior`, so it compiles and its output is
asserted on every `go test ./saliency2/`.

```go
m := saliency2.NewSaliencyMap(9, 9)
	for i := range m.Data {
		m.Data[i] = 1
	}

	p := saliency2.CenterPrior(m, 0.4)
	fmt.Println("centre keeps more weight than the corner:", p.At(4, 4) > p.At(0, 0))
```

```
centre keeps more weight than the corner: true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func FineGrainedSaliency(img *cv.Mat) *cv.Mat` | FineGrainedSaliency is a convenience wrapper that computes the fine-grained saliency map of img with the default detector settings. |
| `func FrequencyTunedSaliency(img *cv.Mat) *cv.Mat` | FrequencyTunedSaliency is a convenience wrapper that computes the frequency-tuned saliency map of img with the default detector settings. |
| `func HeatmapOverlay(img *cv.Mat, m *SaliencyMap, alpha float64) *cv.Mat` | HeatmapOverlay blends a colourised saliency map over a copy of img and returns the three-channel result. |
| `func IttiKochSaliency(img *cv.Mat) *cv.Mat` | IttiKochSaliency is a convenience wrapper that computes the Itti-Koch saliency map of img with the default detector settings. |
| `func SalientMask(m *SaliencyMap) *cv.Mat` | SalientMask returns the adaptive binary segmentation of m using the mean-based rule of Achanta et al.: a pixel is foreground when its value is at… |
| `func SpectralResidualSaliency(img *cv.Mat) *cv.Mat` | SpectralResidualSaliency is a convenience wrapper that computes the spectral-residual saliency map of img with the default detector settings. |

### Types

| Type | What it is |
| --- | --- |
| `Box` | Box is a scored candidate window produced by an `Objectness` detector. |
| `MotionSaliency` | MotionSaliency is implemented by the streaming (video) saliency detectors, mirroring OpenCV's cv::saliency::MotionSaliency. |
| `MotionSaliencyByDifference` | MotionSaliencyByDifference is a streaming motion-saliency detector based on temporal frame differencing. |
| `MotionSaliencyRunningAverage` | MotionSaliencyRunningAverage is a streaming motion-saliency detector that compares each frame against an exponentially decaying background model… |
| `Objectness` | Objectness is implemented by generic object-proposal generators, mirroring OpenCV's cv::saliency::Objectness. |
| `ObjectnessBING` | ObjectnessBING is a lightweight, untrained object-proposal generator in the spirit of Cheng, Zhang, Lin & Torr's BING, "Binarized Normed Gradients… |
| `SaliencyMap` | SaliencyMap is a single-channel grid of float64 saliency values. |
| `StaticSaliency` | StaticSaliency is implemented by the single-image saliency detectors. |
| `StaticSaliencyFineGrained` | StaticSaliencyFineGrained detects saliency with the fine-grained center-surround model of Montabone & Soto, "Human detection using a mobile platform… |
| `StaticSaliencyFrequencyTuned` | StaticSaliencyFrequencyTuned detects saliency with the frequency-tuned model of Achanta, Hemami, Estrada & Susstrunk, "Frequency-tuned Salient Region… |
| `StaticSaliencyIttiKoch` | StaticSaliencyIttiKoch implements a center-surround attention model in the spirit of Itti, Koch & Niebur, "A Model of Saliency-Based Visual Attention… |
| `StaticSaliencySpectralResidual` | StaticSaliencySpectralResidual detects saliency with the spectral-residual method of Hou & Zhang, "Saliency Detection: A Spectral Residual Approach"… |

<details>
<summary><code>Box</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (b Box) Area() int` | Area returns the area of the box in pixels. |
| `func (b Box) Center() (x, y float64)` | Center returns the centre point of the box as floating-point (x, y) coordinates. |
| `func (b Box) IoU(other Box) float64` | IoU returns the intersection-over-union overlap of b and other, a value in [0,1] where 1 means the two windows coincide exactly and 0 means they are… |

</details>

<details>
<summary><code>MotionSaliencyByDifference</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMotionSaliencyByDifference() *MotionSaliencyByDifference` | NewMotionSaliencyByDifference returns a frame-difference detector with a mild sigma-1 smoothing. |
| `func (m *MotionSaliencyByDifference) ComputeSaliency(frame *cv.Mat) *cv.Mat` | ComputeSaliency ingests the next frame and returns its motion saliency map as an 8-bit single-channel `cv.Mat`. |
| `func (m *MotionSaliencyByDifference) ComputeSaliencyMap(frame *cv.Mat) *SaliencyMap` | ComputeSaliencyMap ingests the next frame and returns its motion saliency as a `SaliencyMap`. |
| `func (m *MotionSaliencyByDifference) Reset()` | Reset discards the stored previous frame so the next frame starts a new sequence. |

</details>

<details>
<summary><code>MotionSaliencyRunningAverage</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMotionSaliencyRunningAverage() *MotionSaliencyRunningAverage` | NewMotionSaliencyRunningAverage returns a running-average detector with a 0.05 learning rate and mild sigma-1 smoothing. |
| `func (m *MotionSaliencyRunningAverage) Background() *cv.Mat` | Background returns a snapshot of the current background model as an 8-bit single-channel `cv.Mat`, or nil before any frame has been ingested. |
| `func (m *MotionSaliencyRunningAverage) ComputeSaliency(frame *cv.Mat) *cv.Mat` | ComputeSaliency ingests the next frame and returns its motion saliency map as an 8-bit single-channel `cv.Mat`. |
| `func (m *MotionSaliencyRunningAverage) ComputeSaliencyMap(frame *cv.Mat) *SaliencyMap` | ComputeSaliencyMap ingests the next frame, updates the background model and returns the frame's motion saliency as a `SaliencyMap`. |
| `func (m *MotionSaliencyRunningAverage) Reset()` | Reset discards the background model so the next frame starts a new sequence. |

</details>

<details>
<summary><code>ObjectnessBING</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewObjectnessBING() *ObjectnessBING` | NewObjectnessBING returns a generator with sensible defaults: window sizes at 1/4, 1/2 and 3/4 of the smaller image side, up to 32 proposals, and an… |
| `func (o *ObjectnessBING) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the objectness map of img as an 8-bit single-channel `cv.Mat`, with bright pixels marking likely object-window anchors. |
| `func (o *ObjectnessBING) ObjectnessBoundingBoxes(img *cv.Mat) []Box` | ObjectnessBoundingBoxes returns candidate object windows for img ranked by decreasing objectness score, after non-maximum suppression. |
| `func (o *ObjectnessBING) ObjectnessMap(img *cv.Mat) *SaliencyMap` | ObjectnessMap returns the normed-gradient boundary-energy field used to score windows: for every pixel it holds the mean gradient along the border of… |

</details>

<details>
<summary><code>SaliencyMap</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BoxSmooth(m *SaliencyMap, radius int) *SaliencyMap` | BoxSmooth returns a box-averaged copy of m with the given window radius, computed in linear time with an integral image. |
| `func CenterPrior(m *SaliencyMap, sigmaFrac float64) *SaliencyMap` | CenterPrior multiplies m by an isotropic Gaussian centre prior, damping responses toward the image border. |
| `func CenterSurround(center, surround *SaliencyMap) *SaliencyMap` | CenterSurround returns the center-surround response of a single feature map: the absolute difference between center and an up-sampled copy of… |
| `func CombineMaps(maps ...*SaliencyMap) *SaliencyMap` | CombineMaps averages several saliency maps into one, after normalising each to [0,1] so they contribute equally. |
| `func GammaCorrect(m *SaliencyMap, gamma float64) *SaliencyMap` | GammaCorrect returns a copy of m with each value raised to the power gamma after normalisation to [0,1]. |
| `func GaussianSmooth(m *SaliencyMap, sigma float64) *SaliencyMap` | GaussianSmooth returns a Gaussian-blurred copy of m. |
| `func IttiNormalize(m *SaliencyMap) *SaliencyMap` | IttiNormalize applies Itti, Koch and Niebur's normalisation operator N(.) to m: the map is scaled to [0,1], then multiplied by (1 - mean)^2 where… |
| `func LogScale(m *SaliencyMap) *SaliencyMap` | LogScale returns a copy of m compressed by log(1+v) after shifting so its minimum is zero. |
| `func NewSaliencyMap(rows, cols int) *SaliencyMap` | NewSaliencyMap allocates a zero-filled map of the given size. |
| `func NormalizeRange(m *SaliencyMap, lo, hi float64) *SaliencyMap` | NormalizeRange returns a copy of m linearly rescaled to the closed interval [lo, hi]. |
| `func NormedGradient(img *cv.Mat) *SaliencyMap` | NormedGradient returns the normed-gradient magnitude map of img: the Sobel gradient magnitude of its luminance. |
| `func SaliencyMapFromFloatMat(f *cv.FloatMat) *SaliencyMap` | SaliencyMapFromFloatMat copies the parent package's `cv.FloatMat` into a SaliencyMap of the same size. |
| `func SaliencyMapFromMat(img *cv.Mat) *SaliencyMap` | SaliencyMapFromMat builds a map from the luminance of img, with values in [0,255]. |
| `func (m *SaliencyMap) AddMap(other *SaliencyMap) *SaliencyMap` | AddMap returns the elementwise sum of m and other, which must have the same dimensions. |
| `func (m *SaliencyMap) At(y, x int) float64` | At returns the value at row y, column x. |
| `func (m *SaliencyMap) BoundingBox(t float64) (rect image.Rectangle, ok bool)` | BoundingBox returns the smallest axis-aligned rectangle enclosing every pixel whose value is greater than or equal to t. |
| `func (m *SaliencyMap) CenterOfMass() (x, y float64)` | CenterOfMass returns the value-weighted centroid (x, y) of the map, treating values below zero as zero. |
| `func (m *SaliencyMap) Clone() *SaliencyMap` | Clone returns a deep copy of the map with its own backing storage. |
| `func (m *SaliencyMap) Mean() float64` | Mean returns the arithmetic mean of all values in the map. |
| `func (m *SaliencyMap) MeanThreshold(k float64) (mask *cv.Mat, threshold float64)` | MeanThreshold returns the binary mask produced by thresholding the map at k times its mean value — the adaptive rule of Achanta et al., which uses… |
| `func (m *SaliencyMap) MinMax() (min, max float64)` | MinMax returns the smallest and largest values in the map. |
| `func (m *SaliencyMap) MultiplyMap(other *SaliencyMap) *SaliencyMap` | MultiplyMap returns the elementwise product of m and other, which must have the same dimensions. |
| `func (m *SaliencyMap) Normalize() *SaliencyMap` | Normalize returns a copy of the map linearly rescaled so its minimum maps to 0 and its maximum to 1. |
| `func (m *SaliencyMap) OtsuThreshold() (mask *cv.Mat, threshold float64)` | OtsuThreshold binarises the map using Otsu's method, which chooses the threshold that maximises between-class variance over a 256-bin histogram of… |
| `func (m *SaliencyMap) PercentileThreshold(p float64) (mask *cv.Mat, threshold float64)` | PercentileThreshold returns the binary mask keeping the fraction p (in [0,1]) of the highest-valued pixels, together with the threshold value used. |
| `func (m *SaliencyMap) Scale(factor float64) *SaliencyMap` | Scale returns a copy of the map with every value multiplied by factor. |
| `func (m *SaliencyMap) Set(y, x int, value float64)` | Set stores value at row y, column x. |
| `func (m *SaliencyMap) Size() (rows, cols int)` | Size returns the map dimensions as (rows, cols). |
| `func (m *SaliencyMap) StdDev() float64` | StdDev returns the population standard deviation of the map's values. |
| `func (m *SaliencyMap) Sum() float64` | Sum returns the total of all values in the map. |
| `func (m *SaliencyMap) Threshold(t float64) *cv.Mat` | Threshold returns a binary `cv.Mat` in which pixels whose value is greater than or equal to t are 255 and all others 0. |
| `func (m *SaliencyMap) ToFloatMat() *cv.FloatMat` | ToFloatMat returns a copy of the map as the parent package's `cv.FloatMat`. |
| `func (m *SaliencyMap) ToMat() *cv.Mat` | ToMat returns an 8-bit single-channel `cv.Mat` rendering of the map, min-max normalised so the smallest value becomes 0 and the largest 255. |

</details>

<details>
<summary><code>StaticSaliencyFineGrained</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStaticSaliencyFineGrained() *StaticSaliencyFineGrained` | NewStaticSaliencyFineGrained returns a detector with the default multi-scale radii {1, 2, 4, 8, 16}. |
| `func (s *StaticSaliencyFineGrained) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the fine-grained saliency map of img as an 8-bit single-channel `cv.Mat`. |
| `func (s *StaticSaliencyFineGrained) ComputeSaliencyMap(img *cv.Mat) *SaliencyMap` | ComputeSaliencyMap returns the fine-grained saliency of img as a `SaliencyMap` the same size as img. |

</details>

<details>
<summary><code>StaticSaliencyFrequencyTuned</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStaticSaliencyFrequencyTuned() *StaticSaliencyFrequencyTuned` | NewStaticSaliencyFrequencyTuned returns a detector with a 3x3, sigma-1 smoothing kernel. |
| `func (s *StaticSaliencyFrequencyTuned) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the frequency-tuned saliency map of img as an 8-bit single-channel `cv.Mat`. |
| `func (s *StaticSaliencyFrequencyTuned) ComputeSaliencyMap(img *cv.Mat) *SaliencyMap` | ComputeSaliencyMap returns the frequency-tuned saliency of img as a `SaliencyMap` the same size as img. |

</details>

<details>
<summary><code>StaticSaliencyIttiKoch</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStaticSaliencyIttiKoch() *StaticSaliencyIttiKoch` | NewStaticSaliencyIttiKoch returns a detector with the classical pyramid configuration (up to 9 levels, centres {2,3}, surround deltas {2,3}). |
| `func (s *StaticSaliencyIttiKoch) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the Itti-Koch saliency map of img as an 8-bit single-channel `cv.Mat`. |
| `func (s *StaticSaliencyIttiKoch) ComputeSaliencyMap(img *cv.Mat) *SaliencyMap` | ComputeSaliencyMap returns the Itti-Koch saliency of img as a `SaliencyMap` the same size as img. |

</details>

<details>
<summary><code>StaticSaliencySpectralResidual</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStaticSaliencySpectralResidual() *StaticSaliencySpectralResidual` | NewStaticSaliencySpectralResidual returns a detector configured with the OpenCV default 64x64 working resolution. |
| `func (s *StaticSaliencySpectralResidual) ComputeSaliency(img *cv.Mat) *cv.Mat` | ComputeSaliency returns the spectral-residual saliency map of img as an 8-bit single-channel `cv.Mat`. |
| `func (s *StaticSaliencySpectralResidual) ComputeSaliencyMap(img *cv.Mat) *SaliencyMap` | ComputeSaliencyMap returns the spectral-residual saliency of img as a `SaliencyMap` the same size as img. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/saliency2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
