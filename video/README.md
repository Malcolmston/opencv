# video — implements motion-analysis and tracking primitives on top of the root cv package

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/video.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/video)

Package video implements motion-analysis and tracking primitives on top of the
root cv package, mirroring a useful subset of OpenCV's video module.

The package is written entirely against the Go standard library (only math)
and the root module github.com/malcolmston/opencv (imported as cv). It does
not depend on any sibling subpackage: the few small helpers it needs
(grayscale conversion, bilinear sampling, gradient computation and dense
linear algebra) are reimplemented locally. It reuses cv.PyrDown to build
Gaussian pyramids and cv.SobelFloat to obtain spatial image gradients.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/video"
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func BuildOpticalFlowPyramid(img *cv.Mat, maxLevel int) []*cv.Mat` | BuildOpticalFlowPyramid builds a Gaussian image pyramid for optical-flow tracking. |
| `func CalcOpticalFlowPyrLK(prev, next *cv.Mat, prevPts []cv.Point, winSize, maxLevel int) (nextPts []cv.Point, status []bool, errs []float64)` | CalcOpticalFlowPyrLK computes sparse optical flow with the pyramidal Lucas-Kanade method. |
| `func CamShift(prob *cv.FloatMat, window cv.Rect, crit TermCriteria) (box cv.RotatedRect, result cv.Rect)` | CamShift runs the Continuously Adaptive Mean Shift tracker on the weight image prob, mirroring cv::CamShift. |
| `func FindTransformECC(templateImage, inputImage *cv.Mat, initWarp [][]float64, motionType MotionType, crit TermCriteria) (cc float64, warp [][]float64)` | FindTransformECC estimates the geometric transform that best aligns inputImage to templateImage by maximising the Enhanced Correlation Coefficient… |
| `func MeanShift(prob *cv.FloatMat, window cv.Rect, crit TermCriteria) (iterations int, result cv.Rect)` | MeanShift finds a local mode of the weight image prob (typically a colour back-projection) by iteratively recentring the search window on the… |

### Types

| Type | What it is |
| --- | --- |
| `BackgroundSubtractorKNN` | BackgroundSubtractorKNN classifies pixels with a K-nearest-neighbours model, mirroring cv::BackgroundSubtractorKNN. |
| `BackgroundSubtractorMOG2` | BackgroundSubtractorMOG2 models each pixel as an adaptive mixture of Gaussians and classifies incoming pixels as foreground or background, mirroring… |
| `DISOpticalFlow` | DISOpticalFlow computes dense optical flow with the Dense Inverse Search algorithm (Kroeger et al., 2016), mirroring cv::DISOpticalFlow. |
| `FlowField` | FlowField is a dense, two-channel float64 motion field: for every pixel it stores a horizontal (X) and vertical (Y) displacement from the previous… |
| `KalmanFilter` | KalmanFilter is a classic linear Kalman filter, mirroring cv::KalmanFilter. |
| `MotionType` | MotionType selects the parametric motion model estimated by `FindTransformECC`. |
| `PointF` | PointF is a floating-point image coordinate (X is the column, Y is the row). |
| `SimilarityTransform` | SimilarityTransform is a 2-D partial-affine (similarity) transform: a uniform Scale, a rotation Angle (radians, counter-clockwise in image… |
| `TermCriteria` | TermCriteria describes when an iterative algorithm (mean-shift, ECC, the DIS inverse search) should stop. |
| `TrackerFeaturePyrLK` | TrackerFeaturePyrLK is a stateful sparse point tracker built on `CalcOpticalFlowPyrLK`. |
| `VideoStabilizer` | VideoStabilizer is a causal digital video stabilizer. |

<details>
<summary><code>BackgroundSubtractorKNN</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBackgroundSubtractorKNN() *BackgroundSubtractorKNN` | NewBackgroundSubtractorKNN returns a subtractor with OpenCV-like defaults (history 500, squared-distance threshold 400, requiring 3 near samples). |
| `func (k *BackgroundSubtractorKNN) Apply(frame *cv.Mat) *cv.Mat` | Apply feeds the next frame to the model and returns a single-channel foreground mask (255 = foreground, 0 = background). |

</details>

<details>
<summary><code>BackgroundSubtractorMOG2</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBackgroundSubtractorMOG2() *BackgroundSubtractorMOG2` | NewBackgroundSubtractorMOG2 returns a subtractor with OpenCV-like defaults (500-frame history, variance threshold 16). |
| `func (b *BackgroundSubtractorMOG2) Apply(frame *cv.Mat) *cv.Mat` | Apply feeds the next frame to the model and returns a single-channel foreground mask (255 = foreground, 0 = background) the same size as the frame. |

</details>

<details>
<summary><code>DISOpticalFlow</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDISOpticalFlow(preset int) *DISOpticalFlow` | NewDISOpticalFlow returns a DISOpticalFlow configured from one of the DISPreset* constants. |
| `func (d *DISOpticalFlow) Calc(prev, next *cv.Mat) *FlowField` | Calc computes the dense optical flow from prev to next and returns it as a `FlowField` with the same dimensions as the inputs. |

</details>

<details>
<summary><code>FlowField</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CalcOpticalFlowFarneback(prev, next *cv.Mat, winSize, searchRadius int) *FlowField` | CalcOpticalFlowFarneback computes a dense optical-flow field from prev to next. |
| `func CalcOpticalFlowFarnebackSubpixel(prev, next *cv.Mat, winSize, searchRadius int) *FlowField` | CalcOpticalFlowFarnebackSubpixel refines `CalcOpticalFlowFarneback` to sub-pixel accuracy. |
| `func NewFlowField(rows, cols int) *FlowField` | NewFlowField allocates a zero-filled FlowField. |
| `func (f *FlowField) At(y, x int) (dx, dy float64)` | At returns the (dx, dy) displacement stored at row y, column x. |
| `func (f *FlowField) MeanFlow(border int) (dx, dy float64)` | MeanFlow returns the average (dx, dy) displacement over the interior of the field, excluding a border of the given width where block matching is… |

</details>

<details>
<summary><code>KalmanFilter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewKalmanFilter(stateDim, measureDim int) *KalmanFilter` | NewKalmanFilter creates a filter for an n-dimensional state and an m-dimensional measurement. |
| `func (kf *KalmanFilter) Correct(z []float64) []float64` | Correct fuses a measurement z (length MeasureDim) and returns the corrected state x. |
| `func (kf *KalmanFilter) Predict() []float64` | Predict advances the filter one time step using the transition model and returns the predicted state x'. |
| `func (kf *KalmanFilter) PredictControl(u []float64) []float64` | PredictControl advances the filter one time step including an external control input u (length ControlDim) applied through the control model B. It… |
| `func (kf *KalmanFilter) SetControlMatrix(b [][]float64)` | SetControlMatrix installs the n x c control model B used by `KalmanFilter.PredictControl` and records ControlDim = c. |
| `func (kf *KalmanFilter) SetMeasurementMatrix(h [][]float64)` | SetMeasurementMatrix installs the m x n measurement model H. It panics unless h is exactly MeasureDim x StateDim. |
| `func (kf *KalmanFilter) SetMeasurementNoiseCov(r [][]float64)` | SetMeasurementNoiseCov installs the m x m measurement-noise covariance R. It panics unless r is exactly MeasureDim x MeasureDim. |
| `func (kf *KalmanFilter) SetProcessNoiseCov(q [][]float64)` | SetProcessNoiseCov installs the n x n process-noise covariance Q. It panics unless q is exactly StateDim x StateDim. |
| `func (kf *KalmanFilter) SetState(x []float64)` | SetState overwrites both StatePre and StatePost with the given state vector, providing an initial estimate. |
| `func (kf *KalmanFilter) SetTransitionMatrix(a [][]float64)` | SetTransitionMatrix installs the n x n state-transition model A. It panics unless a is exactly StateDim x StateDim. |

</details>

<details>
<summary><code>PointF</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CalcOpticalFlowPyrLKF(prev, next *cv.Mat, prevPts []PointF, winSize, maxLevel int) (nextPts []PointF, status []bool, errs []float64)` | CalcOpticalFlowPyrLKF is the sub-pixel variant of `CalcOpticalFlowPyrLK`. |

</details>

<details>
<summary><code>SimilarityTransform</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func EstimateAffinePartial2D(from, to []PointF) (SimilarityTransform, bool)` | EstimateAffinePartial2D estimates the best-fit similarity transform (uniform scale + rotation + translation, four degrees of freedom) mapping the… |

</details>

<details>
<summary><code>TermCriteria</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTermCriteria(maxCount int, epsilon float64) TermCriteria` | NewTermCriteria returns a TermCriteria with the given iteration cap and accuracy threshold. |

</details>

<details>
<summary><code>TrackerFeaturePyrLK</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTrackerFeaturePyrLK(winSize, maxLevel int) *TrackerFeaturePyrLK` | NewTrackerFeaturePyrLK creates a tracker with the given Lucas-Kanade window size and pyramid depth. |
| `func (t *TrackerFeaturePyrLK) Init(frame *cv.Mat, points []cv.Point)` | Init seeds the tracker with the first frame and the feature points to follow. |
| `func (t *TrackerFeaturePyrLK) Points() []cv.Point` | Points returns a copy of the tracker's current point set. |
| `func (t *TrackerFeaturePyrLK) Update(frame *cv.Mat) (points []cv.Point, status []bool)` | Update tracks the current point set from the previous frame into frame. |

</details>

<details>
<summary><code>VideoStabilizer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVideoStabilizer(radius int) *VideoStabilizer` | NewVideoStabilizer creates a stabilizer with the given trailing smoothing radius (in frames). |
| `func (s *VideoStabilizer) Stabilize(frame *cv.Mat) *cv.Mat` | Stabilize consumes the next frame of the sequence and returns its stabilized version (same dimensions and channel count). |

</details>

### Constants

`DISPresetUltrafast`, `DISPresetFast`, `DISPresetMedium`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/video).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
