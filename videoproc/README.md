# videoproc — implements temporal (video) computer-vision primitives on top of the root cv package

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/videoproc.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/videoproc)

Package videoproc implements temporal (video) computer-vision primitives on
top of the root cv package (github.com/malcolmston/opencv), mirroring a useful
subset of OpenCV's video, bgsegm and optflow modules.

Every routine operates on the library's existing image type, cv.Mat, and
produces cv.Mat masks or cv.FloatMat accumulators; no competing image type is
introduced. The only small helper type added is `PointF`, a floating-point
image coordinate used by the tracking and trajectory routines, and the
container types `FlowField` and `Trajectory`.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/videoproc"
```

## Usage

This is the package's own `ExampleAccumulateWeighted`, so it compiles and its output is
asserted on every `go test ./videoproc/`.

```go
acc := &cv.FloatMat{Rows: 1, Cols: 4, Data: make([]float64, 4)}

	dark := cv.NewMat(1, 4, 1)
	bright := cv.NewMat(1, 4, 1)
	for i := range bright.Data {
		bright.Data[i] = 200
	}

	videoproc.Accumulate(dark, acc)
	fmt.Println("after one dark frame:", acc.Data)

	videoproc.AccumulateWeighted(bright, acc, 0.5)
	fmt.Println("half-way to bright:  ", acc.Data)
```

```
after one dark frame: [0 0 0 0]
half-way to bright:   [100 100 100 100]
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AbsDiff(a, b *cv.Mat) *cv.Mat` | AbsDiff returns the per-sample absolute difference \|a-b\| of two frames of identical dimensions and channel count. |
| `func Accumulate(src *cv.Mat, acc *cv.FloatMat)` | Accumulate adds the intensity of frame src into the running accumulator acc (acc += src). |
| `func AccumulateSquare(src *cv.Mat, acc *cv.FloatMat)` | AccumulateSquare adds the squared intensity of frame src into acc (acc += src²). |
| `func AccumulateWeighted(src *cv.Mat, acc *cv.FloatMat, alpha float64)` | AccumulateWeighted updates the running-average accumulator acc with frame src using the exponential update acc = (1-alpha)*acc + alpha*src, mirroring… |
| `func AccumulatorToMat(acc *cv.FloatMat, count int) *cv.Mat` | AccumulatorToMat converts a running accumulator into a viewable single-channel Mat by dividing every sample by count (the number of accumulated… |
| `func BlendFrames(a, b *cv.Mat, t float64) *cv.Mat` | BlendFrames returns the linear cross-fade (1-t)*a + t*b of two frames of identical size and channel count, rounded and clamped to 8-bit. |
| `func CountMotionPixels(mask *cv.Mat) int` | CountMotionPixels returns the number of non-zero pixels in a single-channel mask such as the output of `FrameDifference`. |
| `func CrossFade(a, b *cv.Mat, count int) []*cv.Mat` | CrossFade returns count intermediate frames evenly spaced strictly between a and b (exclusive of both endpoints), each a linear blend produced by… |
| `func DetectShotBoundaries(frames []*cv.Mat, threshold float64, metric func(a, b *cv.Mat) float64) []int` | DetectShotBoundaries runs a `ShotBoundaryDetector` over an ordered slice of frames and returns the indices at which a cut begins: an index i in the… |
| `func EdgeChangeRatio(a, b *cv.Mat, edgeThreshold float64) float64` | EdgeChangeRatio returns the edge change ratio between two frames of equal size: the fraction of edge pixels that appear or disappear from a to b,… |
| `func EstimateGlobalTranslation(prev, cur *cv.Mat, maxCorners, searchRadius int) (dx, dy float64, ok bool)` | EstimateGlobalTranslation estimates the dominant inter-frame translation from prev to cur by a feature-based method: it detects up to maxCorners… |
| `func FrameDifference(prev, cur *cv.Mat, threshold uint8) *cv.Mat` | FrameDifference returns a binary motion mask (255 = motion, 0 = static) by thresholding the absolute grayscale difference between prev and cur: a… |
| `func GlobalMotionOrientation(orientation *cv.FloatMat, mask *cv.Mat) (float64, bool)` | GlobalMotionOrientation returns the dominant motion direction in degrees ([0,360)) as the circular (vector) mean of the valid per-pixel orientations… |
| `func HistogramChiSquare(a, b *cv.Mat) float64` | HistogramChiSquare returns the chi-square distance between the normalised grayscale histograms of two frames: sum over bins of (ha-hb)²/(ha+hb),… |
| `func HistogramL1Difference(a, b *cv.Mat) float64` | HistogramL1Difference returns the L1 (city-block) distance between the normalised grayscale intensity histograms of two frames, a value in [0,2]. |
| `func InterpolateFlow(a, b *cv.Mat, flow *FlowField, t float64) *cv.Mat` | InterpolateFlow synthesises a motion-compensated intermediate frame at time t in [0,1] between a and b, given the forward flow field mapping a→b. |
| `func MeanIntensityDelta(a, b *cv.Mat) float64` | MeanIntensityDelta returns the absolute difference between the mean grayscale intensities of two frames. |
| `func MotionEnergy(mask *cv.Mat) float64` | MotionEnergy returns the fraction of moving pixels in a single-channel mask, i.e. |
| `func MotionGradient(mhi *cv.FloatMat, delta1, delta2 float64) (orientation *cv.FloatMat, mask *cv.Mat)` | MotionGradient computes the spatial gradient orientation of a motion-history image, mirroring cv::calcMotionGradient. |
| `func PixelDifferenceRatio(a, b *cv.Mat, threshold uint8) float64` | PixelDifferenceRatio returns the fraction of pixels (in [0,1]) whose absolute grayscale change between a and b exceeds threshold. |
| `func SampleDenseGrid(rows, cols, step int) []cv.Point` | SampleDenseGrid returns integer sampling locations on a regular grid covering an image of the given size, one point every step pixels in both… |
| `func TemporalAverage(frames []*cv.Mat) *cv.Mat` | TemporalAverage returns the per-sample mean of a stack of frames (all the same size and channel count), rounded and clamped to 8-bit. |
| `func TemporalGaussian(frames []*cv.Mat, sigma float64) *cv.Mat` | TemporalGaussian returns a Gaussian-weighted temporal average of a stack of frames: frame k receives weight exp(-((k-c)²)/(2σ²)) where c is the… |
| `func TemporalMaximum(frames []*cv.Mat) *cv.Mat` | TemporalMaximum returns the per-sample maximum across a stack of frames, the brightest value each sample takes over time. |
| `func TemporalMedian(frames []*cv.Mat) *cv.Mat` | TemporalMedian returns the per-sample median of a stack of frames. |
| `func TemporalMinimum(frames []*cv.Mat) *cv.Mat` | TemporalMinimum returns the per-sample minimum across a stack of frames, the darkest value each sample takes over time. |
| `func ThreeFrameDifference(prev, cur, next *cv.Mat, threshold uint8) *cv.Mat` | ThreeFrameDifference returns a binary motion mask using the classic three-frame difference: a pixel is marked as motion (255) only when it changes… |
| `func UpdateMotionHistory(silhouette *cv.Mat, mhi *cv.FloatMat, timestamp, duration float64)` | UpdateMotionHistory applies one motion-history update step in place, mirroring cv::updateMotionHistory. |
| `func WarpByFlow(src *cv.Mat, flow *FlowField, scale float64) *cv.Mat` | WarpByFlow warps src by a dense flow field using backward mapping with bilinear sampling: the output pixel at (x, y) is sampled from src at… |
| `func WarpTranslate(src *cv.Mat, dx, dy float64) *cv.Mat` | WarpTranslate returns src shifted by (dx, dy): the output pixel at (x, y) is sampled from src at (x-dx, y-dy) with bilinear interpolation and edge… |

### Types

| Type | What it is |
| --- | --- |
| `BackgroundSubtractor` | BackgroundSubtractor is the common interface implemented by every background model in this package. |
| `DenseTrajectorySampler` | DenseTrajectorySampler extracts dense trajectories by seeding points on a grid and tracking each across successive frames with `TrackPoints`,… |
| `ExponentialMovingAverage` | ExponentialMovingAverage is an online per-pixel low-pass temporal filter: state = (1-Alpha)*state + Alpha*frame. |
| `FlowField` | FlowField is a dense two-channel optical-flow field: for each pixel, X holds the horizontal displacement and Y the vertical displacement that maps a… |
| `MOGBackgroundSubtractor` | MOGBackgroundSubtractor models each pixel's intensity as an adaptive mixture of Gaussians (MOG), the algorithm of Stauffer & Grimson. |
| `MedianBackgroundSubtractor` | MedianBackgroundSubtractor models the background as the per-pixel temporal median over a sliding window of the last History frames. |
| `MotionHistory` | MotionHistory maintains a Motion History Image (MHI) as described by Bobick & Davis. |
| `MovingWindowFilter` | MovingWindowFilter is an online temporal filter that keeps the most recent Size frames in a ring buffer and, on each push, returns their per-sample… |
| `PointF` | PointF is a floating-point image coordinate (X is the column, Y is the row). |
| `RunningAverageSubtractor` | RunningAverageSubtractor models the background as a per-pixel exponential running average of intensity, mirroring the simplest… |
| `ShotBoundaryDetector` | ShotBoundaryDetector is an online detector that reports a hard cut whenever the chosen dissimilarity between consecutive frames exceeds Threshold. |
| `Stabilizer` | Stabilizer is a causal (online) digital video stabilizer. |
| `Trajectory` | Trajectory is the ordered path of a single tracked point across consecutive frames, the core datum of dense-trajectory descriptors. |

<details>
<summary><code>DenseTrajectorySampler</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDenseTrajectorySampler(step, maxLength int) *DenseTrajectorySampler` | NewDenseTrajectorySampler returns a sampler with reasonable defaults for the given grid step and maximum trajectory length: search radius 4, patch… |
| `func (s *DenseTrajectorySampler) Active() []*Trajectory` | Active returns the trajectories currently being tracked (not yet completed or lost). |
| `func (s *DenseTrajectorySampler) Feed(frame *cv.Mat) []*Trajectory` | Feed advances every active trajectory to the given frame and returns the trajectories that completed on this frame (reached MaxLength). |

</details>

<details>
<summary><code>ExponentialMovingAverage</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewExponentialMovingAverage(alpha float64) *ExponentialMovingAverage` | NewExponentialMovingAverage returns an EMA filter with the given update weight alpha in (0,1]. |
| `func (e *ExponentialMovingAverage) Update(frame *cv.Mat) *cv.Mat` | Update folds frame into the running state and returns the current smoothed frame as a fresh Mat with the same shape as the input. |
| `func (e *ExponentialMovingAverage) Value() *cv.Mat` | Value returns the current smoothed frame, or nil before the first Update. |

</details>

<details>
<summary><code>FlowField</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewFlowField(rows, cols int) *FlowField` | NewFlowField allocates a zero (no-motion) flow field of the given dimensions. |
| `func (f *FlowField) At(x, y int) (dx, dy float64)` | At returns the (dx, dy) displacement stored at pixel (x, y). |
| `func (f *FlowField) Set(x, y int, dx, dy float64)` | Set stores the displacement (dx, dy) at pixel (x, y). |

</details>

<details>
<summary><code>MOGBackgroundSubtractor</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMOGBackgroundSubtractor() *MOGBackgroundSubtractor` | NewMOGBackgroundSubtractor returns a MOG model with OpenCV-like defaults: five Gaussians per pixel, learning rate 0.01, a 2.5σ (6.25 squared) match… |
| `func (s *MOGBackgroundSubtractor) Apply(frame *cv.Mat) *cv.Mat` | Apply feeds the next frame to the mixture model and returns a single-channel foreground mask (255 = foreground). |
| `func (s *MOGBackgroundSubtractor) Background() *cv.Mat` | Background returns the per-pixel mean of the most reliable Gaussian (largest weight/σ) as an 8-bit Mat, or nil before the first frame. |

</details>

<details>
<summary><code>MedianBackgroundSubtractor</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMedianBackgroundSubtractor(history, threshold int) *MedianBackgroundSubtractor` | NewMedianBackgroundSubtractor returns a median model that keeps the most recent history frames (history >= 1) and uses the given foreground threshold… |
| `func (s *MedianBackgroundSubtractor) Apply(frame *cv.Mat) *cv.Mat` | Apply classifies frame against the current per-pixel median and then inserts it into the sliding window. |
| `func (s *MedianBackgroundSubtractor) Background() *cv.Mat` | Background returns the current per-pixel median background as an 8-bit Mat, or nil before any frame has been supplied. |
| `func (s *MedianBackgroundSubtractor) Ready() bool` | Ready reports whether at least one frame has been supplied, so that `MedianBackgroundSubtractor.Background` returns a usable estimate. |

</details>

<details>
<summary><code>MotionHistory</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMotionHistory(rows, cols int, duration float64) *MotionHistory` | NewMotionHistory allocates a rows×cols motion-history buffer with the given fade Duration. |
| `func (h *MotionHistory) EnergyImage() *cv.Mat` | EnergyImage returns the Motion Energy Image (MEI): a binary mask (255 where any non-faded motion is currently recorded, 0 elsewhere). |
| `func (h *MotionHistory) Image() *cv.FloatMat` | Image returns the underlying motion-history buffer (timestamps per pixel) as a FloatMat. |
| `func (h *MotionHistory) NormalizedImage() *cv.Mat` | NormalizedImage returns the motion-history buffer scaled to 0..255, with the newest motion at 255 and fully faded pixels at 0, suitable for display. |
| `func (h *MotionHistory) Update(silhouette *cv.Mat, timestamp float64)` | Update folds a new silhouette (a single-channel mask where non-zero marks motion) into the history at the given timestamp. |

</details>

<details>
<summary><code>MovingWindowFilter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMovingWindowFilter(size int) *MovingWindowFilter` | NewMovingWindowFilter returns a moving-window averaging filter over the last size frames (size >= 1). |
| `func (w *MovingWindowFilter) Full() bool` | Full reports whether the window has accumulated Size frames. |
| `func (w *MovingWindowFilter) Push(frame *cv.Mat) *cv.Mat` | Push inserts frame into the window and returns the average of all frames currently in the window (including the one just pushed) as a fresh Mat. |

</details>

<details>
<summary><code>PointF</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func SmoothTrajectory(traj []PointF, radius int) []PointF` | SmoothTrajectory smooths a cumulative motion trajectory with a centred moving average of the given radius (window size 2*radius+1), clamping the… |
| `func TrackPoints(prev, cur *cv.Mat, pts []PointF, searchRadius, patchRadius int, maxCost float64) (tracked []PointF, valid []bool)` | TrackPoints tracks a set of points from prev to cur by block matching: around each point it searches an integer displacement within ±searchRadius… |

</details>

<details>
<summary><code>RunningAverageSubtractor</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRunningAverageSubtractor(alpha, threshold float64) *RunningAverageSubtractor` | NewRunningAverageSubtractor returns a running-average model with the given learning rate alpha (in (0,1]) and foreground threshold (0..255). |
| `func (s *RunningAverageSubtractor) Apply(frame *cv.Mat) *cv.Mat` | Apply classifies frame against the running-average background and updates the model. |
| `func (s *RunningAverageSubtractor) Background() *cv.Mat` | Background returns the current running-average background as an 8-bit Mat, or nil before the first frame. |

</details>

<details>
<summary><code>ShotBoundaryDetector</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewShotBoundaryDetector(threshold float64, metric func(a, b *cv.Mat) float64) *ShotBoundaryDetector` | NewShotBoundaryDetector returns a detector using the given metric and decision threshold. |
| `func (d *ShotBoundaryDetector) Add(frame *cv.Mat) (isCut bool, score float64)` | Add feeds the next frame and returns true when a shot boundary is detected between the previous frame and this one (metric value strictly greater… |

</details>

<details>
<summary><code>Stabilizer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewStabilizer(smoothing float64) *Stabilizer` | NewStabilizer returns a stabilizer with the given path-smoothing weight (in (0,1]) and OpenCV-like tracking defaults (200 corners, search radius 8). |
| `func (s *Stabilizer) Stabilize(frame *cv.Mat) (*cv.Mat, bool)` | Stabilize feeds the next frame and returns the motion-compensated frame. |

</details>

<details>
<summary><code>Trajectory</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (t *Trajectory) Displacement() (dx, dy float64)` | Displacement returns the straight-line vector from the trajectory's first to its last point as (dx, dy). |
| `func (t *Trajectory) Length() int` | Length returns the number of positions in the trajectory. |
| `func (t *Trajectory) TotalLength() float64` | TotalLength returns the sum of the Euclidean lengths of the trajectory's segments (the arc length of the path). |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/videoproc).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
