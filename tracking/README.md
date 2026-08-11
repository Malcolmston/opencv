# tracking — implements classical, CPU-only computer-vision tracking and motion-estimation

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/tracking.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/tracking)

Package tracking implements classical, CPU-only computer-vision tracking and
motion-estimation algorithms on top of the root cv package
(github.com/malcolmston/opencv, imported as cv).

The package is written entirely against the Go standard library and the root
module. It has no third-party dependencies, no cgo and no GPU code, and every
routine is deterministic: given the same inputs it always produces the same
output. Images are represented by the library's central *cv.Mat type; the
package converts multi-channel input to grayscale internally where an
algorithm operates on intensity. A few small geometric helper types (Rect,
Point2f, RotatedRect, TermCriteria) and a dense Matrix type for the linear
filters are provided locally, mirroring the convention of the sibling cv/video
and cv/optflow packages, which also define their own small value types rather
than reaching across package boundaries.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/tracking"
```

## Usage

This is the package's own `ExampleCalcBackProjection`, so it compiles and its output is
asserted on every `go test ./tracking/`.

```go
img := cv.NewMat(4, 4, 1)
	for i := range img.Data {
		img.Data[i] = 10
	}
	img.Data[5] = 200
	img.Data[6] = 200

	hist := tracking.CalcHistGray(img, tracking.NewRect(1, 1, 2, 1), 16)
	hist.Normalize()

	back := tracking.CalcBackProjection(img, hist)
	fmt.Printf("target pixels weighted %d, background %d\n", back.Data[5], back.Data[0])
```

```
target pixels weighted 255, background 0
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func BuildOpticalFlowPyramid(img *cv.Mat, maxLevel int) []*cv.Mat` | BuildOpticalFlowPyramid builds a Gaussian image pyramid for coarse-to-fine optical-flow tracking. |
| `func CalcBackProjection(img *cv.Mat, hist *Histogram1D) *cv.Mat` | CalcBackProjection produces a probability image the same size as img in which each pixel holds the (normalised) histogram weight of its grayscale… |
| `func CamShift(prob *cv.Mat, window Rect, crit TermCriteria) (RotatedRect, Rect, int)` | CamShift is the Continuously Adaptive Mean Shift tracker. |
| `func IoU(a, b Rect) float64` | IoU returns the intersection-over-union overlap of two rectangles, a value in [0, 1]. |

### Types

| Type | What it is |
| --- | --- |
| `CentroidTracker` | CentroidTracker assigns stable identities to detections across frames by nearest-centroid association. |
| `FarnebackParams` | FarnebackParams configures the dense block-matching flow estimator `CalcOpticalFlowFarneback`. |
| `FlowField` | FlowField is a dense optical-flow field storing a (u, v) displacement per pixel, where u is the horizontal and v the vertical component measured in… |
| `Histogram1D` | Histogram1D is a one-dimensional intensity histogram over the grayscale range [0, 256). |
| `IoUTracker` | IoUTracker associates detections with tracks across frames by intersection-over-union overlap, the core of the classic "IOU Tracker". |
| `KCFParams` | KCFParams configures the `KCFTracker`. |
| `KCFTracker` | KCFTracker is a lightweight template correlation tracker ("KCF-lite"). |
| `KalmanFilter` | KalmanFilter is a linear Kalman filter following the same layout as OpenCV's cv::KalmanFilter. |
| `LKParams` | LKParams configures the Lucas-Kanade solvers. |
| `MatchResult` | MatchResult reports the best location found by `MatchTemplateNCC`: the top-left corner of the match, the template size that produced it and the… |
| `Matrix` | Matrix is a small dense row-major matrix of float64 values. |
| `Particle` | Particle is a single hypothesis in a `ParticleFilter`: a state vector and its importance weight. |
| `ParticleFilter` | ParticleFilter is a bootstrap (sequential importance resampling) particle filter for non-linear/non-Gaussian tracking. |
| `Point2f` | Point2f is a floating-point image coordinate where X is the column and Y is the row. |
| `Rect` | Rect is an axis-aligned integer bounding box with its top-left corner at (X, Y) and the given Width and Height. |
| `RotatedRect` | RotatedRect is an oriented rectangle described by its center, size and rotation. |
| `TermCriteria` | TermCriteria describes when an iterative algorithm should stop. |
| `Track` | Track is one identity maintained by an `IoUTracker`: a stable ID, its most recent bounding box and lifecycle counters. |
| `TrackedObject` | TrackedObject is one identity maintained by a `CentroidTracker`: a stable integer ID, the current centroid and detection box, and how many… |

<details>
<summary><code>CentroidTracker</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewCentroidTracker(maxDisappeared int, maxDistance float64) *CentroidTracker` | NewCentroidTracker creates a tracker. |
| `func (c *CentroidTracker) Count() int` | Count returns the number of currently tracked objects. |
| `func (c *CentroidTracker) Objects() []TrackedObject` | Objects returns a snapshot of the tracked objects ordered by ID. |
| `func (c *CentroidTracker) Update(detections []Rect) []TrackedObject` | Update associates the detection boxes of the current frame with existing identities and returns a snapshot of all live objects, ordered by ID. |

</details>

<details>
<summary><code>FarnebackParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultFarnebackParams() FarnebackParams` | DefaultFarnebackParams returns sensible defaults (3 levels, window radius 4, search radius 4). |

</details>

<details>
<summary><code>FlowField</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CalcOpticalFlowFarneback(prev, next *cv.Mat, params FarnebackParams) *FlowField` | CalcOpticalFlowFarneback computes a dense optical-flow field from prev to next. |
| `func CalcOpticalFlowHornSchunck(prev, next *cv.Mat, alpha float64, iters int) *FlowField` | CalcOpticalFlowHornSchunck computes a dense optical-flow field from prev to next using the classic Horn-Schunck variational method. |
| `func NewFlowField(rows, cols int) *FlowField` | NewFlowField allocates a zero (no-motion) flow field of the given size. |
| `func (f *FlowField) At(y, x int) Point2f` | At returns the displacement vector at pixel (y, x) as a Point2f whose X is the horizontal and Y the vertical flow. |
| `func (f *FlowField) MagnitudeAt(y, x int) float64` | MagnitudeAt returns the Euclidean length of the flow vector at pixel (y, x). |
| `func (f *FlowField) MaxMagnitude() float64` | MaxMagnitude returns the largest flow-vector length in the field. |
| `func (f *FlowField) MeanFlow() Point2f` | MeanFlow returns the average displacement over the whole field. |
| `func (f *FlowField) SetFlow(y, x int, du, dv float64)` | SetFlow stores the displacement (du, dv) at pixel (y, x). |

</details>

<details>
<summary><code>Histogram1D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CalcHistGray(img *cv.Mat, roi Rect, bins int) *Histogram1D` | CalcHistGray builds an intensity histogram of the grayscale of img over the pixels inside roi. |
| `func NewHistogram1D(bins int) *Histogram1D` | NewHistogram1D allocates a zeroed histogram with the given number of bins, which must be between 1 and 256. |
| `func (h *Histogram1D) Normalize()` | Normalize scales the histogram so its largest bin equals 1. |

</details>

<details>
<summary><code>IoUTracker</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewIoUTracker(iouThreshold float64, maxAge int) *IoUTracker` | NewIoUTracker creates a tracker. |
| `func (t *IoUTracker) Count() int` | Count returns the number of live tracks. |
| `func (t *IoUTracker) Tracks() []Track` | Tracks returns a snapshot of the live tracks ordered by ID. |
| `func (t *IoUTracker) Update(detections []Rect) []Track` | Update associates the detection boxes of the current frame with existing tracks by IoU and returns a snapshot of the live tracks ordered by ID. |

</details>

<details>
<summary><code>KCFParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultKCFParams() KCFParams` | DefaultKCFParams returns the default parameters (adaptive search radius, learning rate 0.1). |

</details>

<details>
<summary><code>KCFTracker</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewKCFTracker(params KCFParams) *KCFTracker` | NewKCFTracker creates an uninitialised tracker with the given parameters. |
| `func (t *KCFTracker) BoundingBox() Rect` | BoundingBox returns the tracker's current target box. |
| `func (t *KCFTracker) Init(img *cv.Mat, box Rect)` | Init sets the target from box in img and builds the initial appearance model. |
| `func (t *KCFTracker) Update(img *cv.Mat) (Rect, float64)` | Update locates the target in the next frame img, returns the new bounding box and the correlation confidence in [-1, 1], and updates the internal… |

</details>

<details>
<summary><code>KalmanFilter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewConstantVelocityKalman2D(x0, y0, dt, processNoise, measureNoise float64) *KalmanFilter` | NewConstantVelocityKalman2D constructs a Kalman filter for tracking a 2-D point under a constant-velocity motion model. |
| `func NewKalmanFilter(dynamParams, measureParams, controlParams int) *KalmanFilter` | NewKalmanFilter constructs a Kalman filter with the given state, measurement and control dimensions (controlParams may be 0 for no control input). |
| `func (kf *KalmanFilter) Correct(measurement []float64) []float64` | Correct updates the predicted state with the measurement vector and returns the corrected state as a fresh slice. |
| `func (kf *KalmanFilter) Predict(control []float64) []float64` | Predict advances the filter one step and returns the predicted state as a fresh slice. |
| `func (kf *KalmanFilter) StateVector() []float64` | StateVector returns the current corrected state as a fresh slice. |

</details>

<details>
<summary><code>LKParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultLKParams() LKParams` | DefaultLKParams returns the default Lucas-Kanade parameters (window radius 3, 10 iterations, epsilon 0.01, min-eigen threshold 1e-4). |

</details>

<details>
<summary><code>MatchResult</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MatchTemplateNCC(img, templ *cv.Mat) MatchResult` | MatchTemplateNCC slides templ over img and returns the position of maximum normalized cross-correlation. |
| `func (r MatchResult) BoundingBox() Rect` | BoundingBox returns the matched region as a Rect. |

</details>

<details>
<summary><code>Matrix</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func IdentityMatrix(n int) *Matrix` | IdentityMatrix returns the n-by-n identity matrix. |
| `func MatrixFromRows(rows [][]float64) *Matrix` | MatrixFromRows builds a matrix from a slice of equal-length rows. |
| `func NewMatrix(rows, cols int) *Matrix` | NewMatrix allocates a zero-filled matrix with the given shape. |
| `func (m *Matrix) Add(o *Matrix) *Matrix` | Add returns the element-wise sum m+o. |
| `func (m *Matrix) At(r, c int) float64` | At returns the element at (r, c). |
| `func (m *Matrix) Clone() *Matrix` | Clone returns an independent copy of the matrix. |
| `func (m *Matrix) Inverse() *Matrix` | Inverse returns the inverse of a square matrix, computed by Gauss-Jordan elimination with partial pivoting. |
| `func (m *Matrix) Mul(o *Matrix) *Matrix` | Mul returns the matrix product m*o. |
| `func (m *Matrix) Scale(s float64) *Matrix` | Scale returns the matrix with every element multiplied by s. |
| `func (m *Matrix) Set(r, c int, v float64)` | Set stores v at (r, c). |
| `func (m *Matrix) Sub(o *Matrix) *Matrix` | Sub returns the element-wise difference m-o. |
| `func (m *Matrix) Transpose() *Matrix` | Transpose returns the transpose of the matrix. |

</details>

<details>
<summary><code>ParticleFilter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewParticleFilter(numParticles, stateDim int, seed int64) *ParticleFilter` | NewParticleFilter creates a filter with numParticles hypotheses of dimension stateDim, seeded for deterministic behaviour. |
| `func (pf *ParticleFilter) Count() int` | Count returns the number of particles. |
| `func (pf *ParticleFilter) EffectiveSampleSize() float64` | EffectiveSampleSize returns the effective number of particles 1/Σ wᵢ², a measure of weight degeneracy. |
| `func (pf *ParticleFilter) Estimate() []float64` | Estimate returns the weighted-mean state over all particles. |
| `func (pf *ParticleFilter) Init(mean, std []float64)` | Init scatters the particles around mean with independent Gaussian spread std (both of length stateDim) and resets the weights to uniform. |
| `func (pf *ParticleFilter) Particles() []Particle` | Particles returns a snapshot copy of the current particle set with their weights. |
| `func (pf *ParticleFilter) Predict(motion []float64)` | Predict advances every particle by the deterministic motion vector (which may be nil for a stationary model) and adds the configured Gaussian process… |
| `func (pf *ParticleFilter) Resample()` | Resample performs low-variance systematic resampling, replacing the particle set with a new one drawn in proportion to the current weights and… |
| `func (pf *ParticleFilter) SetProcessNoise(std []float64)` | SetProcessNoise sets the per-dimension standard deviation of the Gaussian process noise added on each `ParticleFilter.Predict`. |
| `func (pf *ParticleFilter) UpdateWeights(likelihood func(state []float64) float64)` | UpdateWeights multiplies each particle's weight by the likelihood that the current measurement was generated from its state, as returned by… |

</details>

<details>
<summary><code>Point2f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CalcOpticalFlowLK(prev, next *cv.Mat, pts []Point2f, params LKParams) (nextPts []Point2f, status []bool)` | CalcOpticalFlowLK computes sparse Lucas-Kanade optical flow at a single image resolution. |
| `func CalcOpticalFlowPyrLK(prev, next *cv.Mat, pts []Point2f, maxLevel int, params LKParams) (nextPts []Point2f, status []bool)` | CalcOpticalFlowPyrLK computes sparse Lucas-Kanade optical flow with a coarse-to-fine Gaussian image pyramid, the standard technique for handling… |
| `func Pt2f(x, y float64) Point2f` | Pt2f returns the Point2f with the given coordinates. |
| `func (p Point2f) Add(o Point2f) Point2f` | Add returns the component-wise sum of p and o. |
| `func (p Point2f) Distance(o Point2f) float64` | Distance returns the Euclidean distance between p and o. |
| `func (p Point2f) Norm() float64` | Norm returns the Euclidean length of p treated as a vector from the origin. |
| `func (p Point2f) Scale(s float64) Point2f` | Scale returns p with both components multiplied by s. |
| `func (p Point2f) String() string` | String renders the point as "(x, y)". |
| `func (p Point2f) Sub(o Point2f) Point2f` | Sub returns the component-wise difference p-o. |

</details>

<details>
<summary><code>Rect</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MeanShift(prob *cv.Mat, window Rect, crit TermCriteria) (result Rect, iterations int)` | MeanShift iterates a search window towards the mode (centroid of probability mass) of a single-channel probability image, the output of… |
| `func NewRect(x, y, width, height int) Rect` | NewRect returns the Rect with the given position and size. |
| `func (r Rect) Area() int` | Area returns the number of pixels covered by the rectangle, clamped to zero for degenerate boxes. |
| `func (r Rect) Bottom() int` | Bottom returns the exclusive bottom edge Y+Height. |
| `func (r Rect) Center() Point2f` | Center returns the geometric center of the rectangle as a sub-pixel point. |
| `func (r Rect) Contains(x, y int) bool` | Contains reports whether the integer point (x, y) lies inside the rectangle, treating the right and bottom edges as exclusive. |
| `func (r Rect) Empty() bool` | Empty reports whether the rectangle has no area. |
| `func (r Rect) Intersect(o Rect) Rect` | Intersect returns the overlapping rectangle of r and o. |
| `func (r Rect) IoU(o Rect) float64` | IoU returns the intersection-over-union overlap of r and o, a value in [0, 1]. |
| `func (r Rect) Right() int` | Right returns the exclusive right edge X+Width. |
| `func (r Rect) String() string` | String renders the rectangle as "[WxH from (X, Y)]". |
| `func (r Rect) Union(o Rect) Rect` | Union returns the smallest rectangle that contains both r and o. |

</details>

<details>
<summary><code>RotatedRect</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (r RotatedRect) BoundingRect() Rect` | BoundingRect returns the smallest axis-aligned integer rectangle that fully contains the rotated rectangle. |

</details>

<details>
<summary><code>TermCriteria</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTermCriteria(maxCount int, epsilon float64) TermCriteria` | NewTermCriteria returns a TermCriteria with the given iteration cap and accuracy threshold. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/tracking).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
