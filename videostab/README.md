# videostab — pure-Go port of a working subset of OpenCV's videostab module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/videostab.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/videostab)

Package videostab is a pure-Go port of a working subset of OpenCV's videostab
module: global-motion estimation, camera-trajectory smoothing and the one-pass
/ two-pass video stabilization pipelines, together with the supporting
border-inpainting and deblurring stages.

It is written entirely against the Go standard library, the root module
github.com/malcolmston/opencv (imported as cv) and the sibling subpackage
github.com/malcolmston/opencv/video (pyramidal Lucas-Kanade tracking and the
closed-form similarity fit); the companion
github.com/malcolmston/opencv/optflow package supplies interchangeable
dense-flow front-ends. There is no cgo.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/videostab"
```

## Usage

This is the package's own `Example`, so it compiles and its output is
asserted on every `go test ./videostab/`.

```go
const size, margin, n = 64, 10, 3

	base := makeTexture(size+2*margin, size+2*margin, 3)
	rng := rand.New(rand.NewSource(4))
	frames := make([]*cv.Mat, n)
	for i := 0; i < n; i++ {
		ox := rng.Intn(2*margin + 1)
		oy := rng.Intn(2*margin + 1)
		frames[i] = base.Region(oy, ox, size, size)
	}

	s := videostab.NewOnePassStabilizer(5)
	s.SetFrames(frames)
	out := s.Stabilize()

	fmt.Printf("stabilized %d frames, size %dx%d\n", len(out), out[0].Rows, out[0].Cols)
```

```
stabilized 3 frames, size 64x64
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func CalcBlurriness(frame *cv.Mat) float64` | CalcBlurriness returns a scalar blurriness measure of a frame, matching cv::videostab::calcBlurriness. |

### Types

| Type | What it is |
| --- | --- |
| `ColorAverageInpainter` | ColorAverageInpainter fills each missing pixel with the average of the same pixel in the temporal neighbours where it is known, assuming the frames… |
| `ColorInpainter` | ColorInpainter fills holes by diffusion: each missing pixel is repeatedly replaced with the average of its currently-known neighbours until the… |
| `DeblurerBase` | DeblurerBase removes motion blur from a stabilized frame in place, using the surrounding frames and their motions as context. |
| `GaussianMotionFilter` | GaussianMotionFilter smooths the camera trajectory by replacing the transform at each frame with a Gaussian-weighted average of the relative motions… |
| `ImageMotionEstimator` | ImageMotionEstimator estimates the global 2-D motion directly from a pair of consecutive frames. |
| `Inpainter` | Inpainter fills the masked (missing) region of a frame, for example the empty border introduced when a frame is warped for stabilization. |
| `InpainterContext` | InpainterContext holds the shared context propagated to inpainters: the frame buffer, the inter-frame motions, the per-frame stabilization warps and… |
| `InpaintingPipeline` | InpaintingPipeline chains several inpainters, running each in turn so that later inpainters only see the holes their predecessors could not fill. |
| `KeypointBasedMotionEstimator` | KeypointBasedMotionEstimator recovers global motion by detecting corner features in the first frame, tracking them into the second frame with… |
| `LpMotionStabilizer` | LpMotionStabilizer computes stabilization warps by fitting a smooth camera path to the accumulated trajectory. |
| `Motion` | Motion is a 2-D transformation stored as a row-major 3×3 homogeneous matrix: |
| `MotionEstimatorBase` | MotionEstimatorBase estimates a global 2-D motion from a set of point correspondences. |
| `MotionEstimatorL1` | MotionEstimatorL1 estimates global motion by minimising the sum of absolute reprojection errors (an L1 fit). |
| `MotionEstimatorRansacL2` | MotionEstimatorRansacL2 estimates global motion with a RANSAC hypothesis loop followed by an L2 (least-squares) refinement over the recovered inlier… |
| `MotionFilterBase` | MotionFilterBase is a per-frame motion filter: it can produce the stabilization warp for a single frame in addition to the batch `MotionStabilizer`… |
| `MotionInpainter` | MotionInpainter fills holes with content borrowed from neighbouring frames that are motion-compensated into the target frame's coordinate system,… |
| `MotionModel` | MotionModel enumerates the 2-D transformation models that the global-motion estimators in this package can fit between two frames. |
| `MotionStabilizationPipeline` | MotionStabilizationPipeline chains several motion stabilizers, applying each in turn to the running stabilization estimate. |
| `MotionStabilizer` | MotionStabilizer converts a sequence of inter-frame motions into a sequence of per-frame stabilization warps. |
| `NullDeblurer` | NullDeblurer performs no deblurring. |
| `OnePassStabilizer` | OnePassStabilizer stabilizes a sequence in a single pass by smoothing the camera trajectory online with a `GaussianMotionFilter`. |
| `Range` | Range is an inclusive index range [First, Last] over the frame sequence. |
| `RansacParams` | RansacParams configures the RANSAC loop used by `MotionEstimatorRansacL2`. |
| `StabilizerBase` | StabilizerBase holds the configuration and per-run state shared by `OnePassStabilizer` and `TwoPassStabilizer`: the frame buffer, the estimated… |
| `TwoPassStabilizer` | TwoPassStabilizer stabilizes a sequence in two passes: the first pass estimates every inter-frame motion and runs a global `MotionStabilizer` (which… |
| `WeightingDeblurer` | WeightingDeblurer sharpens a frame with an unsharp mask whose strength is weighted towards the sharpness of the frame's temporal neighbours,… |

<details>
<summary><code>ColorAverageInpainter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewColorAverageInpainter() *ColorAverageInpainter` | NewColorAverageInpainter creates a temporal-average inpainter. |
| `func (c *ColorAverageInpainter) Inpaint(idx int, frame, mask *cv.Mat)` | Inpaint averages known neighbour-frame pixels into the holes. |
| `func (c *ColorAverageInpainter) SetContext(frames []*cv.Mat, motions, stab []Motion, radius int)` | SetContext supplies the frame buffer and radius. |

</details>

<details>
<summary><code>ColorInpainter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewColorInpainter() *ColorInpainter` | NewColorInpainter creates a diffusion inpainter with a sensible sweep count. |
| `func (c *ColorInpainter) Inpaint(_ int, frame, mask *cv.Mat)` | Inpaint fills holes by iterative neighbour averaging (a discrete diffusion). |
| `func (c *ColorInpainter) SetContext([]*cv.Mat, []Motion, []Motion, int)` | SetContext is a no-op; ColorInpainter uses only the frame itself. |

</details>

<details>
<summary><code>GaussianMotionFilter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGaussianMotionFilter(radius int, stdev float64) *GaussianMotionFilter` | NewGaussianMotionFilter creates a Gaussian filter with the given trailing and leading radius (in frames). |
| `func (f *GaussianMotionFilter) Radius() int` | Radius returns the filter radius. |
| `func (f *GaussianMotionFilter) Stabilize(size int, motions []Motion, rng Range, out []Motion)` | Stabilize fills stabilizationMotions with the per-frame Gaussian-smoothed transforms. |
| `func (f *GaussianMotionFilter) StabilizeAt(idx int, motions []Motion, rng Range) Motion` | StabilizeAt returns the Gaussian-weighted average transform for frame idx. |
| `func (f *GaussianMotionFilter) Stdev() float64` | Stdev returns the Gaussian standard deviation used for weighting. |

</details>

<details>
<summary><code>InpaintingPipeline</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewInpaintingPipeline() *InpaintingPipeline` | NewInpaintingPipeline creates an empty inpainting pipeline. |
| `func (p *InpaintingPipeline) Add(in Inpainter) *InpaintingPipeline` | Add appends an inpainter and returns the pipeline for chaining. |
| `func (p *InpaintingPipeline) Inpaint(idx int, frame, mask *cv.Mat)` | Inpaint runs every inpainter in sequence. |
| `func (p *InpaintingPipeline) Len() int` | Len reports how many inpainters the pipeline contains. |
| `func (p *InpaintingPipeline) SetContext(frames []*cv.Mat, motions, stab []Motion, radius int)` | SetContext propagates the shared context to every inpainter in the pipeline. |

</details>

<details>
<summary><code>KeypointBasedMotionEstimator</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewKeypointBasedMotionEstimator(base MotionEstimatorBase) *KeypointBasedMotionEstimator` | NewKeypointBasedMotionEstimator wraps a sparse motion estimator with feature detection and optical-flow tracking, using defaults tuned for general… |
| `func (k *KeypointBasedMotionEstimator) Base() MotionEstimatorBase` | Base returns the wrapped sparse motion estimator. |
| `func (k *KeypointBasedMotionEstimator) Estimate(frame0, frame1 *cv.Mat) (Motion, bool)` | Estimate detects features in frame0, tracks them into frame1 and fits the wrapped model to the surviving correspondences. |
| `func (k *KeypointBasedMotionEstimator) MotionModel() MotionModel` | MotionModel returns the model of the wrapped estimator. |

</details>

<details>
<summary><code>LpMotionStabilizer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewLpMotionStabilizer() *LpMotionStabilizer` | NewLpMotionStabilizer returns an LP-style stabilizer with OpenCV's default derivative weights (1, 10, 100). |
| `func (s *LpMotionStabilizer) SetWeights(w1, w2, w3 float64)` | SetWeights overrides the first/second/third derivative penalty weights. |
| `func (s *LpMotionStabilizer) Stabilize(size int, motions []Motion, rng Range, out []Motion)` | Stabilize smooths each element of the accumulated affine trajectory and emits the correction warp for every frame. |

</details>

<details>
<summary><code>Motion</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func EstimateGlobalMotionLeastSquares(from, to []video.PointF, model MotionModel) (Motion, bool)` | EstimateGlobalMotionLeastSquares fits the model to the correspondences with a single ordinary least-squares pass (no outlier rejection). |
| `func GetMotion(from, to int, motions []Motion) Motion` | GetMotion composes the per-consecutive-frame motions to obtain the transform mapping coordinates of frame from into coordinates of frame to. |
| `func IdentityMotion() Motion` | IdentityMotion returns the identity transform. |
| `func SimilarityMotion(scale, angle, tx, ty float64) Motion` | SimilarityMotion returns a similarity transform with the given uniform scale, rotation angle (radians, counter-clockwise) and translation. |
| `func TranslationMotion(tx, ty float64) Motion` | TranslationMotion returns a pure translation by (tx, ty). |
| `func (m Motion) Affine() cv.AffineMatrix` | Affine returns the top two rows of the transform as a cv.AffineMatrix, dropping the projective row. |
| `func (m Motion) Apply(x, y float64) (float64, float64)` | Apply maps the point (x, y) through the transform, performing the homogeneous divide. |
| `func (m Motion) Determinant() float64` | Determinant returns the determinant of the 3×3 matrix. |
| `func (m Motion) Inverse() (Motion, bool)` | Inverse returns the matrix inverse and reports whether the transform is invertible. |
| `func (m Motion) Mul(n Motion) Motion` | Mul returns the matrix product m·n (apply n first, then m). |

</details>

<details>
<summary><code>MotionEstimatorL1</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMotionEstimatorL1(model MotionModel) *MotionEstimatorL1` | NewMotionEstimatorL1 creates an L1 estimator for the given model. |
| `func (e *MotionEstimatorL1) Estimate(from, to []video.PointF) (Motion, bool)` | Estimate fits the model in the L1 sense via iteratively reweighted least squares. |
| `func (e *MotionEstimatorL1) MotionModel() MotionModel` | MotionModel returns the fitted model. |
| `func (e *MotionEstimatorL1) SetMotionModel(m MotionModel)` | SetMotionModel changes the fitted model. |

</details>

<details>
<summary><code>MotionEstimatorRansacL2</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMotionEstimatorRansacL2(model MotionModel) *MotionEstimatorRansacL2` | NewMotionEstimatorRansacL2 creates a RANSAC/L2 estimator for the given model with default RANSAC parameters and a fixed random seed (so results are… |
| `func (e *MotionEstimatorRansacL2) Estimate(from, to []video.PointF) (Motion, bool)` | Estimate fits the model with RANSAC and refines it over the inliers. |
| `func (e *MotionEstimatorRansacL2) MotionModel() MotionModel` | MotionModel returns the fitted model. |
| `func (e *MotionEstimatorRansacL2) RansacParams() RansacParams` | RansacParams returns the current RANSAC configuration. |
| `func (e *MotionEstimatorRansacL2) SetMinInlierRatio(r float64)` | SetMinInlierRatio sets the minimum fraction of inliers required to accept an estimate. |
| `func (e *MotionEstimatorRansacL2) SetMotionModel(m MotionModel)` | SetMotionModel changes the fitted model and resets the RANSAC parameters to the model's defaults. |
| `func (e *MotionEstimatorRansacL2) SetRansacParams(p RansacParams)` | SetRansacParams overrides the RANSAC configuration. |
| `func (e *MotionEstimatorRansacL2) SetSeed(seed int64)` | SetSeed reseeds the internal random number generator, allowing reproducible but configurable RANSAC sampling. |

</details>

<details>
<summary><code>MotionInpainter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMotionInpainter() *MotionInpainter` | NewMotionInpainter creates a motion-compensated inpainter. |
| `func (m *MotionInpainter) Inpaint(idx int, frame, mask *cv.Mat)` | Inpaint fills holes from motion-compensated neighbour frames, nearest first, then guarantees a hole-free result with a diffusion pass. |
| `func (m *MotionInpainter) SetContext(frames []*cv.Mat, motions, stab []Motion, radius int)` | SetContext supplies the frame buffer, motions, stabilization warps and radius. |

</details>

<details>
<summary><code>MotionModel</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (m MotionModel) String() string` | String returns the canonical name of the motion model. |

</details>

<details>
<summary><code>MotionStabilizationPipeline</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMotionStabilizationPipeline() *MotionStabilizationPipeline` | NewMotionStabilizationPipeline creates an empty pipeline. |
| `func (p *MotionStabilizationPipeline) Add(s MotionStabilizer) *MotionStabilizationPipeline` | Add appends a stabilizer to the pipeline and returns the pipeline for chaining. |
| `func (p *MotionStabilizationPipeline) Len() int` | Len reports how many stabilizers the pipeline contains. |
| `func (p *MotionStabilizationPipeline) Stabilize(size int, motions []Motion, rng Range, out []Motion)` | Stabilize runs every stabilizer in sequence. |

</details>

<details>
<summary><code>NullDeblurer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (NullDeblurer) Deblur(int, *cv.Mat)` | Deblur does nothing. |
| `func (NullDeblurer) SetContext([]*cv.Mat, []Motion, []float64)` | SetContext does nothing. |

</details>

<details>
<summary><code>OnePassStabilizer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewOnePassStabilizer(radius int) *OnePassStabilizer` | NewOnePassStabilizer creates a one-pass stabilizer with the given smoothing radius. |
| `func (s *OnePassStabilizer) NextFrame() (*cv.Mat, bool)` | NextFrame returns the next stabilized frame. |
| `func (s *OnePassStabilizer) SetMotionFilter(f MotionFilterBase)` | SetMotionFilter overrides the trajectory-smoothing filter. |
| `func (s *OnePassStabilizer) Stabilize() []*cv.Mat` | Stabilize returns every stabilized frame of the sequence. |

</details>

<details>
<summary><code>RansacParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultRansacParams(model MotionModel) RansacParams` | DefaultRansacParams returns sensible RANSAC parameters for the given motion model, matching cv::videostab::RansacParams::default2dMotion: the minimal… |
| `func (p RansacParams) NumIters() int` | NumIters returns the number of RANSAC iterations required to reach the configured success probability given the outlier ratio and sample size,… |

</details>

<details>
<summary><code>StabilizerBase</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (s *StabilizerBase) Frames() []*cv.Mat` | Frames returns the input frame sequence. |
| `func (s *StabilizerBase) Motions() []Motion` | Motions returns the estimated inter-frame motions (available after the first NextFrame call). |
| `func (s *StabilizerBase) Radius() int` | Radius returns the temporal radius. |
| `func (s *StabilizerBase) SetDeblurer(d DeblurerBase)` | SetDeblurer enables deblurring with the given deblurer. |
| `func (s *StabilizerBase) SetFrames(frames []*cv.Mat)` | SetFrames sets the input frame sequence and resets the stabilizer. |
| `func (s *StabilizerBase) SetInpainter(in Inpainter)` | SetInpainter enables border inpainting with the given inpainter. |
| `func (s *StabilizerBase) SetMotionEstimator(e ImageMotionEstimator)` | SetMotionEstimator overrides the image motion estimator. |
| `func (s *StabilizerBase) SetRadius(r int)` | SetRadius sets the temporal radius used for trajectory smoothing and for the inpainting / deblurring neighbourhoods. |
| `func (s *StabilizerBase) SetTrimRatio(r float64)` | SetTrimRatio sets the fraction of the frame trimmed from every side (and zoomed back to full size) to hide the empty border produced by warping. |
| `func (s *StabilizerBase) StabilizationMotions() []Motion` | StabilizationMotions returns the per-frame stabilization warps (available after the first NextFrame call). |
| `func (s *StabilizerBase) TrimRatio() float64` | TrimRatio returns the current trim ratio. |

</details>

<details>
<summary><code>TwoPassStabilizer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTwoPassStabilizer(radius int) *TwoPassStabilizer` | NewTwoPassStabilizer creates a two-pass stabilizer with the given radius. |
| `func (s *TwoPassStabilizer) NextFrame() (*cv.Mat, bool)` | NextFrame returns the next stabilized frame. |
| `func (s *TwoPassStabilizer) SetMotionStabilizer(m MotionStabilizer)` | SetMotionStabilizer overrides the global motion stabilizer used in the first pass. |
| `func (s *TwoPassStabilizer) Stabilize() []*cv.Mat` | Stabilize returns every stabilized frame of the sequence. |

</details>

<details>
<summary><code>WeightingDeblurer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewWeightingDeblurer(radius int) *WeightingDeblurer` | NewWeightingDeblurer creates a weighting deblurer with the given temporal radius and sensible unsharp defaults (base amount 1.0, sensitivity 0.8, a… |
| `func (d *WeightingDeblurer) Deblur(idx int, frame *cv.Mat)` | Deblur sharpens the frame at index idx in place with a neighbour-weighted unsharp mask. |
| `func (d *WeightingDeblurer) SetContext(frames []*cv.Mat, motions []Motion, blurriness []float64)` | SetContext supplies the frame buffer, motions and per-frame blurriness measures. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/videostab).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
