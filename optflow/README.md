# optflow — implements extended and dense optical-flow algorithms on top of the root cv package

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/optflow.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/optflow)

Package optflow implements extended and dense optical-flow algorithms on top
of the root cv package, mirroring a useful subset of OpenCV's contrib optflow
module.

The package is written entirely against the Go standard library and the root
module github.com/malcolmston/opencv (imported as cv). It deliberately does
not depend on any sibling subpackage — in particular it is distinct from and
self-contained relative to cv/video, which offers only sparse pyramidal
Lucas-Kanade and a block-matching Farneback stand-in. Everything optflow needs
(grayscale conversion, gradients, Gaussian pyramids, sub-pixel sampling and a
small local Lucas-Kanade solver) is reimplemented locally so the module has no
hidden coupling.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/optflow"
```

## Usage

This is the package's own `ExampleAverageEndpointError`, so it compiles and its output is
asserted on every `go test ./optflow/`.

```go
est := optflow.NewFlowField(2, 2)
	gt := optflow.NewFlowField(2, 2)
	for y := 0; y < 2; y++ {
		for x := 0; x < 2; x++ {
			gt.Set(y, x, 1, 0)
			est.Set(y, x, 1, 0)
		}
	}
	est.Set(0, 0, 1, 1)
	fmt.Printf("AEE = %.2f\n", optflow.AverageEndpointError(est, gt))
```

```
AEE = 0.25
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AngularError(est, gt *FlowField) []float64` | AngularError returns the per-pixel angular error (in radians) between an estimated and a ground-truth flow. |
| `func AverageAngularError(est, gt *FlowField) float64` | AverageAngularError returns the mean angular error (AAE, in radians) over the whole field. |
| `func AverageEndpointError(est, gt *FlowField) float64` | AverageEndpointError returns the mean endpoint error (AEE) over the whole field — the single most common scalar accuracy score for optical flow. |
| `func EndpointError(est, gt *FlowField) []float64` | EndpointError returns the per-pixel endpoint error between an estimated flow and a ground-truth flow: for each pixel the Euclidean distance |
| `func FlowToColor(flow *FlowField) *cv.Mat` | FlowToColor renders a flow field as a three-channel RGB image using the Middlebury colour-wheel convention: the hue of each pixel encodes the flow… |
| `func WarpByFlow(img *cv.Mat, flow *FlowField) *cv.Mat` | WarpByFlow warps img by the given flow field and returns a new image of the same size. |
| `func WriteFlow(w io.Writer, flow *FlowField) error` | WriteFlow serialises a FlowField to w in the Middlebury .flo format: the ASCII tag "PIEH" (the float32 202021.25), then the width and height as… |
| `func WriteOpticalFlow(path string, flow *FlowField) (err error)` | WriteOpticalFlow writes a FlowField to the named file in the Middlebury .flo format (see `WriteFlow`). |

### Types

| Type | What it is |
| --- | --- |
| `DeepFlowParams` | DeepFlowParams configures `DeepFlow`. |
| `DualTVL1OpticalFlow` | DualTVL1OpticalFlow is a reusable TV-L1 dense optical-flow estimator, the stdlib port of OpenCV's cv::optflow::DualTVL1OpticalFlow /… |
| `ErrorStats` | ErrorStats summarises a per-pixel error distribution. |
| `FlowField` | FlowField is a dense, two-channel float64 motion field: for every pixel it stores a horizontal (X / u) and vertical (Y / v) displacement measured… |
| `PCAFlowParams` | PCAFlowParams configures `CalcOpticalFlowPCAFlow`. |
| `PointF` | PointF is a sub-pixel 2-D point with float64 coordinates (column X, row Y). |
| `RLOFParams` | RLOFParams configures the Robust Local Optical Flow trackers (`CalcOpticalFlowSparseRLOF` and `CalcOpticalFlowDenseRLOF`). |
| `SimpleFlowParams` | SimpleFlowParams configures `CalcOpticalFlowSimpleFlow`. |
| `TVL1Params` | TVL1Params configures the duality-based TV-L1 optical-flow solver (`CalcOpticalFlowDenseTVL1` and `DualTVL1OpticalFlow`). |

<details>
<summary><code>DeepFlowParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultDeepFlowParams() DeepFlowParams` | DefaultDeepFlowParams returns a sensible DeepFlow configuration (Alpha=6, Beta=0.4, Scales=4, Warps=3, Iterations=25, MatchRadius=4, PatchRadius=3). |

</details>

<details>
<summary><code>DualTVL1OpticalFlow</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDualTVL1OpticalFlow(p TVL1Params) *DualTVL1OpticalFlow` | NewDualTVL1OpticalFlow returns a DualTVL1OpticalFlow configured with p. |
| `func (d *DualTVL1OpticalFlow) Calc(prev, next *cv.Mat) *FlowField` | Calc computes the dense TV-L1 flow from prev to next. |

</details>

<details>
<summary><code>ErrorStats</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func EndpointErrorStats(est, gt *FlowField) ErrorStats` | EndpointErrorStats computes summary statistics (mean, RMS, median and max) of the endpoint error between an estimated and a ground-truth flow in a… |

</details>

<details>
<summary><code>FlowField</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CalcOpticalFlowDIS(prev, next *cv.Mat, patchRadius, searchRadius, levels int) *FlowField` | CalcOpticalFlowDIS computes a dense optical-flow field from prev to next using a simplified Dense Inverse Search (DIS) scheme: coarse-to-fine patch… |
| `func CalcOpticalFlowDenseHS(prev, next *cv.Mat, alpha float64, iters int) *FlowField` | CalcOpticalFlowDenseHS computes a dense optical-flow field from prev to next using the classic Horn-Schunck variational method. |
| `func CalcOpticalFlowDenseRLOF(prev, next *cv.Mat, p RLOFParams) *FlowField` | CalcOpticalFlowDenseRLOF computes a dense flow field from prev to next with the dense RLOF pipeline of OpenCV's… |
| `func CalcOpticalFlowDenseTVL1(prev, next *cv.Mat, p TVL1Params) *FlowField` | CalcOpticalFlowDenseTVL1 computes a dense optical-flow field from prev to next with the duality-based TV-L1 method of Zach, Pock & Bischof (2007). |
| `func CalcOpticalFlowPCAFlow(prev, next *cv.Mat, p PCAFlowParams) *FlowField` | CalcOpticalFlowPCAFlow computes a dense flow field from prev to next with a PCAFlow-style estimator in the spirit of Wulff & Black, "Efficient… |
| `func CalcOpticalFlowSimpleFlow(prev, next *cv.Mat, p SimpleFlowParams) *FlowField` | CalcOpticalFlowSimpleFlow computes a dense flow field from prev to next with the SimpleFlow algorithm (Tao et al., 2012), the stdlib port of OpenCV's… |
| `func CalcOpticalFlowSparseToDense(prev, next *cv.Mat, sparsePts []image.Point) *FlowField` | CalcOpticalFlowSparseToDense estimates a dense flow field by first tracking a set of sparse seed points with local Lucas-Kanade and then… |
| `func DeepFlow(prev, next *cv.Mat, p DeepFlowParams) *FlowField` | DeepFlow computes a dense optical-flow field from prev to next with a lightweight reimplementation of the DeepFlow idea (Weinzaepfel et al., 2013): a… |
| `func InterpolateFlow(rows, cols int, points, vectors []PointF, sigma float64) *FlowField` | InterpolateFlow densifies a set of sparse flow samples into a full FlowField of size rows×cols using purely geometric (Gaussian / Shepard)… |
| `func InterpolateFlowGuided(guide *cv.Mat, points, vectors []PointF, sigmaS, sigmaC float64) *FlowField` | InterpolateFlowGuided densifies sparse flow samples with edge-aware weighting driven by a guide image. |
| `func NewFlowField(rows, cols int) *FlowField` | NewFlowField allocates a zero-filled FlowField. |
| `func ReadFlow(r io.Reader) (*FlowField, error)` | ReadFlow deserialises a Middlebury .flo stream written by `WriteFlow` (or any conforming tool) from r and returns the reconstructed FlowField. |
| `func ReadOpticalFlow(path string) (*FlowField, error)` | ReadOpticalFlow reads a Middlebury .flo file (see `ReadFlow`) from the named path and returns the reconstructed FlowField. |
| `func (f *FlowField) At(y, x int) (u, v float64)` | At returns the (u, v) displacement stored at row y, column x. |
| `func (f *FlowField) Clone() *FlowField` | Clone returns a deep copy of the field with its own backing storage. |
| `func (f *FlowField) Magnitude() []float64` | Magnitude returns the per-pixel flow magnitude sqrt(u²+v²) as a row-major slice of length Rows*Cols. |
| `func (f *FlowField) MaxMagnitude() float64` | MaxMagnitude returns the largest flow magnitude in the field, or zero for an empty field. |
| `func (f *FlowField) MeanFlow(border int) (u, v float64)` | MeanFlow returns the average (u, v) displacement over the interior of the field, excluding a border of the given width where estimates are least… |
| `func (f *FlowField) Set(y, x int, u, v float64)` | Set stores the (u, v) displacement at row y, column x. |

</details>

<details>
<summary><code>PCAFlowParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultPCAFlowParams() PCAFlowParams` | DefaultPCAFlowParams returns a sensible PCAFlow configuration (BasisOrder=4, GridStep=6, Ridge=1e-3, WinRadius=6). |

</details>

<details>
<summary><code>PointF</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CalcOpticalFlowSparseRLOF(prev, next *cv.Mat, prevPts []image.Point, p RLOFParams) (nextPts []PointF, status []bool)` | CalcOpticalFlowSparseRLOF tracks a set of points from prev to next with Robust Local Optical Flow (Senst et al.), the illumination-robust successor… |

</details>

<details>
<summary><code>RLOFParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultRLOFParams() RLOFParams` | DefaultRLOFParams returns a robust RLOF configuration (WinRadius=6, MaxIters=20, MinEigen=1e-3, HuberDelta=12, Illumination=true, GridStep=6). |

</details>

<details>
<summary><code>SimpleFlowParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultSimpleFlowParams() SimpleFlowParams` | DefaultSimpleFlowParams returns a sensible SimpleFlow configuration (Scales=3, SearchRadius=3, WinRadius=3, SigmaS=4.5, SigmaC=25). |

</details>

<details>
<summary><code>TVL1Params</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultTVL1Params() TVL1Params` | DefaultTVL1Params returns the standard TV-L1 configuration (Tau=0.25, Lambda=0.15, Theta=0.3, Scales=5, Warps=5, Iterations=30, Epsilon=0.01). |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/optflow).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
