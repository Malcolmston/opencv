# phase_unwrapping — implements two-dimensional phase unwrapping on top of the root cv package

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/phase_unwrapping.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/phase_unwrapping)

Package phase_unwrapping implements two-dimensional phase unwrapping on top of
the root cv package, mirroring a useful subset of OpenCV's phase_unwrapping
contrib module (whose principal class is HistogramPhaseUnwrapping).

Phase unwrapping is the problem of recovering a continuous ("absolute") phase
surface phi from a measured phase map that is only known modulo 2*pi, i.e.
wrapped into the principal interval (-pi, pi]. Wrapping is the operation

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/phase_unwrapping"
```

## Usage

This is the package's own `ExampleGoldsteinBranchCut`, so it compiles and its output is
asserted on every `go test ./phase_unwrapping/`.

```go
const n = 16
	spiral := make([][]float64, n)
	for i := 0; i < n; i++ {
		spiral[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			spiral[i][j] = pu.Wrap(math.Atan2(float64(i)-8, float64(j)-8))
		}
	}
	unwrapped, err := pu.GoldsteinBranchCut(spiral, 0)
	if err != nil {
		panic(err)
	}

	rw := pu.Rewrap(unwrapped)
	maxErr := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			e := math.Abs(pu.Wrap(rw[i][j] - spiral[i][j]))
			if e > maxErr {
				maxErr = e
			}
		}
	}
	fmt.Printf("residues=%v congruent=%v\n", pu.CountResidues(spiral) > 0, maxErr < 1e-9)
```

```
residues=true congruent=true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func BranchCuts(wrapped [][]float64, maxBoxRadius int) [][]bool` | BranchCuts computes the Goldstein-style branch-cut mask of a wrapped phase map without unwrapping it, exposing exactly the barrier that… |
| `func Congruence(estimate, wrapped [][]float64) ([][]float64, error)` | Congruence projects a real-valued phase estimate onto the surface that is closest to it while still being congruent to a wrapped map, i.e. |
| `func CountResidues(wrapped [][]float64) int` | CountResidues returns the number of non-zero residues reported by `Residues` for a wrapped phase map, i.e. |
| `func FlynnMinimumDiscontinuity(wrapped, weights [][]float64, maxIter int) ([][]float64, error)` | FlynnMinimumDiscontinuity unwraps a wrapped phase map by minimising the total weighted 2*pi discontinuity of the result, in the spirit of Flynn's… |
| `func GoldsteinBranchCut(wrapped [][]float64, maxBoxRadius int) ([][]float64, error)` | GoldsteinBranchCut unwraps a wrapped phase map with a Goldstein-style branch-cut method (Goldstein, Zebker and Werner, 1988). |
| `func LeastSquaresUnwrap(wrapped [][]float64) ([][]float64, error)` | LeastSquaresUnwrap performs unweighted least-squares (minimum-norm) phase unwrapping following Ghiglia and Pritt, "Two-Dimensional Phase Unwrapping:… |
| `func LeastSquaresUnwrapMat(wrapped *cv.FloatMat) (*cv.FloatMat, error)` | LeastSquaresUnwrapMat is the `cv.FloatMat` counterpart of `LeastSquaresUnwrap`. |
| `func MaskedUnwrap(wrapped [][]float64, mask [][]bool, quality [][]float64) ([][]float64, error)` | MaskedUnwrap unwraps only the pixels selected by mask (mask`i``j` true), using the same quality-guided path following as `QualityGuidedUnwrap` but… |
| `func MaximumPhaseGradient(wrapped [][]float64) [][]float64` | MaximumPhaseGradient returns the maximum-phase-gradient quality map: for each pixel, the largest absolute wrapped derivative (in x or y) found in its… |
| `func PhaseDerivativeVariance(wrapped [][]float64) [][]float64` | PhaseDerivativeVariance returns the phase-derivative-variance quality map of a wrapped phase map (Ghiglia and Pritt). |
| `func PseudoCorrelation(wrapped [][]float64) [][]float64` | PseudoCorrelation returns the pseudocorrelation quality map (Ghiglia and Pritt): for each pixel, over a 3x3 window, |
| `func QualityGuidedUnwrap(wrapped, quality [][]float64) ([][]float64, error)` | QualityGuidedUnwrap unwraps a wrapped phase map by following a quality-guided path (Ghiglia and Pritt): pixels are unwrapped most-reliable first, so… |
| `func ReliabilityMap(wrapped [][]float64) [][]float64` | ReliabilityMap returns the inverse-reliability image of a wrapped phase map: for each pixel, the sum of squared second differences of the wrapped… |
| `func ReliabilityMapMat(wrapped *cv.FloatMat) (*cv.FloatMat, error)` | ReliabilityMapMat is the FloatMat-valued counterpart of `ReliabilityMap`. |
| `func Residues(wrapped [][]float64) [][]int` | Residues detects phase residues using the classic Goldstein loop-integration test. |
| `func Rewrap(unwrapped [][]float64) [][]float64` | Rewrap re-wraps an unwrapped surface back into (-pi, pi], returning a new grid of the same shape. |
| `func TemporalUnwrap(wrapped [][][]float64, scales []float64) ([][]float64, error)` | TemporalUnwrap performs temporal (multi-wavelength / multi-frequency) phase unwrapping. |
| `func TotalDiscontinuity(unwrapped [][]float64) int` | TotalDiscontinuity counts the 2*pi discontinuities remaining in an unwrapped surface: the sum over all 4-connected edges of… |
| `func WeightedLeastSquaresUnwrap(wrapped, weights [][]float64, maxIter int, tol float64) ([][]float64, error)` | WeightedLeastSquaresUnwrap performs weighted least-squares phase unwrapping (Ghiglia and Pritt) by minimising the weighted L2 gradient mismatch |
| `func Wrap(x float64) float64` | Wrap maps an arbitrary phase value into the principal interval (-pi, pi]. |
| `func WrapPhaseMap(continuous [][]float64) [][]float64` | WrapPhaseMap wraps every value of a continuous `row``col` surface into (-pi, pi], returning a new grid of the same shape. |
| `func WrapToPi(x float64) float64` | WrapToPi maps an arbitrary phase value into the closed interval [-pi, pi]. |

### Types

| Type | What it is |
| --- | --- |
| `FlynnUnwrapper` | FlynnUnwrapper adapts `FlynnMinimumDiscontinuity` to the `Unwrapper` interface. |
| `GoldsteinUnwrapper` | GoldsteinUnwrapper adapts `GoldsteinBranchCut` to the `Unwrapper` interface. |
| `HistogramPhaseUnwrapping` | HistogramPhaseUnwrapping is a quality-guided, path-following 2-D phase unwrapper. |
| `HistogramUnwrapper` | HistogramUnwrapper adapts the quality-guided `HistogramPhaseUnwrapping` to the `Unwrapper` interface. |
| `LeastSquaresUnwrapper` | LeastSquaresUnwrapper adapts `LeastSquaresUnwrap` (unweighted DCT Poisson) to the `Unwrapper` interface. |
| `Params` | Params configures a `HistogramPhaseUnwrapping`. |
| `QualityGuidedUnwrapper` | QualityGuidedUnwrapper adapts `QualityGuidedUnwrap` to the `Unwrapper` interface. |
| `Residue` | Residue is a detected phase residue: the loop with top-left corner (Row, Col) carries the integer Charge (+1 or -1, occasionally larger in magnitude)… |
| `Unwrapper` | Unwrapper is the common interface implemented by every 2-D phase-unwrapping method in this package. |
| `WeightedLeastSquaresUnwrapper` | WeightedLeastSquaresUnwrapper adapts `WeightedLeastSquaresUnwrap` (weighted least squares by PCG) to the `Unwrapper` interface. |

<details>
<summary><code>FlynnUnwrapper</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (f FlynnUnwrapper) Unwrap(wrapped [][]float64) ([][]float64, error)` | Unwrap implements `Unwrapper`. |

</details>

<details>
<summary><code>GoldsteinUnwrapper</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (g GoldsteinUnwrapper) Unwrap(wrapped [][]float64) ([][]float64, error)` | Unwrap implements `Unwrapper`. |

</details>

<details>
<summary><code>HistogramPhaseUnwrapping</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewHistogramPhaseUnwrapping(p Params) *HistogramPhaseUnwrapping` | NewHistogramPhaseUnwrapping returns an unwrapper configured by p. |
| `func (h *HistogramPhaseUnwrapping) GetInverseReliabilityMap() (*cv.FloatMat, error)` | GetInverseReliabilityMap returns the per-pixel inverse-reliability map computed during the most recent unwrap, as a single-channel FloatMat. |
| `func (h *HistogramPhaseUnwrapping) Params() Params` | Params returns the (possibly normalised) parameters in effect. |
| `func (h *HistogramPhaseUnwrapping) UnwrapPhaseMap(wrapped *cv.FloatMat) (*cv.FloatMat, error)` | UnwrapPhaseMap unwraps a wrapped phase map stored as a single-channel github.com/malcolmston/opencv.FloatMat and returns a new FloatMat holding the… |
| `func (h *HistogramPhaseUnwrapping) UnwrapPhaseMapGrid(wrapped [][]float64) ([][]float64, error)` | UnwrapPhaseMapGrid unwraps a wrapped phase map given as a `row``col` grid and returns the unwrapped absolute phase as a new grid of the same shape. |

</details>

<details>
<summary><code>HistogramUnwrapper</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (h HistogramUnwrapper) Unwrap(wrapped [][]float64) ([][]float64, error)` | Unwrap implements `Unwrapper`. |

</details>

<details>
<summary><code>LeastSquaresUnwrapper</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (LeastSquaresUnwrapper) Unwrap(wrapped [][]float64) ([][]float64, error)` | Unwrap implements `Unwrapper`. |

</details>

<details>
<summary><code>Params</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultParams(width, height int) Params` | DefaultParams returns Params for a map of the given size using the same default histogram configuration as OpenCV: HistogramThresh = 3*pi*pi,… |

</details>

<details>
<summary><code>QualityGuidedUnwrapper</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (q QualityGuidedUnwrapper) Unwrap(wrapped [][]float64) ([][]float64, error)` | Unwrap implements `Unwrapper`. |

</details>

<details>
<summary><code>Residue</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ResidueList(wrapped [][]float64) []Residue` | ResidueList returns every non-zero residue of a wrapped phase map as an explicit list in row-major order, complementing the dense grid produced by… |

</details>

<details>
<summary><code>WeightedLeastSquaresUnwrapper</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (w WeightedLeastSquaresUnwrapper) Unwrap(wrapped [][]float64) ([][]float64, error)` | Unwrap implements `Unwrapper`. |

</details>

### Variables

`ErrEmptyInput`, `ErrShapeMismatch`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/phase_unwrapping).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
