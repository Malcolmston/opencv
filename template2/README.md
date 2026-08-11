# template2 — implements classic template matching on top of the parent package's `cv.Mat` image type

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/template2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/template2)

Package template2 implements classic template matching on top of the parent
package's `cv.Mat` image type.

The package is a pure-Go, standard-library-only companion to the core opencv
module. It never defines its own image type: every routine that consumes an
image accepts a `cv.Mat`, every routine that produces a dense score map
returns a `cv.FloatMat`, and match locations are reported with the parent
package's coordinate conventions (x is the column, y is the row, origin
top-left). Results therefore interoperate directly with the rest of the
library — drawing, I/O, `cv.MinMaxLoc`, and so on.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/template2"
```

## Usage

This is the package's own `ExampleBestMatch`, so it compiles and its output is
asserted on every `go test ./template2/`.

```go
src := scene(40, 60, 22, 11, 8)

	templ := patch(src, 20, 9, 12, 12)

	m, err := template2.BestMatch(src, templ, template2.MethodZNCC)
	if err != nil {
		panic(err)
	}
	fmt.Printf("found %dx%d template at (%d,%d), score %.3f\n", m.Width, m.Height, m.X, m.Y, m.Score)
```

```
found 12x12 template at (20,9), score 1.000
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func BuildAngles(minAngle, maxAngle, step float64) []float64` | BuildAngles returns the rotation angles (in degrees) sampled from minAngle to maxAngle inclusive with the given step. |
| `func BuildScales(minScale, maxScale float64, levels int) []float64` | BuildScales returns the Levels scale factors sampled uniformly from minScale to maxScale inclusive. |
| `func FastNCC(src, templ *cv.Mat) (*cv.FloatMat, error)` | FastNCC computes the (non-zero-mean) normalised cross-correlation score map using an integral image for each window's energy. |
| `func FastZNCC(src, templ *cv.Mat) (*cv.FloatMat, error)` | FastZNCC computes the zero-mean normalised cross-correlation score map using integral images to obtain each window's mean and energy in constant time. |
| `func LocateExtremum(scores *cv.FloatMat, higherIsBetter bool) (x, y int, value float64, ok bool)` | LocateExtremum scans a score map and returns the location and value of its best entry: the maximum when higherIsBetter is true, otherwise the minimum. |
| `func MatchCorrCoeff(src, templ *cv.Mat) (*cv.FloatMat, error)` | MatchCorrCoeff returns the mean-subtracted correlation-coefficient score map. |
| `func MatchCrossCorr(src, templ *cv.Mat) (*cv.FloatMat, error)` | MatchCrossCorr returns the raw cross-correlation score map. |
| `func MatchNCC(src, templ *cv.Mat) (*cv.FloatMat, error)` | MatchNCC returns the normalised cross-correlation score map. |
| `func MatchSAD(src, templ *cv.Mat) (*cv.FloatMat, error)` | MatchSAD returns the sum-of-absolute-differences score map. |
| `func MatchSSD(src, templ *cv.Mat) (*cv.FloatMat, error)` | MatchSSD returns the sum-of-squared-differences score map. |
| `func MatchSSDNormed(src, templ *cv.Mat) (*cv.FloatMat, error)` | MatchSSDNormed returns the normalised sum-of-squared-differences score map. |
| `func MatchTemplate(src, templ *cv.Mat, method Method) (*cv.FloatMat, error)` | MatchTemplate slides templ over src and returns a dense score map of shape (src.Rows-templ.Rows+1) × (src.Cols-templ.Cols+1). |
| `func MatchZNCC(src, templ *cv.Mat) (*cv.FloatMat, error)` | MatchZNCC returns the zero-mean normalised cross-correlation score map. |
| `func ParabolicPeak(left, center, right float64) float64` | ParabolicPeak fits a parabola through three equally spaced samples (left, center, right) and returns the offset, in the range (-0.5, 0.5], of the… |
| `func RotateTemplate(templ *cv.Mat, angleDeg float64) *cv.Mat` | RotateTemplate rotates templ counter-clockwise by angleDeg degrees about its centre and returns a new `cv.Mat` whose canvas is expanded so the whole… |
| `func SortByScore(matches []Match, higherIsBetter bool)` | SortByScore orders matches best-first in place: descending score when higherIsBetter is true, ascending otherwise. |
| `func ToGrayscale(src *cv.Mat) *cv.Mat` | ToGrayscale returns a single-channel copy of src. |

### Types

| Type | What it is |
| --- | --- |
| `Integral` | Integral holds the summed-area tables of a grayscale image: the running sum of samples and of squared samples. |
| `Match` | Match describes one detected template location. |
| `Method` | Method selects the similarity measure used by `MatchTemplate` and the other matching routines. |
| `MultiScaleParams` | MultiScaleParams configures `MatchMultiScale`. |
| `Pyramid` | Pyramid is a sequence of successively downscaled copies of an image, together with the scale factor of each level relative to the original. |
| `RotationParams` | RotationParams configures `MatchRotationInvariant`. |
| `SubPixel` | SubPixel is a peak location refined to fractional pixel precision, together with the interpolated score at that location. |

<details>
<summary><code>Integral</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewIntegral(src *cv.Mat) *Integral` | NewIntegral computes the summed-area tables of src. |
| `func (in *Integral) RegionMean(x0, y0, x1, y1 int) float64` | RegionMean returns the mean sample value over the rectangle spanning columns [x0,x1) and rows [y0,y1). |
| `func (in *Integral) RegionSqSum(x0, y0, x1, y1 int) float64` | RegionSqSum returns the sum of squared sample values over the rectangle spanning columns [x0,x1) and rows [y0,y1). |
| `func (in *Integral) RegionSum(x0, y0, x1, y1 int) float64` | RegionSum returns the sum of sample values over the rectangle spanning columns [x0,x1) and rows [y0,y1). |
| `func (in *Integral) RegionVariance(x0, y0, x1, y1 int) float64` | RegionVariance returns the population variance of sample values over the rectangle spanning columns [x0,x1) and rows [y0,y1). |

</details>

<details>
<summary><code>Match</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BestMatch(src, templ *cv.Mat, method Method) (Match, error)` | BestMatch computes the score map for the given method and returns the single strongest `Match`. |
| `func BestMatchMultiScale(src, templ *cv.Mat, method Method) (Match, error)` | BestMatchMultiScale returns the single strongest detection of templ in src across scales 0.5 to 1.5 (11 steps) under the given method. |
| `func BestMatchRotated(src, templ *cv.Mat, method Method) (Match, error)` | BestMatchRotated returns the single strongest detection of templ in src across a full 0..350 degree sweep in 10 degree steps under the given method. |
| `func ExtractMatches(scores *cv.FloatMat, width, height int, threshold float64, higherIsBetter bool) []Match` | ExtractMatches converts a score map into the list of positions that pass the threshold. |
| `func FilterByScore(matches []Match, threshold float64, higherIsBetter bool) []Match` | FilterByScore returns the subset of matches whose score passes threshold: at least threshold when higherIsBetter is true, at most threshold otherwise. |
| `func FindMatches(src, templ *cv.Mat, method Method, threshold float64) ([]Match, error)` | FindMatches computes the score map for the given method and returns every location whose score passes threshold, sorted best-first. |
| `func MatchMultiScale(src, templ *cv.Mat, params MultiScaleParams) ([]Match, error)` | MatchMultiScale searches for templ in src across a range of template scales and returns the pooled detections that pass the score threshold,… |
| `func MatchRotationInvariant(src, templ *cv.Mat, params RotationParams) ([]Match, error)` | MatchRotationInvariant searches for templ in src across a range of template rotations and returns the pooled detections that pass the score… |
| `func NonMaxSuppression(matches []Match, iouThreshold float64, higherIsBetter bool) []Match` | NonMaxSuppression greedily prunes overlapping detections. |
| `func NonMaxSuppressionDistance(matches []Match, minCenterDistance float64, higherIsBetter bool) []Match` | NonMaxSuppressionDistance greedily prunes detections whose centres lie close together. |
| `func RefineMatch(scores *cv.FloatMat, m Match) Match` | RefineMatch returns a copy of m whose top-left corner is nudged by the sub-pixel offset of the score-map peak at (m.X, m.Y). |
| `func (m Match) Area() int` | Area returns the number of pixels covered by the matched patch. |
| `func (m Match) Center() cv.Point` | Center returns the integer-rounded pixel at the centre of the matched patch. |
| `func (m Match) CenterX() float64` | CenterX returns the (possibly fractional) column of the patch centre. |
| `func (m Match) CenterY() float64` | CenterY returns the (possibly fractional) row of the patch centre. |
| `func (m Match) IoU(o Match) float64` | IoU returns the intersection-over-union overlap of two matched rectangles, a value in [0,1]. |
| `func (m Match) Overlaps(o Match, iouThreshold float64) bool` | Overlaps reports whether the intersection-over-union overlap of m and o is at least iouThreshold. |
| `func (m Match) Rect() image.Rectangle` | Rect returns the matched patch as a standard-library rectangle spanning [X, X+Width) × [Y, Y+Height). |

</details>

<details>
<summary><code>Method</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (m Method) HigherIsBetter() bool` | HigherIsBetter reports whether larger score values indicate stronger matches for this method. |
| `func (m Method) String() string` | String returns a short human-readable name for the method. |
| `func (m Method) Valid() bool` | Valid reports whether m names a known similarity measure. |

</details>

<details>
<summary><code>MultiScaleParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultMultiScaleParams() MultiScaleParams` | DefaultMultiScaleParams returns sensible defaults: zero-mean normalised cross-correlation over scales 0.5 to 1.5 in 11 steps, a 0.7 score threshold… |

</details>

<details>
<summary><code>Pyramid</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BuildPyramid(src *cv.Mat, levels int, scaleFactor float64) *Pyramid` | BuildPyramid builds a Gaussian-free image pyramid of the given number of levels. |
| `func (p *Pyramid) Level(i int) *cv.Mat` | Level returns the image at pyramid level i. |
| `func (p *Pyramid) NumLevels() int` | NumLevels returns the number of levels in the pyramid. |

</details>

<details>
<summary><code>RotationParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultRotationParams() RotationParams` | DefaultRotationParams returns sensible defaults: zero-mean normalised cross-correlation over a full 0..350 degree sweep in 10 degree steps, a 0.7… |

</details>

<details>
<summary><code>SubPixel</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func RefinePeak(scores *cv.FloatMat, x, y int) SubPixel` | RefinePeak refines the integer peak at column x, row y of a score map to sub-pixel precision by fitting independent parabolas along the horizontal… |
| `func RefinePeakQuadratic(scores *cv.FloatMat, x, y int) SubPixel` | RefinePeakQuadratic refines the integer peak at column x, row y using the full 3x3 neighbourhood, fitting a bivariate quadratic surface and returning… |

</details>

### Variables

`ErrEmptyImage`, `ErrTemplateLarger`, `ErrChannelMismatch`, `ErrInvalidMethod`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/template2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
