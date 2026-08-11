# stereo — standard-library-only implementation of the classic stereo-correspondence and disparity

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/stereo.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/stereo)

Package stereo is a standard-library-only implementation of the classic
stereo-correspondence and disparity routines from OpenCV's calib3d and contrib
stereo modules, built on top of the root cv package
(github.com/malcolmston/opencv).

It depends only on the Go standard library and the root cv package: no cgo, no
third-party code, and it does not import any sibling cv/* subpackage. The root
package supplies the `github.com/malcolmston/opencv.Mat` container and
grayscale conversion via `github.com/malcolmston/opencv.CvtColor`; everything
specific to stereo matching is implemented here from scratch.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/stereo"
```

## Usage

This is the package's own `ExampleBlockMatcher`, so it compiles and its output is
asserted on every `go test ./stereo/`.

```go
left, right := shiftedPair(64, 24, 8)
	bm := stereo.BlockMatcher{
		NumDisparities: 16,
		BlockSize:      7,
		PreFilterType:  stereo.PrefilterXSobel,
		Disp12MaxDiff:  2,
	}
	d := bm.Compute(left, right)
	fmt.Println(d.Data[12*d.Cols+50])
```

```
8
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ApplyNormalizedPrefilter(m *cv.Mat, preFilterSize, preFilterCap int) *cv.Mat` | ApplyNormalizedPrefilter applies the normalized-response pre-filter: it subtracts the local mean over a preFilterSize×preFilterSize box window and… |
| `func ApplyXSobelPrefilter(m *cv.Mat, preFilterCap int) *cv.Mat` | ApplyXSobelPrefilter applies the horizontal Sobel pre-filter to a grayscale (or RGB, converted to gray) image. |
| `func CensusTransform(m *cv.Mat, windowW, windowH int) (codes []uint64, rows, cols int)` | CensusTransform computes the census transform of a grayscale (or RGB, converted to gray) image, mirroring the classic non-parametric local descriptor… |
| `func ComputeConfidence(vol *CostVolume) *cv.Mat` | ComputeConfidence estimates a per-pixel matching confidence from a cost volume using the peak-ratio (PKRN) measure: the ratio between the best and… |
| `func FilterSpecklesDisparity(disp *cv.Mat, newVal uint8, maxSpeckleSize, maxDiff int) *cv.Mat` | FilterSpecklesDisparity removes small speckles from a disparity map in place, mirroring OpenCV's cv::filterSpeckles. |
| `func HammingDistance64(a, b uint64) int` | HammingDistance64 returns the number of differing bits between a and b, i.e. |
| `func Rectify(left, right *cv.Mat) (rectLeft, rectRight *cv.Mat)` | Rectify is a passthrough stub for stereo rectification. |
| `func ReprojectImageTo3D(disparity *cv.Mat, Q [4][4]float64) [][3]float64` | ReprojectImageTo3D maps a disparity map to 3-D coordinates through the 4×4 reprojection matrix Q (as produced by stereo rectification). |
| `func SubpixelParabola(cLo, cMid, cHi int) float64` | SubpixelParabola fits a parabola through the three cost samples cLo = c(d-1), cMid = c(d) and cHi = c(d+1) at consecutive disparities and returns the… |
| `func ValidateDisparity(dispLeft, dispRight *cv.Mat, disp12MaxDiff int, newVal uint8) *cv.Mat` | ValidateDisparity enforces the left-right consistency constraint between a left-referenced and a right-referenced disparity map, mirroring OpenCV's… |

### Types

| Type | What it is |
| --- | --- |
| `BlockMatcher` | BlockMatcher is a full-featured local block matcher, the counterpart to OpenCV's cv::StereoBM. |
| `CostType` | CostType selects the per-pixel dissimilarity measure used when building a `CostVolume` with `MatchingCostVolume`. |
| `CostVolume` | CostVolume holds the dense matching cost of a rectified stereo pair: for every pixel (y, x) it stores the cost of assigning each candidate disparity… |
| `DisparityF` | DisparityF is a single-channel, row-major map of sub-pixel disparities held as float32 values. |
| `DisparityWLSFilter` | DisparityWLSFilter is an edge-aware weighted-least-squares post-filter for disparity maps, mirroring the role of ximgproc::DisparityWLSFilter. |
| `PrefilterType` | PrefilterType selects the pre-normalisation applied to the input images before block matching, matching OpenCV's StereoBM pre-filter modes. |
| `QuasiDenseStereo` | QuasiDenseStereo implements quasi-dense stereo matching by seed-and-grow propagation, mirroring cv::stereo::QuasiDenseStereo. |
| `Rect` | Rect is an axis-aligned integer rectangle, analogous to cv::Rect. |
| `SGMMode` | SGMMode selects the set of aggregation paths used by `StereoSGM`. |
| `StereoBM` | StereoBM computes a disparity map by local block matching, mirroring OpenCV's cv::StereoBM. |
| `StereoSGBM` | StereoSGBM computes a disparity map with semi-global-matching-lite, a reduced form of OpenCV's cv::StereoSGBM. |
| `StereoSGM` | StereoSGM is a full semi-global matcher. |

<details>
<summary><code>BlockMatcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (b BlockMatcher) Compute(left, right *cv.Mat) *cv.Mat` | Compute matches left against right and returns a single-channel 8-bit disparity map. |

</details>

<details>
<summary><code>CostVolume</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CensusCostVolume(left, right *cv.Mat, minDisparity, numDisparities, windowW, windowH, blockSize int) *CostVolume` | CensusCostVolume builds a `CostVolume` whose data term is the Hamming distance between census codes, a robust illumination-invariant matching cost. |
| `func MatchingCostVolume(left, right *cv.Mat, minDisparity, numDisparities, blockSize int, cost CostType) *CostVolume` | MatchingCostVolume builds a `CostVolume` for a rectified pair using the chosen `CostType`. |
| `func (v *CostVolume) At(y, x, d int) int32` | At returns the cost of disparity d at pixel (y, x). |
| `func (v *CostVolume) WinnerTakeAll() *cv.Mat` | WinnerTakeAll decodes the volume to an 8-bit disparity map by choosing, at each pixel, the disparity with the smallest cost. |

</details>

<details>
<summary><code>DisparityF</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDisparityF(rows, cols int) *DisparityF` | NewDisparityF allocates a DisparityF whose samples are all `InvalidDisparityF`. |
| `func RefineSubpixel(disp *cv.Mat, vol *CostVolume) *DisparityF` | RefineSubpixel refines an integer disparity map to sub-pixel precision using the aggregated (or data) costs in vol. |
| `func (d *DisparityF) At(y, x int) float32` | At returns the disparity at row y, column x. |
| `func (d *DisparityF) ToMat() *cv.Mat` | ToMat quantises the sub-pixel map to an 8-bit single-channel github.com/malcolmston/opencv.Mat. |

</details>

<details>
<summary><code>DisparityWLSFilter</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (f DisparityWLSFilter) Filter(disp, guide *cv.Mat) *cv.Mat` | Filter smooths and fills disp using guide as the edge reference. |
| `func (f DisparityWLSFilter) FilterWithConfidence(disp, conf, guide *cv.Mat) *cv.Mat` | FilterWithConfidence is like `DisparityWLSFilter.Filter` but takes an explicit per-pixel confidence map (as produced by `ComputeConfidence`);… |

</details>

<details>
<summary><code>QuasiDenseStereo</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (q QuasiDenseStereo) Process(left, right *cv.Mat) *cv.Mat` | Process matches left against right and returns a single-channel 8-bit quasi-dense disparity map; unmatched pixels hold `InvalidDisparity`. |

</details>

<details>
<summary><code>Rect</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func GetValidDisparityROI(roi1, roi2 Rect, minDisparity, numberOfDisparities, blockSize int) Rect` | GetValidDisparityROI computes the rectangle of a rectified image within which disparities can be estimated reliably, mirroring OpenCV's… |
| `func (r Rect) Area() int` | Area returns the number of pixels the rectangle covers (0 when empty). |
| `func (r Rect) Empty() bool` | Empty reports whether the rectangle covers no pixels. |

</details>

<details>
<summary><code>StereoBM</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (bm StereoBM) Compute(left, right *cv.Mat) *cv.Mat` | Compute matches left against right and returns a single-channel 8-bit disparity map the same size as the inputs. |

</details>

<details>
<summary><code>StereoSGBM</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (sg StereoSGBM) Compute(left, right *cv.Mat) *cv.Mat` | Compute matches left against right and returns a single-channel 8-bit disparity map the same size as the inputs. |

</details>

<details>
<summary><code>StereoSGM</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (s StereoSGM) Aggregate(left, right *cv.Mat) *CostVolume` | Aggregate builds the semi-global aggregated cost volume for a rectified pair. |
| `func (s StereoSGM) Compute(left, right *cv.Mat) *cv.Mat` | Compute matches left against right and returns a single-channel 8-bit disparity map. |
| `func (s StereoSGM) ComputeFloat(left, right *cv.Mat) *DisparityF` | ComputeFloat is like `StereoSGM.Compute` but returns a sub-pixel `DisparityF`. |

</details>

### Constants

`InvalidDisparity`, `InvalidDisparityF`, `NumPaths`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/stereo).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
