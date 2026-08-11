# imgprocx — provides extended geometric and frequency-domain image processing on top

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/imgprocx.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/imgprocx)

Package imgprocx provides extended geometric and frequency-domain image
processing on top of the standard-library-only OpenCV port
github.com/malcolmston/opencv (imported here as cv). It complements the root
package's geometric transforms (`cv.WarpAffine`, `cv.WarpPerspective`,
`cv.Resize`) and linear filters (`cv.Filter2D`, `cv.GaussianBlur`) with
routines OpenCV groups under imgproc's "geometric image transformations",
"structural analysis" and "motion analysis" sections.

`GetAffineTransform` solves the exact 2×3 affine transform that maps three
source points to three destinations, mirroring cv2.getAffineTransform. When
correspondences are noisy and contaminated by outliers, `EstimateAffine2D`
(six degrees of freedom) and `EstimateAffinePartial2D` (four degrees of
freedom: rotation, a single uniform scale and translation) fit a robust model
with a deterministic RANSAC loop followed by a least-squares refit over the
inliers. All three return a plain [2][3]float64 (or the compatible
`cv.AffineMatrix`); `ApplyAffine` evaluates such a matrix at a point and
`ToAffineMatrix`/`FromAffineMatrix` convert between the two representations so
the result can be handed straight to `cv.WarpAffine`.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/imgprocx"
```

## Usage

This is the package's own `ExampleEMD`, so it compiles and its output is
asserted on every `go test ./imgprocx/`.

```go
d := imgprocx.EMD(
		[]float64{0.4, 0.6},
		[]float64{0.5, 0.5},
		[][]float64{{0, 1}, {1, 0}},
	)
	fmt.Printf("%.1f\n", d)
```

```
0.1
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func Accumulate(src *cv.Mat, dst *cv.FloatMat)` | Accumulate adds the single-channel image src into the running sum dst, updating dst in place: dst`y``x` += src`y``x`. |
| `func AccumulateProduct(src1, src2 *cv.Mat, dst *cv.FloatMat)` | AccumulateProduct adds the element-wise product of src1 and src2 into dst in place: dst`y``x` += src1`y``x`·src2`y``x`. |
| `func AccumulateSquare(src *cv.Mat, dst *cv.FloatMat)` | AccumulateSquare adds the element-wise square of src into dst in place: dst`y``x` += src`y``x`². |
| `func AccumulateWeighted(src *cv.Mat, dst *cv.FloatMat, alpha float64)` | AccumulateWeighted blends src into dst as a running exponentially-weighted average, updating dst in place: |
| `func BlendLinear(src1, src2 *cv.Mat, weights1, weights2 *cv.FloatMat) *cv.Mat` | BlendLinear blends two images with per-pixel weights, mirroring cv2.blendLinear. |
| `func CornerEigenValsAndVecs(src *cv.Mat, blockSize, ksize int) [][]EigenValsVecs` | CornerEigenValsAndVecs computes, for every pixel of a single-channel image, the eigenvalues and eigenvectors of the local structure tensor, mirroring… |
| `func CornerMinEigenVal(src *cv.Mat, blockSize, ksize int) *cv.FloatMat` | CornerMinEigenVal computes the smaller eigenvalue of the local structure tensor at every pixel of a single-channel image, mirroring… |
| `func CreateHanningWindow(rows, cols int) *cv.FloatMat` | CreateHanningWindow returns a rows×cols separable Hann (raised-cosine) window as a `cv.FloatMat`, mirroring cv2.createHanningWindow. |
| `func DistanceTransformWithLabels(src *cv.Mat) (dist *cv.FloatMat, labels [][]int)` | DistanceTransformWithLabels computes, for every pixel of a binary single-channel image, both the approximate Euclidean distance to the nearest zero… |
| `func EMD(supply, demand []float64, cost [][]float64) float64` | EMD returns the Earth Mover's Distance between two weighted signatures, mirroring cv2.EMD. |
| `func EstimateAffine2D(src, dst []cv.Point) (m [2][3]float64, inliers []bool)` | EstimateAffine2D robustly fits the 2×3 affine transform (six degrees of freedom) mapping src to dst from a set of noisy, possibly… |
| `func EstimateAffinePartial2D(src, dst []cv.Point) (m [2][3]float64, inliers []bool)` | EstimateAffinePartial2D robustly fits a partial (similarity) 2×3 transform with four degrees of freedom — a rotation, a single uniform scale and a… |
| `func FloodFill(img *cv.Mat, seed cv.Point, newVal cv.Scalar, opts *FloodFillOptions) (area int, rect cv.Rect)` | FloodFill fills a connected region of img starting at seed, mirroring cv2.floodFill. |
| `func FromAffineMatrix(a cv.AffineMatrix) [2][3]float64` | FromAffineMatrix converts a `cv.AffineMatrix` into the 2×3 form used by the estimators in this package. |
| `func GetAffineTransform(src, dst [3]cv.Point) cv.AffineMatrix` | GetAffineTransform computes the 2×3 affine transform that maps the three source points src to the three destination points dst exactly, mirroring… |
| `func GetDerivKernels(dx, dy, ksize int, normalize bool) (kx, ky []float64)` | GetDerivKernels returns the pair of 1-D filter kernels (kx applied along the columns, ky along the rows) whose separable application yields the image… |
| `func GetGaborKernel(ksize int, sigma, theta, lambda, gamma, psi float64) cv.Kernel` | GetGaborKernel builds a Gabor filter kernel — a planar sinusoid modulated by a Gaussian envelope — of size ksize×ksize, mirroring… |
| `func GetGaussianKernel(ksize int, sigma float64) []float64` | GetGaussianKernel returns the ksize normalised 1-D Gaussian smoothing coefficients, mirroring cv2.getGaussianKernel. |
| `func IntegralImage(img *cv.Mat) (sum, sqSum [][]float64)` | IntegralImage computes the summed-area tables of img. |
| `func IntegralTilted(img *cv.Mat) [][]float64` | IntegralTilted computes the tilted (45°-rotated) summed-area table of img, mirroring the tilted output of cv::integral. |
| `func LinearPolar(src *cv.Mat, center Point2f, maxRadius float64, width, height int) *cv.Mat` | LinearPolar remaps src from Cartesian coordinates into a polar (radius, angle) grid about center, mirroring cv2.linearPolar's forward transform. |
| `func LogPolar(src *cv.Mat, center Point2f, maxRadius float64, width, height int) *cv.Mat` | LogPolar remaps src from Cartesian coordinates into a log-polar (log-radius, angle) grid about center, mirroring cv2.logPolar's forward transform. |
| `func PreCornerDetect(src *cv.Mat, ksize int) *cv.FloatMat` | PreCornerDetect computes the corner-feature map used to pre-select corner candidates, mirroring cv2.preCornerDetect. |
| `func RectSum(sum [][]float64, x0, y0, x1, y1 int) float64` | RectSum returns the total intensity over the half-open rectangle [x0, x1) × [y0, y1) from a summed-area table produced by `IntegralImage`, using the… |
| `func SpatialGradient(src *cv.Mat) (dx, dy *cv.FloatMat)` | SpatialGradient computes the first-order image derivatives of a single-channel image with a 3×3 Sobel operator, mirroring cv2.spatialGradient. |
| `func TiltedSum(tilted [][]float64, x, y int) float64` | TiltedSum returns tilted`y``x` from a table produced by `IntegralTilted`: the total intensity over the upward 45° triangle with apex just above… |
| `func ToAffineMatrix(m [2][3]float64) cv.AffineMatrix` | ToAffineMatrix converts a 2×3 matrix into the root package's `cv.AffineMatrix` (a flat [6]float64) so it can be used with `cv.WarpAffine`. |

### Types

| Type | What it is |
| --- | --- |
| `EigenValsVecs` | EigenValsVecs holds the eigen-decomposition of the 2×2 structure tensor at one pixel, as produced by `CornerEigenValsAndVecs`. |
| `FloodFillOptions` | FloodFillOptions configures `FloodFill`. |
| `HoughLine3` | HoughLine3 is one line found by `HoughLinesPointSet`: Votes point counts, and the line is in Hesse normal form x·cos(Theta) + y·sin(Theta) = Rho… |
| `Point2f` | Point2f is a sub-pixel image coordinate (x is the column, y is the row). |

<details>
<summary><code>HoughLine3</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func HoughLinesPointSet(points []cv.Point, linesMax, threshold int, minRho, maxRho, rhoStep, minTheta, maxTheta, thetaStep float64) []HoughLine3` | HoughLinesPointSet finds the lines best supported by an arbitrary set of 2-D points using the standard Hough transform, mirroring… |

</details>

<details>
<summary><code>Point2f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ApplyAffine(m [2][3]float64, p cv.Point) Point2f` | ApplyAffine maps the point p through the 2×3 affine matrix m (stored row-major, so m[0] holds the coefficients of the output x-coordinate) and… |
| `func CornerSubPix(img *cv.Mat, corners []cv.Point, winSize int) []Point2f` | CornerSubPix refines integer corner locations to sub-pixel accuracy, mirroring cv2.cornerSubPix. |
| `func PhaseCorrelate(a, b *cv.Mat) (shift Point2f, response float64)` | PhaseCorrelate estimates the translational shift between two same-sized images a and b from the phase of their cross-power spectrum, mirroring… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/imgprocx).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
