# edges2 — standard-library-only toolkit of edge

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/edges2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/edges2)

Package edges2 is a standard-library-only toolkit of edge, line and circle
detection built on top of the parent package's
`github.com/malcolmston/opencv.Mat` image type.

- Gradient operators: Sobel, Scharr, Prewitt and Roberts cross, each returning
a signed `GradientField` from which magnitude and orientation are derived. -
Edge detectors: the full Canny pipeline (Gaussian smoothing, gradient,
non-maximum suppression and double-threshold hysteresis) exposed both as a
one-shot `Canny` and as its composable stages, plus the Marr-Hildreth
Laplacian-of-Gaussian zero-crossing detector and a model-free multi-scale
`StructuredEdges` strength map. - Line and circle fitting: the standard and
probabilistic Hough transforms (`HoughLines`, `HoughLinesP`), the gradient
Hough circle transform (`HoughCircles`) and a simplified region-growing
line-segment detector (`LSD`). - Post-processing: 8-connected `LinkEdges`
chaining, gradient orientation histograms and a dense
histogram-of-oriented-gradients (`HOG`) descriptor.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/edges2"
```

## Usage

This is the package's own `ExampleCanny`, so it compiles and its output is
asserted on every `go test ./edges2/`.

```go
edges := edges2.Canny(stripe(16, 16, 8), 50, 150, 1.0)

	count := 0
	for _, v := range edges.Data {
		if v == 255 {
			count++
		}
	}
	fmt.Println("edge pixels found:", count > 0)
```

```
edge pixels found: true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func Canny(src *cv.Mat, lowThresh, highThresh, sigma float64) *cv.Mat` | Canny runs the complete Canny edge-detection pipeline on a single-channel image and returns a binary edge map (edges 255, background 0). |
| `func CannyAuto(src *cv.Mat, scale, sigma float64) *cv.Mat` | CannyAuto runs `Canny` with thresholds derived automatically from the median gradient magnitude of the image: low = (1-scale)*median and high =… |
| `func CannyField(f *GradientField, lowThresh, highThresh float64) *cv.Mat` | CannyField runs the last two Canny stages — non-maximum suppression and double-threshold hysteresis — on a precomputed `GradientField` and… |
| `func DoubleThreshold(mag *FloatGrid, lowThresh, highThresh float64) *cv.Mat` | DoubleThreshold classifies the samples of a (typically non-maximum suppressed) magnitude grid into three levels and returns them as an 8-bit… |
| `func DrawCircles(img *cv.Mat, circles []Circle, value uint8) *cv.Mat` | DrawCircles rasterises the outline of every circle into a copy of img with the given intensity value (using the midpoint circle algorithm) and… |
| `func DrawLines(img *cv.Mat, lines []Line, value uint8) *cv.Mat` | DrawLines clips each infinite line to the image, rasterises the visible portion into a copy of img with the given intensity value and returns the… |
| `func DrawSegments(img *cv.Mat, segs []Segment, value uint8) *cv.Mat` | DrawSegments rasterises every segment into a copy of img with the given intensity value and returns the result, leaving the input untouched. |
| `func GradientMagnitude(src *cv.Mat) *cv.Mat` | GradientMagnitude is a convenience wrapper that computes the Sobel gradient of src and returns its magnitude as a display-ready 8-bit `cv.Mat`. |
| `func HOG(src *cv.Mat, opts HOGOptions) []float64` | HOG computes a dense histogram-of-oriented-gradients descriptor of a single-channel image. |
| `func Hysteresis(mag *FloatGrid, lowThresh, highThresh float64) *cv.Mat` | Hysteresis performs Canny double-threshold edge tracking on a magnitude grid and returns a binary edge map (255 for edges, 0 otherwise). |
| `func MarrHildreth(src *cv.Mat, sigma, thresh float64) *cv.Mat` | MarrHildreth runs the Marr-Hildreth edge detector: it computes the Laplacian of Gaussian of src at the given sigma and returns the zero crossings of… |
| `func OrientationHistogram(f *GradientField, bins int, signed bool) []float64` | OrientationHistogram builds a magnitude-weighted histogram of gradient orientations from a `GradientField`. |
| `func StructuredEdges(src *cv.Mat, sigma float64) *cv.Mat` | StructuredEdges computes a structured edge-strength map by combining Sobel gradient magnitude across several Gaussian scales (sigma, 2*sigma,… |
| `func ZeroCrossings(g *FloatGrid, thresh float64) *cv.Mat` | ZeroCrossings detects sign changes in a signed response grid (such as a LoG output) and returns a binary edge map (255 at a crossing, 0 otherwise). |

### Types

| Type | What it is |
| --- | --- |
| `Circle` | Circle is a circle with centre (X,Y) and the given Radius, as returned by `HoughCircles`. |
| `EdgeChain` | EdgeChain is an ordered list of connected edge pixels produced by `LinkEdges`. |
| `FloatGrid` | FloatGrid is a dense, single-channel raster of float64 samples in row-major order. |
| `GradientField` | GradientField holds the signed horizontal (Gx) and vertical (Gy) first derivatives of an image, as produced by `Sobel`, `Scharr`, `Prewitt` and… |
| `HOGOptions` | HOGOptions configures the `HOG` descriptor. |
| `LSDOptions` | LSDOptions configures the `LSD` line-segment detector. |
| `Line` | Line is a straight line in the normal (Hesse) parametrisation x*cos(Theta) + y*sin(Theta) = Rho, as returned by `HoughLines`. |
| `Point` | Point is an integer pixel coordinate with column X and row Y. |
| `Segment` | Segment is a finite line segment with floating-point endpoints (X1,Y1)-(X2,Y2), as returned by `HoughLinesP` and `LSD`. |

<details>
<summary><code>Circle</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func HoughCircles(src *cv.Mat, minRadius, maxRadius, threshold int, cannyLow, cannyHigh, minDist float64) []Circle` | HoughCircles detects circles in a single-channel image with the gradient Hough transform. |

</details>

<details>
<summary><code>EdgeChain</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func LinkEdges(edges *cv.Mat, minLength int) []EdgeChain` | LinkEdges groups the foreground pixels of a binary edge image into connected `EdgeChain` chains using 8-connectivity. |
| `func (c EdgeChain) BoundingBox() (min, max Point)` | BoundingBox returns the inclusive axis-aligned bounding box of the chain as its top-left and bottom-right corners. |
| `func (c EdgeChain) Length() int` | Length returns the number of pixels in the chain. |

</details>

<details>
<summary><code>FloatGrid</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DifferenceOfGaussians(src *cv.Mat, sigma1, sigma2 float64) *FloatGrid` | DifferenceOfGaussians returns the difference of two Gaussian-smoothed copies of src (blur at sigma1 minus blur at sigma2) as a signed `FloatGrid`. |
| `func HoughLineAccumulator(edges *cv.Mat, rhoRes, thetaRes float64) *FloatGrid` | HoughLineAccumulator computes the raw (rho, theta) Hough voting accumulator of a binary edge image and returns it as a `FloatGrid` with one row per… |
| `func Laplacian(src *cv.Mat) *FloatGrid` | Laplacian computes the discrete Laplacian (second derivative) of the image with the 4-neighbour 3×3 kernel and returns the signed response as a… |
| `func LaplacianOfGaussian(src *cv.Mat, sigma float64) *FloatGrid` | LaplacianOfGaussian convolves src with a Laplacian-of-Gaussian (LoG, also known as the Mexican-hat) kernel of the given standard deviation and… |
| `func MatToFloatGrid(m *cv.Mat) *FloatGrid` | MatToFloatGrid converts a single-channel `cv.Mat` to a FloatGrid, copying its samples as float64 values. |
| `func NewFloatGrid(rows, cols int) *FloatGrid` | NewFloatGrid allocates a zero-filled FloatGrid with the given dimensions. |
| `func NonMaximumSuppression(f *GradientField) *FloatGrid` | NonMaximumSuppression thins the gradient magnitude of f by keeping only the pixels that are local maxima along the gradient direction, returning the… |
| `func (g *FloatGrid) Abs() *FloatGrid` | Abs returns a new grid holding the absolute value of every sample. |
| `func (g *FloatGrid) At(y, x int) float64` | At returns the sample at row y, column x. |
| `func (g *FloatGrid) Clone() *FloatGrid` | Clone returns an independent deep copy of the grid. |
| `func (g *FloatGrid) MinMax() (min, max float64)` | MinMax returns the smallest and largest samples in the grid. |
| `func (g *FloatGrid) Set(y, x int, value float64)` | Set stores value at row y, column x. |
| `func (g *FloatGrid) Threshold(t float64) *cv.Mat` | Threshold returns a binary `cv.Mat` in which every sample strictly greater than t becomes foreground (255) and every other sample background (0). |
| `func (g *FloatGrid) ToMat() *cv.Mat` | ToMat converts the grid to an 8-bit `cv.Mat` by rounding each sample and clamping it to the [0,255] range (values are not rescaled). |
| `func (g *FloatGrid) ToMatNormalized() *cv.Mat` | ToMatNormalized converts the grid to an 8-bit `cv.Mat`, linearly rescaling the full value range [min,max] onto [0,255]. |

</details>

<details>
<summary><code>GradientField</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Prewitt(src *cv.Mat) *GradientField` | Prewitt computes the image gradient with the 3×3 Prewitt operator (uniform smoothing weights) and returns it as a `GradientField`. |
| `func Roberts(src *cv.Mat) *GradientField` | Roberts computes the image gradient with the 2×2 Roberts cross operator and returns it as a `GradientField`. |
| `func Scharr(src *cv.Mat) *GradientField` | Scharr computes the image gradient with the 3×3 Scharr operator, a variant of Sobel with better rotational symmetry, and returns it as a… |
| `func Sobel(src *cv.Mat) *GradientField` | Sobel computes the image gradient with the 3×3 Sobel operator and returns it as a `GradientField`. |
| `func (f *GradientField) At(y, x int) (gx, gy float64)` | At returns the gradient components (gx, gy) at row y, column x. |
| `func (f *GradientField) Magnitude() *FloatGrid` | Magnitude returns the Euclidean gradient magnitude sqrt(Gx^2+Gy^2) as a `FloatGrid`. |
| `func (f *GradientField) MagnitudeMat() *cv.Mat` | MagnitudeMat returns the gradient magnitude as an 8-bit `cv.Mat`, rescaling the response range onto [0,255] for display. |
| `func (f *GradientField) Orientation() *FloatGrid` | Orientation returns the gradient direction atan2(Gy,Gx) in radians in the range (-pi, pi] as a `FloatGrid`. |
| `func (f *GradientField) OrientationDegrees() *FloatGrid` | OrientationDegrees returns the gradient direction in degrees in the range [0,360) as a `FloatGrid`. |

</details>

<details>
<summary><code>HOGOptions</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultHOGOptions() HOGOptions` | DefaultHOGOptions returns the conventional HOG parameters: 8×8-pixel cells, 2×2-cell blocks, nine unsigned orientation bins. |

</details>

<details>
<summary><code>LSDOptions</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultLSDOptions() LSDOptions` | DefaultLSDOptions returns sensible defaults for `LSD`: light smoothing, a 22.5° orientation tolerance, an automatic gradient threshold and a minimum… |

</details>

<details>
<summary><code>Line</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func HoughLines(edges *cv.Mat, rhoRes, thetaRes float64, threshold int) []Line` | HoughLines detects straight lines in a binary edge image with the standard Hough transform. |
| `func (l Line) ToSegment(rows, cols int) (Segment, bool)` | ToSegment clips the infinite line to the rows×cols image rectangle and returns the visible portion as a `Segment`. |

</details>

<details>
<summary><code>Segment</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func HoughLinesP(edges *cv.Mat, rhoRes, thetaRes float64, threshold int, minLineLength, maxLineGap float64) []Segment` | HoughLinesP extracts finite line segments from a binary edge image. |
| `func LSD(src *cv.Mat, opts LSDOptions) []Segment` | LSD detects line segments in a single-channel image with a region-growing line-segment detector. |
| `func LineToSegment(l Line, rows, cols int) (Segment, bool)` | LineToSegment clips the infinite line l to the rows×cols image rectangle and returns the visible portion as a `Segment`. |
| `func (s Segment) Angle() float64` | Angle returns the orientation of the segment atan2(dy,dx) in radians in the range (-pi, pi]. |
| `func (s Segment) Length() float64` | Length returns the Euclidean length of the segment. |
| `func (s Segment) Midpoint() (x, y float64)` | Midpoint returns the coordinates of the segment centre. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/edges2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
