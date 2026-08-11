# contours2 — provides contour extraction and shape analysis for the github.com/malcolmston/opencv

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/contours2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/contours2)

Package contours2 provides contour extraction and shape analysis for the
github.com/malcolmston/opencv library.

It operates on the library's existing image type, `cv.Mat`, and its geometry
types `cv.Point` and `cv.Point2f`; it does not introduce a competing image
representation. Binary or grayscale single-channel Mats are the raster input,
and contours are represented as ordered slices of `cv.Point` (the `Contour`
type), matching the parent package's conventions.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/contours2"
```

## Usage

This is the package's own `ExampleApproxPolyDP`, so it compiles and its output is
asserted on every `go test ./contours2/`.

```go
curve := []cv.Point{
		{X: 0, Y: 0}, {X: 5, Y: 0}, {X: 10, Y: 0},
		{X: 10, Y: 5}, {X: 10, Y: 10},
		{X: 5, Y: 10}, {X: 0, Y: 10},
		{X: 0, Y: 5},
	}
	fmt.Println(len(contours2.ApproxPolyDP(curve, 1.0, true)))
	fmt.Println(contours2.ContourArea(contours2.ApproxPolyDP(curve, 1.0, true)))
```

```
4
100
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ApproxPolyDP(curve []cv.Point, epsilon float64, closed bool) []cv.Point` | ApproxPolyDP approximates a polygonal curve with fewer vertices using the Ramer–Douglas–Peucker algorithm. |
| `func ArcLength(curve []cv.Point, closed bool) float64` | ArcLength returns the perimeter (or curve length) of a set of points. |
| `func AspectRatio(contour []cv.Point) float64` | AspectRatio returns the ratio of width to height of the upright bounding box of a contour. |
| `func ContourArea(contour []cv.Point) float64` | ContourArea returns the non-negative area enclosed by a contour, computed via the shoelace formula. |
| `func ContourAreaSigned(contour []cv.Point) float64` | ContourAreaSigned returns the signed polygon area of a contour using the shoelace formula. |
| `func ContourCentroid(contour []cv.Point) cv.Point2f` | ContourCentroid returns the centroid of a contour computed from its moments. |
| `func ConvexHull(pts []cv.Point, clockwise bool) []cv.Point` | ConvexHull returns the vertices of the convex hull of a set of points, computed with Andrew's monotone-chain algorithm (an O(n log n) variant that… |
| `func ConvexHullIndices(pts []cv.Point) []int` | ConvexHullIndices returns the indices into pts of the convex-hull vertices, sorted in ascending index (contour-traversal) order so that consecutive… |
| `func Eccentricity(contour []cv.Point) float64` | Eccentricity returns the eccentricity of the ellipse with the same second-order central moments as the contour, in the range [0, 1): 0 for a circle,… |
| `func EquivalentDiameter(contour []cv.Point) float64` | EquivalentDiameter returns the diameter of the circle whose area equals the contour area, sqrt(4*area/pi). |
| `func Extent(contour []cv.Point) float64` | Extent returns the ratio of the contour area to the area of its upright bounding box, a measure of how fully the shape fills its box. |
| `func ExtremePoints(contour []cv.Point) (left, right, top, bottom cv.Point)` | ExtremePoints returns the leftmost, rightmost, topmost and bottommost points of a contour. |
| `func FindContours(src *cv.Mat, mode RetrievalMode, approx ChainApproxMethod) ([]Contour, []HierarchyNode)` | FindContours extracts contours from a binary single-channel image using the Suzuki–Abe border-following algorithm. |
| `func FitLine(pts []cv.Point) (vx, vy, x0, y0 float64)` | FitLine fits a line to a set of points by total least squares (minimising the orthogonal distance) and returns a unit direction (vx, vy) together… |
| `func HuMoments(m Moments) [7]float64` | HuMoments returns the seven Hu invariant moments computed from a set of normalised central moments, matching OpenCV's HuMoments. |
| `func IsContourConvex(contour []cv.Point) bool` | IsContourConvex reports whether a polygon is convex, i.e. |
| `func MatchShapes(a, b Moments, method ShapeMatchMethod) float64` | MatchShapes compares two shapes given by their moments using one of the log-scaled Hu-moment metrics (`ContoursMatchI1`, `ContoursMatchI2` or… |
| `func MatchShapesHu(ha, hb [7]float64, method ShapeMatchMethod) float64` | MatchShapesHu compares two shapes directly from their Hu-moment vectors using the selected metric, matching OpenCV's matchShapes. |
| `func Orientation(contour []cv.Point) float64` | Orientation returns the angle, in degrees, of the major axis of the ellipse with the same second-order central moments as the contour, in the range… |
| `func PointPolygonTest(contour []cv.Point, pt cv.Point, measureDist bool) float64` | PointPolygonTest reports the relationship between a point and a polygon given by its vertices. |
| `func Solidity(contour []cv.Point) float64` | Solidity returns the ratio of the contour area to its convex-hull area, a measure of convexity in the range (0, 1]. |

### Types

| Type | What it is |
| --- | --- |
| `ChainApproxMethod` | ChainApproxMethod selects how `FindContours` stores each border's points. |
| `Circle` | Circle is a circle in fractional pixel coordinates, as returned by `MinEnclosingCircle`. |
| `Contour` | Contour is an ordered list of boundary points describing a connected shape, as produced by `FindContours`. |
| `ConvexityDefect` | ConvexityDefect describes a concavity of a contour relative to its convex hull, matching OpenCV's convexityDefects output. |
| `Ellipse` | Ellipse describes an ellipse by its centre, full axis lengths and rotation, matching the parameterisation of OpenCV's fitEllipse output. |
| `HierarchyNode` | HierarchyNode mirrors OpenCV's 4-element hierarchy entry. |
| `Moments` | Moments holds the spatial, central and normalised central moments of a shape up to third order, mirroring OpenCV's cv::Moments. |
| `Rect` | Rect is an axis-aligned rectangle in integer pixel coordinates, matching OpenCV's cv::Rect. |
| `RetrievalMode` | RetrievalMode selects which contours `FindContours` returns and how their hierarchy is built, mirroring OpenCV's RETR_* modes. |
| `RotatedRect` | RotatedRect is a rectangle that may be rotated about its centre, matching OpenCV's cv::RotatedRect. |
| `ShapeMatchMethod` | ShapeMatchMethod selects the metric used by `MatchShapes`, mirroring OpenCV's CONTOURS_MATCH_I1/I2/I3. |

<details>
<summary><code>Circle</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MinEnclosingCircle(pts []cv.Point) Circle` | MinEnclosingCircle returns the smallest circle that encloses every input point, computed with Welzl's algorithm evaluated deterministically over the… |
| `func (c Circle) Area() float64` | Area returns the area of the circle (pi*r^2). |
| `func (c Circle) Contains(p cv.Point2f) bool` | Contains reports whether the point lies within the circle (inclusive of the boundary, within a small tolerance). |

</details>

<details>
<summary><code>Contour</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FindExternalContours(src *cv.Mat, approx ChainApproxMethod) []Contour` | FindExternalContours is a convenience wrapper that returns only the outermost contours (`RetrExternal`) with the given chain approximation. |

</details>

<details>
<summary><code>ConvexityDefect</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ConvexityDefects(contour []cv.Point, hullIndices []int) []ConvexityDefect` | ConvexityDefects finds the concavities of a contour relative to its convex hull. |

</details>

<details>
<summary><code>Ellipse</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FitEllipse(pts []cv.Point) Ellipse` | FitEllipse fits an ellipse to a set of points and returns it in OpenCV's `Ellipse` parameterisation (centre, full axis lengths and rotation in… |
| `func (e Ellipse) Area() float64` | Area returns the area enclosed by the ellipse (pi*a*b for semi-axes a, b). |

</details>

<details>
<summary><code>Moments</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ContourMoments(contour []cv.Point) Moments` | ContourMoments computes the spatial, central and normalised central moments of a polygon (contour) using Green's theorem, matching the closed-form… |
| `func ImageMoments(src *cv.Mat) Moments` | ImageMoments computes the moments of a single-channel image, weighting each pixel (x, y) by its sample value. |
| `func (m Moments) Centroid() cv.Point2f` | Centroid returns the centre of mass (M10/M00, M01/M00). |
| `func (m Moments) HuMoments() [7]float64` | HuMoments returns the seven Hu invariant moments of the receiver's normalised central moments. |

</details>

<details>
<summary><code>Rect</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BoundingRect(pts []cv.Point) Rect` | BoundingRect returns the smallest axis-aligned integer rectangle that contains all the given points. |
| `func (r Rect) Area() int` | Area returns the rectangle area (Width*Height). |
| `func (r Rect) BottomRight() cv.Point` | BottomRight returns the bottom-right corner (exclusive) of the rectangle. |
| `func (r Rect) Contains(p cv.Point) bool` | Contains reports whether the point lies inside the rectangle, treating the top and left edges as inside and the bottom and right edges as outside… |
| `func (r Rect) TopLeft() cv.Point` | TopLeft returns the top-left corner of the rectangle. |

</details>

<details>
<summary><code>RotatedRect</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MinAreaRect(pts []cv.Point) RotatedRect` | MinAreaRect returns the minimum-area rotated rectangle enclosing the points, found by rotating calipers over the convex hull: for each hull edge the… |
| `func (r RotatedRect) Area() float64` | Area returns the area of the rotated rectangle (Width*Height). |
| `func (r RotatedRect) BoundingRect() Rect` | BoundingRect returns the smallest axis-aligned integer rectangle that contains all four corners of the rotated rectangle. |
| `func (r RotatedRect) Points() [4]cv.Point` | Points returns the four corner points of the rotated rectangle, rounded to the nearest pixel, in the same order as OpenCV's boxPoints /… |
| `func (r RotatedRect) PointsF() [4]cv.Point2f` | PointsF returns the four corner points of the rotated rectangle as floating-point coordinates, without rounding. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/contours2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
