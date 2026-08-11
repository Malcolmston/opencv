# shapefit — provides geometric primitive fitting and Hough-transform shape detection for point sets

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/shapefit.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/shapefit)

Package shapefit provides geometric primitive fitting and Hough-transform
shape detection for point sets and binary edge images.

The package builds on the parent library's core types — it consumes binary
edge images as `github.com/malcolmston/opencv.Mat` and represents point sets
as slices of `github.com/malcolmston/opencv.Point2f` (floating-point image
coordinates, x is the column and y is the row). Fitted results are returned as
the small value types defined here (`Line`, `Circle`, `Ellipse`) which
interoperate with the parent library where natural: an `Ellipse` converts to
and from `github.com/malcolmston/opencv.RotatedRect`.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/shapefit"
```

## Usage

This is the package's own `ExampleCircleThrough`, so it compiles and its output is
asserted on every `go test ./shapefit/`.

```go
c, ok := shapefit.CircleThrough(
		cv.Point2f{X: 1, Y: 0}, cv.Point2f{X: 0, Y: 1}, cv.Point2f{X: -1, Y: 0})
	fmt.Printf("ok=%v center=(%.1f, %.1f) radius=%.1f\n", ok, c.Center.X, c.Center.Y, c.Radius)

	_, ok = shapefit.CircleThrough(
		cv.Point2f{X: 0, Y: 0}, cv.Point2f{X: 1, Y: 0}, cv.Point2f{X: 2, Y: 0})
	fmt.Println("collinear ok:", ok)
```

```
ok=true center=(0.0, 0.0) radius=1.0
collinear ok: false
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func BoundingBox(pts []cv.Point2f) (min, max cv.Point2f)` | BoundingBox returns the axis-aligned bounding box of the point set as its minimum and maximum corners. |
| `func Centroid(pts []cv.Point2f) cv.Point2f` | Centroid returns the arithmetic mean of the point set. |
| `func CircleRMSE(c Circle, pts []cv.Point2f) float64` | CircleRMSE returns the root-mean-square radial residual of the points with respect to the circle. |
| `func CircleResiduals(c Circle, pts []cv.Point2f) []float64` | CircleResiduals returns the radial residual of each point with respect to the circle, in input order. |
| `func DetectRotationalSymmetry(pts []cv.Point2f, maxFold int, tol float64) (fold int, score float64)` | DetectRotationalSymmetry estimates the order of rotational symmetry of a point set about its centroid. |
| `func EllipseAlgebraicResiduals(e Ellipse, pts []cv.Point2f) []float64` | EllipseAlgebraicResiduals returns, for each point, the absolute value of the normalized conic polynomial \|A·x²+B·x·y+C·y²+D·x+E·y+F\|… |
| `func EstimateOrientations(pts []cv.Point2f, k int) []float64` | EstimateOrientations returns, for each point, the local edge orientation in radians in the range [0, π), estimated by fitting a total-least-squares… |
| `func LineRMSE(l Line, pts []cv.Point2f) float64` | LineRMSE returns the root-mean-square perpendicular distance from the points to the line. |
| `func LineResiduals(l Line, pts []cv.Point2f) []float64` | LineResiduals returns the unsigned perpendicular distance from each point to the line, in the same order as the input. |
| `func PointsFromMat(m *cv.Mat, threshold uint8) []cv.Point2f` | PointsFromMat extracts the coordinates of the foreground pixels of a binary or grayscale image as a point set. |
| `func PointsToMat(pts []cv.Point2f, rows, cols int) *cv.Mat` | PointsToMat rasterizes a point set into a new single-channel Mat of the given size. |

### Types

| Type | What it is |
| --- | --- |
| `Circle` | Circle is a circle described by its center and radius. |
| `Ellipse` | Ellipse is an ellipse described by its center, semi-axis lengths and orientation. |
| `EllipseDetection` | EllipseDetection is an ellipse found by `DetectEllipses` together with the accumulator support that produced it. |
| `GeneralizedHough` | GeneralizedHough implements Ballard's generalized Hough transform for detecting instances of an arbitrary shape described by a template point set. |
| `HoughCircle` | HoughCircle is a circle detected by the Hough transform. |
| `HoughLine` | HoughLine is a line detected by the standard Hough transform, parameterized in polar (rho, theta) form: the line is the set of points where… |
| `HoughMatch` | HoughMatch is a candidate placement of a template found by the generalized Hough transform. |
| `Line` | Line is an infinite 2D line in normalized normal form a·x + b·y + c = 0 with a² + b² = 1. |
| `Primitive` | Primitive is the result of `FitBestPrimitive`: the geometric shape that best explains a point set, along with its RANSAC inlier support. |
| `PrimitiveKind` | PrimitiveKind identifies which geometric primitive a `Primitive` holds. |
| `RANSACParams` | RANSACParams configures the randomized fitting routines. |
| `RectangleDetection` | RectangleDetection is a rectangle recovered from a set of corner points by `DetectRectangles`. |
| `Segment` | Segment is a finite line segment between two endpoints. |
| `SymmetryAxis` | SymmetryAxis is a line of reflective symmetry through a point set, given by a point on the axis and its orientation. |

<details>
<summary><code>Circle</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CircleThrough(a, b, c cv.Point2f) (Circle, bool)` | CircleThrough returns the unique circle passing through the three points. |
| `func DetectCircles(pts []cv.Point2f, params RANSACParams, maxCircles int) []Circle` | DetectCircles extracts up to maxCircles circles from a point set by sequential RANSAC: it fits the dominant circle, removes its inliers, and repeats… |
| `func FitCircle(pts []cv.Point2f) (Circle, error)` | FitCircle fits a circle to the points using the Kåsa algebraic method, a linear least-squares fit that minimizes the algebraic residual (x²+y²) +… |
| `func FitCircleTaubin(pts []cv.Point2f) (Circle, error)` | FitCircleTaubin fits a circle using Taubin's method, which minimizes a gradient-weighted algebraic distance and largely removes the small-radius bias… |
| `func RANSACCircle(pts []cv.Point2f, params RANSACParams) (Circle, []int, error)` | RANSACCircle robustly fits a circle to points containing outliers, sampling random point triples as minimal circle models. |
| `func (c Circle) Area() float64` | Area returns the area π·r² enclosed by the circle. |
| `func (c Circle) Circumference() float64` | Circumference returns the perimeter 2·π·r of the circle. |
| `func (c Circle) Contains(p cv.Point2f) bool` | Contains reports whether p lies inside or on the circle. |
| `func (c Circle) Distance(p cv.Point2f) float64` | Distance returns the unsigned radial residual of p with respect to the circle: the absolute difference between the distance from p to the center and… |
| `func (c Circle) PointAt(theta float64) cv.Point2f` | PointAt returns the point on the circle at angle theta (radians), measured counter-clockwise from the positive x-axis. |

</details>

<details>
<summary><code>Ellipse</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func EllipseFromConic(A, B, C, D, E, F float64) (Ellipse, bool)` | EllipseFromConic converts the implicit conic A·x² + B·x·y + C·y² + D·x + E·y + F = 0 to a geometric `Ellipse`. |
| `func EllipseFromRotatedRect(r cv.RotatedRect) Ellipse` | EllipseFromRotatedRect builds an `Ellipse` from a github.com/malcolmston/opencv.RotatedRect, the inverse of `Ellipse.RotatedRect`. |
| `func FitEllipse(pts []cv.Point2f) (Ellipse, error)` | FitEllipse fits an ellipse to the points using the Halíř–Flusser numerically stable formulation of Fitzgibbon's direct least-squares method,… |
| `func RANSACEllipse(pts []cv.Point2f, params RANSACParams) (Ellipse, []int, error)` | RANSACEllipse robustly fits an ellipse to points containing outliers, sampling random subsets of five points as minimal ellipse models and scoring by… |
| `func (e Ellipse) Area() float64` | Area returns the area π·a·b enclosed by the ellipse. |
| `func (e Ellipse) Conic() (A, B, C, D, E, F float64)` | Conic returns the coefficients (A, B, C, D, E, F) of the implicit conic A·x² + B·x·y + C·y² + D·x + E·y + F = 0 that represents the ellipse. |
| `func (e Ellipse) Contains(p cv.Point2f) bool` | Contains reports whether p lies inside or on the ellipse. |
| `func (e Ellipse) Distance(p cv.Point2f) float64` | Distance returns an approximate signed-free radial distance from p to the ellipse boundary, measured along the ray from the center through p. |
| `func (e Ellipse) Eccentricity() float64` | Eccentricity returns the ellipse eccentricity in [0, 1); 0 is a circle. |
| `func (e Ellipse) Foci() (cv.Point2f, cv.Point2f)` | Foci returns the two focal points of the ellipse. |
| `func (e Ellipse) PointAt(t float64) cv.Point2f` | PointAt returns the point on the ellipse at parametric angle t (radians): the point (a·cos t, b·sin t) in the ellipse's own frame, rotated by Angle… |
| `func (e Ellipse) RotatedRect() cv.RotatedRect` | RotatedRect converts the ellipse to the parent library's github.com/malcolmston/opencv.RotatedRect, whose Width and Height are the full axis lengths… |

</details>

<details>
<summary><code>EllipseDetection</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DetectEllipses(pts []cv.Point2f, minVotes int, minMajor, maxMajor, minCenterDist float64) []EllipseDetection` | DetectEllipses detects ellipses in a point set using the Xie–Ji method. |

</details>

<details>
<summary><code>GeneralizedHough</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGeneralizedHough(template []cv.Point2f, reference cv.Point2f, orientationBins, neighbors int) *GeneralizedHough` | NewGeneralizedHough builds a generalized Hough model from a template edge point set and a reference point (commonly the template centroid). |
| `func (g *GeneralizedHough) Bins() int` | Bins returns the number of orientation bins in the R-table. |
| `func (g *GeneralizedHough) Detect(query []cv.Point2f, width, height, threshold int) []HoughMatch` | Detect locates instances of the template in a query edge point set spanning a width×height image. |
| `func (g *GeneralizedHough) Reference() cv.Point2f` | Reference returns the reference point the model was built with. |

</details>

<details>
<summary><code>HoughCircle</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func HoughCircles(src *cv.Mat, minRadius, maxRadius, threshold int, minDist float64) []HoughCircle` | HoughCircles detects circles in a binary edge image by voting into a per-radius center accumulator. |
| `func (h HoughCircle) ToCircle() Circle` | ToCircle converts the detection to a plain `Circle`. |

</details>

<details>
<summary><code>HoughLine</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func HoughLines(src *cv.Mat, rhoStep, thetaStep float64, threshold int) []HoughLine` | HoughLines runs the standard Hough line transform over the foreground pixels of a binary image (a pixel is foreground when its first-channel sample… |
| `func (h HoughLine) ToLine() Line` | ToLine converts the polar Hough line to the normalized normal-form `Line`. |

</details>

<details>
<summary><code>Line</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DetectLines(pts []cv.Point2f, params RANSACParams, maxLines int) []Line` | DetectLines extracts up to maxLines lines from a point set by sequential RANSAC: it fits the dominant line, removes its inliers, and repeats on the… |
| `func FitLine(pts []cv.Point2f) (Line, error)` | FitLine fits a line to the points by total least squares, minimizing the sum of squared perpendicular distances. |
| `func LineThroughPoints(p, q cv.Point2f) Line` | LineThroughPoints returns the normalized line passing through the two points. |
| `func RANSACLine(pts []cv.Point2f, params RANSACParams) (Line, []int, error)` | RANSACLine robustly fits a line to points containing outliers. |
| `func (l Line) Angle() float64` | Angle returns the orientation of the line in radians in the range (-π/2, π/2], measured from the positive x-axis to the line's direction. |
| `func (l Line) ClosestPoint(p cv.Point2f) cv.Point2f` | ClosestPoint returns the point on the line nearest to p (its orthogonal projection onto the line). |
| `func (l Line) Direction() cv.Point2f` | Direction returns a unit vector along the line (perpendicular to its normal). |
| `func (l Line) Distance(p cv.Point2f) float64` | Distance returns the unsigned perpendicular distance from p to the line. |
| `func (l Line) Normal() cv.Point2f` | Normal returns the line's unit normal vector (A, B). |
| `func (l Line) Normalized() Line` | Normalized returns an equivalent line whose normal (A, B) is unit length. |
| `func (l Line) SignedDistance(p cv.Point2f) float64` | SignedDistance returns the signed perpendicular distance from p to the line; its magnitude is the distance and its sign indicates which side of the… |

</details>

<details>
<summary><code>Primitive</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FitBestPrimitive(pts []cv.Point2f, params RANSACParams) Primitive` | FitBestPrimitive fits a line, a circle and an ellipse to the point set with RANSAC and returns whichever explains the most points as inliers. |

</details>

<details>
<summary><code>PrimitiveKind</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (k PrimitiveKind) String() string` | String returns the primitive kind name. |

</details>

<details>
<summary><code>RANSACParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultRANSACParams() RANSACParams` | DefaultRANSACParams returns a reasonable default configuration: 1000 iterations, a 2-pixel inlier threshold, no minimum inlier count, seed 1 and… |

</details>

<details>
<summary><code>RectangleDetection</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DetectRectangles(corners []cv.Point2f, tol float64) []RectangleDetection` | DetectRectangles finds rectangles among a set of corner points. |
| `func (r RectangleDetection) RotatedRect() cv.RotatedRect` | RotatedRect converts the detection to a github.com/malcolmston/opencv.RotatedRect with its angle in degrees, for interoperation with the parent… |

</details>

<details>
<summary><code>Segment</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func HoughLinesP(src *cv.Mat, rhoStep, thetaStep float64, threshold, minLineLength, maxLineGap int) []Segment` | HoughLinesP detects line segments in a binary image. |
| `func (s Segment) Length() float64` | Length returns the Euclidean length of the segment. |

</details>

<details>
<summary><code>SymmetryAxis</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DetectReflectionSymmetry(pts []cv.Point2f, angleSteps int, tol float64) (SymmetryAxis, bool)` | DetectReflectionSymmetry searches for the strongest axis of reflective symmetry of a point set. |
| `func (s SymmetryAxis) Reflect(p cv.Point2f) cv.Point2f` | Reflect returns the reflection of p across the symmetry axis. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/shapefit).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
