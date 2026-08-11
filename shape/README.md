# shape — provides shape descriptors, fitting and matching built on top

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/shape.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/shape)

Package shape provides shape descriptors, fitting and matching built on top of
the standard-library-only OpenCV port github.com/malcolmston/opencv (imported
here as cv). It mirrors a useful subset of OpenCV's shape-analysis routines
that operate on point sets and contours rather than on raster images.

- Enclosing shapes. `MinEnclosingCircle` finds the smallest circle covering a
point set using Welzl's algorithm, and `MinEnclosingTriangle` finds a small
enclosing triangle by optimising supporting lines around the convex hull. -
Fitting. `FitLine` fits a line to a point set by total least squares (the
first principal component of the centred points), `FitLineRobust` refits it
with an M-estimator (`DistL1`, `DistHuber`, `DistFair`, `DistWelsch`…) so a
few outliers no longer dominate, and `FitEllipse` fits an ellipse with the
direct algebraic least-squares method of Fitzgibbon, reformulated for
numerical stability by Halir and Flusser. - Moments and invariants.
`ContourMoments` computes the geometric moments of a closed polygon via
Green's theorem, `HuMoments` derives the seven Hu invariants (invariant to
translation, scale and rotation) from a `cv.Moments` value, and `MatchShapes`
compares two contours through their Hu moments. - Convexity and predicates.
`ConvexityDefects` reports the notches of a contour relative to its convex
hull, `ConvexHullIndices` returns the hull as indices into the input,
`IsContourConvex` tests convexity, `PointPolygonTest` locates a point relative
to a polygon (with an optional signed distance), and
`RotatedRectangleIntersection` computes the overlap polygon of two rotated
rectangles. - Shape matching. `ShapeContextDistanceExtractor` measures the
dissimilarity of two contours with the shape-context pipeline of Belongie,
Malik and Puzicha — log-polar `ShapeContext` histograms, optimal
correspondence by the Hungarian algorithm (`SolveAssignment`) and a
thin-plate-spline bending-energy term — while `HausdorffDistanceExtractor`
uses the (partial) Hausdorff distance. `EMDL1` gives the earth mover's
distance between one-dimensional histograms. - Geometric transforms. The
`ShapeTransformer` interface is implemented by
`ThinPlateSplineShapeTransformer` (a minimum-bending-energy warp that
interpolates control-point correspondences) and `AffineTransformer` (a
full-affine or similarity fit); both estimate, apply and warp images.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/shape"
```

## Usage

This is the package's own `ExampleConvexityDefects`, so it compiles and its output is
asserted on every `go test ./shape/`.

```go
contour := []cv.Point{
		{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 5, Y: 3}, {X: 6, Y: 0},
		{X: 10, Y: 0}, {X: 10, Y: 10}, {X: 0, Y: 10},
	}
	hull := shape.ConvexHullIndices(contour)
	defects := shape.ConvexityDefects(contour, hull)
	for _, d := range defects {
		fmt.Printf("notch at %v depth %.0f\n", d.Far, d.Depth)
	}
```

```
notch at {5 3} depth 3
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ContourMoments(contour []cv.Point) cv.Moments` | ContourMoments computes the spatial, central and normalised central moments of a closed polygon (a contour) up to third order using Green's theorem,… |
| `func ConvexHullIndices(pts []cv.Point) []int` | ConvexHullIndices returns the convex hull of pts as indices into pts, in the order the hull vertices appear when walking the contour (ascending… |
| `func DirectedHausdorff(a, b []Point2D, metric int) float64` | DirectedHausdorff returns the directed Hausdorff distance from point set a to point set b: the largest, over points of a, of the distance from that… |
| `func EMDL1(h1, h2 []float64) float64` | EMDL1 returns the Earth Mover's Distance between two one-dimensional histograms (or signatures) under the L1 ground metric, mirroring the intent of… |
| `func FitEllipse(pts []cv.Point) cv.RotatedRect` | FitEllipse fits an ellipse to the point set with the direct least-squares method of Fitzgibbon, Pilu and Fisher, using the numerically stable… |
| `func FitLine(pts []cv.Point) (vx, vy, x0, y0 float64)` | FitLine fits a straight line to the point set by total least squares and returns it in OpenCV's (vx, vy, x0, y0) form: (vx, vy) is a unit vector… |
| `func FitLineRobust(pts []cv.Point, distType int) (vx, vy, x0, y0 float64)` | FitLineRobust fits a straight line to the point set with a robust M-estimator, returning it in OpenCV's (vx, vy, x0, y0) form: (vx, vy) is a unit… |
| `func HuMoments(m cv.Moments) [7]float64` | HuMoments returns the seven Hu invariant moments derived from the normalised central moments of m. |
| `func IsContourConvex(contour []cv.Point) bool` | IsContourConvex reports whether the closed polygon described by contour is convex. |
| `func MatchShapes(c1, c2 []cv.Point, method int) float64` | MatchShapes compares two contours by their Hu moments and returns a dissimilarity score: 0 for a perfect match, growing as the shapes differ. |
| `func MinEnclosingCircle(pts []cv.Point) (cx, cy, radius float64)` | MinEnclosingCircle returns the smallest circle that encloses every point in pts, computed with Welzl's move-to-front algorithm. |
| `func MinEnclosingTriangle(pts []cv.Point) (area float64, tri [3][2]float64)` | MinEnclosingTriangle returns a minimal-area triangle that encloses every point in pts, together with its three vertices in fractional pixel… |
| `func PointPolygonTest(contour []cv.Point, pt Point2D, measureDist bool) float64` | PointPolygonTest determines the relationship between the point pt and the closed polygon contour, mirroring OpenCV's cv::pointPolygonTest. |
| `func RotatedRectangleIntersection(r1, r2 cv.RotatedRect) (IntersectionKind, []Point2D)` | RotatedRectangleIntersection computes the overlap of two rotated rectangles, mirroring OpenCV's cv::rotatedRectangleIntersection. |
| `func RoundPoints(pts []Point2D) []cv.Point` | RoundPoints converts a slice of `Point2D` to integer `cv.Point` values, rounding each coordinate to the nearest pixel. |
| `func SolveAssignment(cost [][]float64) (assignment []int, total float64)` | SolveAssignment solves the linear assignment problem: given a cost matrix in which cost`i``j` is the price of assigning row i to column j, it returns… |

### Types

| Type | What it is |
| --- | --- |
| `AffineTransformer` | AffineTransformer maps the plane with a single affine (or, optionally, similarity) transform fitted by least squares to point correspondences,… |
| `Defect` | Defect describes one convexity defect of a contour: a stretch of the contour that dips inward, away from its convex hull. |
| `HausdorffDistanceExtractor` | HausdorffDistanceExtractor measures the dissimilarity of two shapes by the (partial) Hausdorff distance between their point sets, mirroring OpenCV's… |
| `IntersectionKind` | IntersectionKind classifies the outcome of `RotatedRectangleIntersection`, mirroring OpenCV's RectanglesIntersectTypes. |
| `Point2D` | Point2D is a two-dimensional point with floating-point coordinates. |
| `ShapeContext` | ShapeContext computes the shape-context descriptor of a point set: for each point, a coarse log-polar histogram of the relative positions of all the… |
| `ShapeContextDistanceExtractor` | ShapeContextDistanceExtractor measures the dissimilarity of two shapes with the shape-context matching pipeline of Belongie, Malik and Puzicha,… |
| `ShapeTransformer` | ShapeTransformer is the common interface of the geometric shape transformers in this package, mirroring OpenCV's cv::ShapeTransformer. |
| `ThinPlateSplineShapeTransformer` | ThinPlateSplineShapeTransformer warps the plane with a thin-plate spline, the minimum-bending-energy interpolant through a set of control-point… |

<details>
<summary><code>AffineTransformer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewAffineTransformer(fullAffine bool) *AffineTransformer` | NewAffineTransformer returns an affine transformer. |
| `func (t *AffineTransformer) ApplyTransformation(pts []Point2D) []Point2D` | ApplyTransformation maps points through the fitted affine transform. |
| `func (t *AffineTransformer) EstimateTransformation(source, target []Point2D)` | EstimateTransformation fits the affine (or similarity) transform mapping source`i` to target`i` in the least-squares sense. |
| `func (t *AffineTransformer) Matrix() [2][3]float64` | Matrix returns the fitted forward affine matrix as a 2×3 array (rows are the x and y output equations). |
| `func (t *AffineTransformer) WarpImage(src *cv.Mat) *cv.Mat` | WarpImage resamples src through the fitted affine transform into a new image of the same size, using bilinear sampling and a zero border. |

</details>

<details>
<summary><code>Defect</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ConvexityDefects(contour []cv.Point, hull []int) []Defect` | ConvexityDefects computes the convexity defects of a contour relative to its convex hull. |

</details>

<details>
<summary><code>HausdorffDistanceExtractor</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewHausdorffDistanceExtractor(distanceFlag int, rankProportion float64) *HausdorffDistanceExtractor` | NewHausdorffDistanceExtractor returns an extractor using the given ground metric and rank proportion (pass 1.0 for the classic Hausdorff distance). |
| `func (e *HausdorffDistanceExtractor) ComputeDistance(c1, c2 []cv.Point) float64` | ComputeDistance returns the symmetric (partial) Hausdorff distance between contours c1 and c2. |

</details>

<details>
<summary><code>Point2D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FloatPoints(pts []cv.Point) []Point2D` | FloatPoints converts a slice of integer `cv.Point` to `Point2D` values. |

</details>

<details>
<summary><code>ShapeContext</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (sc ShapeContext) Descriptor(points []Point2D) [][]float64` | Descriptor returns the shape-context histograms of the point set: one normalised histogram per point, each of length RadialBins×AngularBins, laid… |

</details>

<details>
<summary><code>ShapeContextDistanceExtractor</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewShapeContextDistanceExtractor(nPoints int, rotationInvariant bool) *ShapeContextDistanceExtractor` | NewShapeContextDistanceExtractor returns an extractor that resamples each contour to nPoints and, when rotationInvariant is true, uses… |
| `func (e *ShapeContextDistanceExtractor) ComputeDistance(c1, c2 []cv.Point) float64` | ComputeDistance returns the shape-context dissimilarity between contours c1 and c2: 0 for identical shapes and growing as they differ. |

</details>

<details>
<summary><code>ThinPlateSplineShapeTransformer</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewThinPlateSplineShapeTransformer(regularization float64) *ThinPlateSplineShapeTransformer` | NewThinPlateSplineShapeTransformer returns a thin-plate spline transformer with the given regularization parameter (0 for exact interpolation). |
| `func (t *ThinPlateSplineShapeTransformer) ApplyTransformation(pts []Point2D) []Point2D` | ApplyTransformation maps points through the fitted spline. |
| `func (t *ThinPlateSplineShapeTransformer) BendingEnergy() float64` | BendingEnergy returns the bending energy of the fitted spline's non-affine part: zero for a purely affine correspondence and larger for more warping. |
| `func (t *ThinPlateSplineShapeTransformer) EstimateTransformation(source, target []Point2D)` | EstimateTransformation fits the thin-plate spline so that source`i` maps to target`i`. |
| `func (t *ThinPlateSplineShapeTransformer) WarpImage(src *cv.Mat) *cv.Mat` | WarpImage resamples src through the fitted spline into a new image of the same size. |

</details>

### Constants

`HausdorffL1`, `HausdorffL2`, `ContoursMatchI1`, `ContoursMatchI2`, `ContoursMatchI3`, `DistL1`, `DistL2`, `DistL12`, `DistFair`, `DistWelsch`, `DistHuber`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/shape).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
