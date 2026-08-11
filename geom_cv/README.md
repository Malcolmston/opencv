# geom_cv — provides computational-geometry primitives that are useful for computer vision

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/geom_cv.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/geom_cv)

Package geom_cv provides computational-geometry primitives that are useful for
computer vision, built entirely on the Go standard library and on the core
types of the parent github.com/malcolmston/opencv package.

The package deliberately reuses the parent library's geometry and image types
rather than inventing incompatible ones: planar points are
`github.com/malcolmston/opencv.Point2f` (floating point) and
`github.com/malcolmston/opencv.Point` (integer pixel), rectangles are
`github.com/malcolmston/opencv.Rect` and
`github.com/malcolmston/opencv.RotatedRect`, and raster output is written into
`github.com/malcolmston/opencv.Mat`. Functions here therefore compose directly
with the rest of the library.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/geom_cv"
```

## Usage

This is the package's own `ExampleClipPolygon`, so it compiles and its output is
asserted on every `go test ./geom_cv/`.

```go
a := []cv.Point2f{{X: 0, Y: 0}, {X: 4, Y: 0}, {X: 4, Y: 4}, {X: 0, Y: 4}}
	b := []cv.Point2f{{X: 2, Y: 2}, {X: 6, Y: 2}, {X: 6, Y: 6}, {X: 2, Y: 6}}
	fmt.Println(len(geom_cv.ClipPolygon(a, b)))
	fmt.Println(geom_cv.PolygonIntersectionArea(a, b))
```

```
4
4
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func Add(a, b cv.Point2f) cv.Point2f` | Add returns the component-wise sum a+b of two vectors. |
| `func AngleBetween(a, b cv.Point2f) float64` | AngleBetween returns the unsigned angle in radians, in [0, π], between the vectors a and b. |
| `func AntipodalPairs(hull []cv.Point2f) [][2]int` | AntipodalPairs returns the indices of all antipodal vertex pairs of a convex polygon given in counter-clockwise order, computed with the… |
| `func Circumcenter(a, b, c cv.Point2f) (cv.Point2f, bool)` | Circumcenter returns the center of the circle passing through the three points a, b and c, and true on success. |
| `func ClipPolygon(subject, clip []cv.Point2f) []cv.Point2f` | ClipPolygon clips the subject polygon against the convex clip polygon with the Sutherland–Hodgman algorithm and returns the intersection polygon in… |
| `func ClosestPointOnSegment(a, b, p cv.Point2f) cv.Point2f` | ClosestPointOnSegment returns the point on segment ab that is nearest to p, which is one of the endpoints when the perpendicular foot falls outside… |
| `func Collinear(a, b, c cv.Point2f) bool` | Collinear reports whether the three points a, b and c lie on a common line within the package tolerance. |
| `func ConvexDiameter(pts []cv.Point2f) float64` | ConvexDiameter returns the diameter of a point set: the largest distance between any two of its points. |
| `func ConvexHull(pts []cv.Point2f) []cv.Point2f` | ConvexHull computes the convex hull of a set of points using Andrew's monotone-chain algorithm and returns the hull vertices in counter-clockwise… |
| `func ConvexHullArea(pts []cv.Point2f) float64` | ConvexHullArea returns the area enclosed by the convex hull of the given points. |
| `func ConvexWidth(pts []cv.Point2f) float64` | ConvexWidth returns the minimum width of a point set: the smallest distance between two parallel supporting lines. |
| `func Cross(a, b cv.Point2f) float64` | Cross returns the scalar 2-D cross product a×b = a.X*b.Y - a.Y*b.X. Its sign tells whether b lies counter-clockwise (positive) or clockwise… |
| `func Distance(a, b cv.Point2f) float64` | Distance returns the Euclidean distance between points a and b. |
| `func DistanceSquared(a, b cv.Point2f) float64` | DistanceSquared returns the squared Euclidean distance between a and b. |
| `func Dot(a, b cv.Point2f) float64` | Dot returns the dot product a·b = a.X*b.X + a.Y*b.Y. |
| `func DrawPolygonOutline(m *cv.Mat, poly []cv.Point2f, value uint8)` | DrawPolygonOutline draws the closed polygon boundary in the image m using Bresenham line segments between consecutive vertices, writing value into… |
| `func EnsureCounterClockwise(poly []cv.Point2f) []cv.Point2f` | EnsureCounterClockwise returns the polygon with a guaranteed counter-clockwise winding (positive `SignedArea`): the input is returned as a copy if it… |
| `func FillPolygon(m *cv.Mat, poly []cv.Point2f, value uint8)` | FillPolygon fills the interior of the simple polygon in the existing image m, writing value into every channel of each covered pixel. |
| `func InCircumcircle(a, b, c, p cv.Point2f) bool` | InCircumcircle reports whether the point p lies strictly inside the circle circumscribing triangle abc. |
| `func IsClockwise(poly []cv.Point2f) bool` | IsClockwise reports whether the polygon vertices wind clockwise in a right-handed frame (equivalently, whether `SignedArea` is negative). |
| `func IsConvex(poly []cv.Point2f) bool` | IsConvex reports whether the simple polygon is convex, i.e. |
| `func Lerp(a, b cv.Point2f, t float64) cv.Point2f` | Lerp returns the linear interpolation (1-t)*a + t*b. |
| `func LineIntersection(a1, a2, b1, b2 cv.Point2f) (cv.Point2f, bool)` | LineIntersection computes the intersection point of the two infinite lines through (a1,a2) and (b1,b2). |
| `func Midpoint(a, b cv.Point2f) cv.Point2f` | Midpoint returns the point halfway between a and b. |
| `func MinAreaRect(pts []cv.Point2f) cv.RotatedRect` | MinAreaRect returns the minimum-area rotated rectangle enclosing the points as the parent library's github.com/malcolmston/opencv.RotatedRect. |
| `func MinEnclosingTriangleArea(pts []cv.Point2f) float64` | MinEnclosingTriangleArea returns the area of the smallest enclosing triangle aligned to the point set's convex hull edges. |
| `func NearestSite(sites []cv.Point2f, q cv.Point2f) int` | NearestSite returns the index of the site closest to the query point q, or -1 when sites is empty. |
| `func Norm(a cv.Point2f) float64` | Norm returns the Euclidean length (L2 norm) of the vector a. |
| `func Normalize(a cv.Point2f) cv.Point2f` | Normalize returns a unit-length vector pointing in the same direction as a. |
| `func Orientation(a, b, c cv.Point2f) int` | Orientation classifies the turn made going from a to b to c. |
| `func Perpendicular(a cv.Point2f) cv.Point2f` | Perpendicular returns the vector a rotated 90 degrees counter-clockwise in a right-handed frame, i.e. |
| `func PointInPolygon(poly []cv.Point2f, p cv.Point2f) bool` | PointInPolygon reports whether the point p lies strictly inside the simple polygon, using the even-odd ray-casting rule. |
| `func PointOnPolygonBoundary(poly []cv.Point2f, p cv.Point2f) bool` | PointOnPolygonBoundary reports whether the point p lies on any edge of the polygon, within the package tolerance. |
| `func PointOnSegment(a, b, p cv.Point2f) bool` | PointOnSegment reports whether the point p lies on the segment ab, within the package tolerance. |
| `func PointToLineDistance(a, b, p cv.Point2f) float64` | PointToLineDistance returns the perpendicular distance from the point p to the infinite line through a and b. |
| `func PointToSegmentDistance(a, b, p cv.Point2f) float64` | PointToSegmentDistance returns the shortest distance from the point p to the segment ab. |
| `func PolygonArea(poly []cv.Point2f) float64` | PolygonArea returns the (non-negative) area enclosed by the simple polygon, independent of winding direction. |
| `func PolygonCentroid(poly []cv.Point2f) cv.Point2f` | PolygonCentroid returns the area centroid (center of mass of the filled region) of the simple polygon. |
| `func PolygonIntersectionArea(subject, clip []cv.Point2f) float64` | PolygonIntersectionArea returns the area of the overlap between the subject polygon and the convex clip polygon. |
| `func PolygonMask(width, height int, poly []cv.Point2f) *cv.Mat` | PolygonMask rasterizes the simple polygon into a new single-channel github.com/malcolmston/opencv.Mat of the given size, setting every pixel whose… |
| `func PolygonPerimeter(poly []cv.Point2f) float64` | PolygonPerimeter returns the total length of the polygon boundary, treating the vertex list as a closed loop. |
| `func ReversePolygon(poly []cv.Point2f) []cv.Point2f` | ReversePolygon returns a new slice with the vertices in reverse order, which flips the polygon winding. |
| `func Scale(a cv.Point2f, s float64) cv.Point2f` | Scale returns the vector a multiplied by the scalar s. |
| `func SegmentDistance(p1, p2, p3, p4 cv.Point2f) float64` | SegmentDistance returns the shortest distance between the two closed segments p1p2 and p3p4. |
| `func SegmentIntersection(p1, p2, p3, p4 cv.Point2f) (cv.Point2f, bool)` | SegmentIntersection computes the intersection point of the two segments p1p2 and p3p4. |
| `func SegmentsIntersect(p1, p2, p3, p4 cv.Point2f) bool` | SegmentsIntersect reports whether the two closed segments p1p2 and p3p4 share at least one point, including collinear overlap and touching at an… |
| `func SignedArea(poly []cv.Point2f) float64` | SignedArea returns the signed area of the simple polygon given by its vertices in order (the polygon is implicitly closed). |
| `func Sub(a, b cv.Point2f) cv.Point2f` | Sub returns the component-wise difference a-b of two vectors. |
| `func ToPoint(p cv.Point2f) cv.Point` | ToPoint rounds a floating-point point to the nearest integer pixel and returns it as the parent library's github.com/malcolmston/opencv.Point. |
| `func ToPoint2f(p cv.Point) cv.Point2f` | ToPoint2f converts an integer pixel to a floating-point point. |
| `func VectorAngle(a cv.Point2f) float64` | VectorAngle returns the direction of the vector a as an angle in radians in the range (-π, π], measured with math.Atan2 (positive toward +Y). |
| `func WindingNumber(poly []cv.Point2f, p cv.Point2f) int` | WindingNumber returns the winding number of the closed polygon around the point p: the signed number of times the polygon boundary wraps around p… |

### Types

| Type | What it is |
| --- | --- |
| `BoundingBox` | BoundingBox is an axis-aligned rectangle described by its minimum and maximum corners. |
| `Circle` | Circle is a circle with a center and a radius. |
| `Line` | Line is an infinite line represented by a point P on the line and a direction vector Dir. |
| `Segment` | Segment is a finite line segment between two endpoints A and B. |
| `Triangle` | Triangle is a triangle given by its three corners. |
| `VoronoiCell` | VoronoiCell is the region of the plane closer to Site than to any other site, clipped to a bounding box. |
| `VoronoiEdge` | VoronoiEdge is a bounded edge of a Voronoi diagram, connecting the circumcenters of two adjacent Delaunay triangles. |

<details>
<summary><code>BoundingBox</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MinEnclosingBox(pts []cv.Point2f) BoundingBox` | MinEnclosingBox returns the smallest axis-aligned `BoundingBox` containing all input points. |
| `func PolygonBoundingBox(poly []cv.Point2f) BoundingBox` | PolygonBoundingBox returns the smallest axis-aligned `BoundingBox` that contains every vertex. |
| `func (b BoundingBox) Area() float64` | Area returns Width times Height. |
| `func (b BoundingBox) Center() cv.Point2f` | Center returns the midpoint of the box. |
| `func (b BoundingBox) Contains(p cv.Point2f) bool` | Contains reports whether the point p lies inside or on the box, within the package tolerance. |
| `func (b BoundingBox) Corners() []cv.Point2f` | Corners returns the four corners of the box in counter-clockwise order starting from Min: (Min.X,Min.Y), (Max.X,Min.Y), (Max.X,Max.Y), (Min.X,Max.Y). |
| `func (b BoundingBox) Height() float64` | Height returns the vertical extent Max.Y-Min.Y of the box. |
| `func (b BoundingBox) Rect() cv.Rect` | Rect converts the box to the parent library's integer github.com/malcolmston/opencv.Rect, rounding the corners outward so the rectangle fully covers… |
| `func (b BoundingBox) Width() float64` | Width returns the horizontal extent Max.X-Min.X of the box. |

</details>

<details>
<summary><code>Circle</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Circumcircle(a, b, c cv.Point2f) (Circle, bool)` | Circumcircle returns the unique circle passing through the three points a, b and c, and true on success. |
| `func MinEnclosingCircle(pts []cv.Point2f) Circle` | MinEnclosingCircle returns the smallest circle that contains every input point, computed with Welzl's algorithm. |
| `func (c Circle) Area() float64` | Area returns the area π·r² enclosed by the circle. |
| `func (c Circle) Contains(p cv.Point2f) bool` | Contains reports whether the point p lies inside or on the circle, within the package tolerance. |

</details>

<details>
<summary><code>Line</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func LineThrough(a, b cv.Point2f) Line` | LineThrough returns the infinite line passing through the two points a and b, with its direction pointing from a toward b. |
| `func (l Line) PointAt(t float64) cv.Point2f` | PointAt returns the point P + t*Dir on the line. |

</details>

<details>
<summary><code>Segment</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func AlphaShapeEdges(pts []cv.Point2f, alpha float64) []Segment` | AlphaShapeEdges returns the boundary edges of the alpha shape of the point set for the given alpha radius. |
| `func ClipSegmentToBox(a, b cv.Point2f, box BoundingBox) (Segment, bool)` | ClipSegmentToBox clips the segment ab to the axis-aligned box using the Liang–Barsky algorithm. |
| `func (s Segment) Direction() cv.Point2f` | Direction returns the unit vector pointing from A to B, or the zero vector if the endpoints coincide. |
| `func (s Segment) Length() float64` | Length returns the Euclidean length of the segment. |
| `func (s Segment) Midpoint() cv.Point2f` | Midpoint returns the point halfway between the segment endpoints. |

</details>

<details>
<summary><code>Triangle</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func AlphaComplexTriangles(pts []cv.Point2f, alpha float64) []Triangle` | AlphaComplexTriangles returns the triangles of the Delaunay triangulation whose circumradius is at most alpha. |
| `func DelaunayTriangulation(points []cv.Point2f) []Triangle` | DelaunayTriangulation computes a Delaunay triangulation of the input points using the Bowyer–Watson incremental algorithm and returns the resulting… |
| `func (t Triangle) Area() float64` | Area returns the unsigned area of the triangle. |
| `func (t Triangle) Centroid() cv.Point2f` | Centroid returns the arithmetic mean of the three corners, which is the triangle's center of mass. |
| `func (t Triangle) Contains(p cv.Point2f) bool` | Contains reports whether the point p lies inside or on the boundary of the triangle. |
| `func (t Triangle) SignedArea() float64` | SignedArea returns the signed area of the triangle: positive when A→B→C winds counter-clockwise in a right-handed frame and negative otherwise. |

</details>

<details>
<summary><code>VoronoiCell</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func VoronoiCells(sites []cv.Point2f, box BoundingBox) []VoronoiCell` | VoronoiCells returns the Voronoi region of every site, clipped to the given bounding box. |

</details>

<details>
<summary><code>VoronoiEdge</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func VoronoiEdges(sites []cv.Point2f) []VoronoiEdge` | VoronoiEdges returns the finite (bounded) edges of the Voronoi diagram of the input sites, obtained as the dual of the Delaunay triangulation: for… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/geom_cv).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
