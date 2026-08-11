# draw2 — provides a self-contained 2-D drawing and rendering toolkit for the parent

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/draw2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/draw2)

Package draw2 provides a self-contained 2-D drawing and rendering toolkit for
the parent computer-vision library's Mat image type.

It complements the core drawing primitives with higher-quality and higher
level routines: Wu anti-aliased lines and circles, thick and dashed lines,
polylines, filled and outlined polygons, midpoint circles, axis-aligned and
rotated ellipses, elliptical arcs, quadratic and cubic Bezier curves,
Catmull-Rom splines, a built-in bitmap text renderer, CV-style markers,
contour overlays, false-colour heatmaps and alpha compositing.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/draw2"
```

## Usage

This is the package's own `ExampleAlphaBlend`, so it compiles and its output is
asserted on every `go test ./draw2/`.

```go
dst := cv.NewMat(4, 4, 1)
	dst.SetTo(0)
	src := cv.NewMat(4, 4, 1)
	src.SetTo(200)

	draw2.AlphaBlend(dst, src, 0.25)
	fmt.Println(dst.At(0, 0, 0))
```

```
50
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AACircle(m *cv.Mat, center cv.Point, radius float64, color cv.Scalar)` | AACircle draws an anti-aliased circle outline of the given radius centred at center. |
| `func AAPolyline(m *cv.Mat, pts []cv.Point, closed bool, color cv.Scalar)` | AAPolyline draws a connected sequence of anti-aliased segments through pts using `WuLine`. |
| `func Alpha(c cv.Scalar) float64` | Alpha returns the alpha (fourth) component of a colour normalised to [0,1]. |
| `func AlphaBlend(dst, src *cv.Mat, alpha float64)` | AlphaBlend composites src over dst in place with a single global opacity: dst = (1-alpha)*dst + alpha*src for every sample. |
| `func AlphaCompositeMask(dst, src, mask *cv.Mat)` | AlphaCompositeMask composites src over dst in place using a single-channel mask as a per-pixel opacity map: where the mask is 255 src replaces dst,… |
| `func ArrowedLine(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar, thickness int, tipLength float64)` | ArrowedLine draws a line from pt1 to pt2 with an arrow head at pt2. |
| `func Blend(a, b *cv.Mat, wa, wb, gamma float64) *cv.Mat` | Blend returns a new Mat holding the weighted sum wa*a + wb*b + gamma of two equally-shaped Mats, mirroring OpenCV's addWeighted. |
| `func BoundingRect(pts []cv.Point) (tl, br cv.Point)` | BoundingRect returns the top-left and bottom-right corners of the smallest axis-aligned rectangle enclosing pts. |
| `func CatmullRomSpline(m *cv.Mat, pts []PointF, closed bool, color cv.Scalar, thickness, stepsPerSegment int)` | CatmullRomSpline draws a smooth Catmull-Rom spline passing through pts. |
| `func Centroid(pts []cv.Point) cv.Point` | Centroid returns the arithmetic mean of the points as an integer coordinate. |
| `func Circle(m *cv.Mat, center cv.Point, radius int, color cv.Scalar, thickness int)` | Circle draws the outline of a circle of the given radius centred at center using the midpoint (Bresenham) circle algorithm. |
| `func ColorBar(width, height int, p Palette) *cv.Mat` | ColorBar renders a vertical colour-bar legend of the given width and height into a new three-channel Mat, with the palette's maximum at the top and… |
| `func Colorize(src *cv.Mat, p Palette) *cv.Mat` | Colorize maps a single-channel Mat to a new three-channel RGB Mat by passing each 8-bit intensity (scaled to [0,1]) through the given palette. |
| `func CubicBezier(m *cv.Mat, p0, p1, p2, p3 PointF, color cv.Scalar, thickness, steps int)` | CubicBezier draws a cubic Bezier curve through control points p0, p1, p2 and p3. |
| `func DashedLine(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar, thickness, dashLen, gapLen int)` | DashedLine draws a dashed straight line from pt1 to pt2. |
| `func DrawBoundingBoxes(m *cv.Mat, contours [][]cv.Point, color cv.Scalar, thickness int)` | DrawBoundingBoxes draws the axis-aligned bounding rectangle of each contour. |
| `func DrawContours(m *cv.Mat, contours [][]cv.Point, contourIdx int, color cv.Scalar, thickness int)` | DrawContours draws each contour in contours as a closed polyline. |
| `func DrawMarker(m *cv.Mat, center cv.Point, marker MarkerType, size int, color cv.Scalar, thickness int)` | DrawMarker draws a marker glyph of the given type centred at center. |
| `func Ellipse(m *cv.Mat, center cv.Point, axesX, axesY int, color cv.Scalar, thickness int)` | Ellipse draws the outline of an axis-aligned ellipse centred at center with semi-axes axesX and axesY, using the midpoint ellipse algorithm. |
| `func EllipseRotated(m *cv.Mat, center cv.Point, axesX, axesY int, angle float64, color cv.Scalar, thickness int)` | EllipseRotated draws a rotated ellipse centred at center with semi-axes axesX and axesY, rotated angle degrees clockwise. |
| `func EllipticArc(m *cv.Mat, center cv.Point, axesX, axesY int, startAngle, endAngle float64, color cv.Scalar, thickness int)` | EllipticArc draws an elliptical arc centred at center with semi-axes axesX and axesY, sweeping from startAngle to endAngle in degrees (measured… |
| `func FillContours(m *cv.Mat, contours [][]cv.Point, color cv.Scalar)` | FillContours fills every contour with a solid colour using the even-odd rule, so nested contours (holes) cancel just as OpenCV's drawContours does… |
| `func FillConvexPolygon(m *cv.Mat, pts []cv.Point, color cv.Scalar)` | FillConvexPolygon fills a single convex polygon with a solid colour using a non-zero span per scanline. |
| `func FillPolygon(m *cv.Mat, polys [][]cv.Point, color cv.Scalar)` | FillPolygon fills one or more polygons with a solid colour using an even-odd scanline rule, so overlapping regions within a single call cancel. |
| `func FillRectangleAlpha(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar, alpha float64)` | FillRectangleAlpha fills the axis-aligned rectangle spanning pt1 and pt2 with color at the given opacity, alpha-compositing over the existing pixels. |
| `func FilledCircle(m *cv.Mat, center cv.Point, radius int, color cv.Scalar)` | FilledCircle fills a disc of the given radius centred at center with a solid colour, using an exact per-scanline span so the boundary matches the… |
| `func FilledEllipse(m *cv.Mat, center cv.Point, axesX, axesY int, color cv.Scalar)` | FilledEllipse fills an axis-aligned ellipse centred at center with semi-axes axesX and axesY, using an exact per-scanline span derived from the… |
| `func FilledRectangle(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar)` | FilledRectangle fills the axis-aligned rectangle spanning corners pt1 and pt2 with a solid colour. |
| `func Gray(v uint8) cv.Scalar` | Gray builds a colour Scalar whose three colour components all equal v, i.e. |
| `func Heatmap(values [][]float64, p Palette) *cv.Mat` | Heatmap renders a two-dimensional grid of scalar values as a three-channel RGB Mat. |
| `func LerpColor(a, b cv.Scalar, t float64) cv.Scalar` | LerpColor linearly interpolates between colours a and b. |
| `func Line(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar, thickness int)` | Line draws a straight line from pt1 to pt2 with the given colour and thickness in pixels (minimum 1) using Bresenham's algorithm. |
| `func OverlayColor(m *cv.Mat, color cv.Scalar, alpha float64)` | OverlayColor fills the whole image with color at the given opacity, alpha-compositing over the existing pixels (a full-frame tint). |
| `func OverlayContours(m *cv.Mat, contours [][]cv.Point, color cv.Scalar, alpha float64)` | OverlayContours fills every contour with color at the given opacity, alpha-compositing the fill over the image. |
| `func PaletteColor(p Palette, t float64) cv.Scalar` | PaletteColor maps a normalised intensity t in [0,1] through the given palette and returns the resulting opaque RGB colour. |
| `func Polygon(m *cv.Mat, pts []cv.Point, color cv.Scalar, thickness int)` | Polygon draws the closed outline of a polygon through pts. |
| `func PolygonArea(pts []cv.Point) float64` | PolygonArea returns the absolute area enclosed by the polygon pts using the shoelace formula. |
| `func PolygonPerimeter(pts []cv.Point, closed bool) float64` | PolygonPerimeter returns the total edge length of the polygon pts. |
| `func Polyline(m *cv.Mat, pts []cv.Point, closed bool, color cv.Scalar, thickness int)` | Polyline draws a connected sequence of straight segments through pts. |
| `func PutText(m *cv.Mat, text string, org cv.Point, scale int, color cv.Scalar)` | PutText renders text using a built-in 5x7 bitmap font. |
| `func PutTextCentered(m *cv.Mat, text string, center cv.Point, scale int, color cv.Scalar)` | PutTextCentered renders text horizontally centred on the point center, with center.Y giving the text's vertical middle. |
| `func QuadraticBezier(m *cv.Mat, p0, p1, p2 PointF, color cv.Scalar, thickness, steps int)` | QuadraticBezier draws a quadratic Bezier curve through control points p0, p1 and p2. |
| `func RGB(r, g, b uint8) cv.Scalar` | RGB builds an opaque colour Scalar from 8-bit red, green and blue components. |
| `func RGBA(r, g, b, a uint8) cv.Scalar` | RGBA builds a colour Scalar from 8-bit red, green, blue and alpha components. |
| `func Rectangle(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar, thickness int)` | Rectangle draws the outline of an axis-aligned rectangle spanning corners pt1 and pt2. |
| `func RegularPolygon(m *cv.Mat, center cv.Point, radius, n int, rotation float64, color cv.Scalar, thickness int)` | RegularPolygon draws a regular polygon with n sides inscribed in a circle of the given radius centred at center. |
| `func RoundedRectangle(m *cv.Mat, pt1, pt2 cv.Point, radius int, color cv.Scalar, thickness int)` | RoundedRectangle draws the outline of a rectangle whose corners are rounded with the given radius. |
| `func TextSize(text string, scale int) (width, height int)` | TextSize returns the pixel width and height that `PutText` would occupy when rendering text at the given scale, matching the cursor advance PutText… |
| `func ThickLine(m *cv.Mat, pt1, pt2 cv.Point, color cv.Scalar, width int)` | ThickLine draws a filled rectangular line of the given width from pt1 to pt2. |
| `func WuLine(m *cv.Mat, x0, y0, x1, y1 float64, color cv.Scalar)` | WuLine draws an anti-aliased line from pt1 to pt2 using Xiaolin Wu's algorithm. |

### Types

| Type | What it is |
| --- | --- |
| `MarkerType` | MarkerType selects the glyph drawn by `DrawMarker`. |
| `Palette` | Palette selects a false-colour ramp for the colour-mapping routines. |
| `PointF` | PointF is a floating-point 2-D coordinate used by the curve routines for sub-pixel control points. |

<details>
<summary><code>PointF</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CubicBezierPoint(p0, p1, p2, p3 PointF, t float64) PointF` | CubicBezierPoint evaluates a cubic Bezier curve with control points p0, p1, p2 and p3 at parameter t in [0,1] and returns the resulting position. |
| `func PointFOf(p cv.Point) PointF` | PointFOf converts an integer Point into a PointF. |
| `func QuadBezierPoint(p0, p1, p2 PointF, t float64) PointF` | QuadBezierPoint evaluates a quadratic Bezier curve with control points p0, p1 and p2 at parameter t in [0,1] and returns the resulting position. |
| `func (p PointF) Add(q PointF) PointF` | Add returns the component-wise sum of p and q. |
| `func (p PointF) Scale(s float64) PointF` | Scale returns p with both components multiplied by s. |
| `func (p PointF) Sub(q PointF) PointF` | Sub returns the component-wise difference p minus q. |
| `func (p PointF) ToPoint() cv.Point` | ToPoint rounds p to the nearest integer github.com/malcolmston/opencv.Point. |

</details>

### Constants

`Filled`, `FontHeight`, `FontWidth`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/draw2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
