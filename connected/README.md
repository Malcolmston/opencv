# connected — provides connected-component analysis for binary images

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/connected.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/connected)

Package connected provides connected-component analysis for binary images,
built entirely on the parent module's `github.com/malcolmston/opencv.Mat` type
and the Go standard library.

- Labelling: a two-pass union-find labeller (`Label`, `LabelWithStats`) that
assigns a unique integer to every 4- or 8-connected foreground region and
returns a compact, gap-free `Labels` map. - Component statistics:
per-component area, bounding box and centroid (`Component`, `ComponentStats`),
plus convenience selectors such as `LargestComponent`, `FilterByArea` and
`RemoveSmallComponents`. - Blob analysis: richer shape descriptors (`Blob`,
`AnalyzeBlobs`) with perimeter, circularity, extent and equivalent diameter. -
Flood fill: a span-based (scanline) region fill (`FloodFill`, `FloodFillMask`,
`FloodFillTolerance`) that grows a region from a seed. - Topology: interior
hole filling (`FillHoles`), hole counting (`CountHoles`) and the
`EulerNumber`. - Boundaries: inner/outer boundary extraction (`Boundary`,
`OuterBoundary`), perimeter measurement (`Perimeter`) and Moore boundary
tracing (`TraceBoundary`).

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/connected"
```

## Usage

This is the package's own `ExampleAnalyzeBlobs`, so it compiles and its output is
asserted on every `go test ./connected/`.

```go
m := fromRows(
		"......",
		".####.",
		".####.",
		".####.",
		".####.",
		"......",
	)
	b := connected.AnalyzeBlobs(m, connected.Conn8)[0]
	fmt.Println(b.Area, b.Perimeter, b.Holes)
	fmt.Printf("%.1f %.1f\n", b.AspectRatio(), b.Extent())
```

```
16 12 0
1.0 1.0
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func And(a, b *cv.Mat) *cv.Mat` | And returns the pixelwise intersection of two binary masks: a pixel is 255 only where both a and b are non-zero. |
| `func Areas(src *cv.Mat, conn Connectivity) []int` | Areas returns the pixel area of every foreground component, ordered by label. |
| `func Binarize(src *cv.Mat, thresh uint8) *cv.Mat` | Binarize returns a binary single-channel image in which every input sample strictly greater than thresh becomes foreground (255) and every other… |
| `func Boundary(src *cv.Mat, conn Connectivity) *cv.Mat` | Boundary returns the inner boundary of the foreground: every foreground pixel that has at least one background neighbour under conn, or that lies on… |
| `func BoundaryPoints(src *cv.Mat, conn Connectivity) []cv.Point` | BoundaryPoints returns, in raster order, the coordinates of all inner boundary pixels of the foreground (the pixels `Boundary` marks). |
| `func BoundingBoxes(src *cv.Mat, conn Connectivity) []cv.Rect` | BoundingBoxes returns the bounding box of every foreground component, ordered by label. |
| `func ComponentBoundaries(src *cv.Mat, conn Connectivity) [][]cv.Point` | ComponentBoundaries labels src and returns the traced outer contour of every foreground component, ordered by label. |
| `func ConnectedAt(m *cv.Mat, x, y int, conn Connectivity) *cv.Mat` | ConnectedAt returns a binary mask of the single foreground component that contains pixel (x, y). |
| `func CountBackground(src *cv.Mat) int` | CountBackground returns the number of zero (background) samples in src. |
| `func CountComponents(src *cv.Mat, conn Connectivity) int` | CountComponents returns the number of connected foreground components in a binary image. |
| `func CountForeground(src *cv.Mat) int` | CountForeground returns the number of non-zero (foreground) samples in src. |
| `func CountHoles(src *cv.Mat, conn Connectivity) int` | CountHoles returns the number of holes in the foreground: the number of background components that are fully enclosed and not connected to the image… |
| `func Equal(a, b *cv.Mat) bool` | Equal reports whether two binary masks agree at every pixel, comparing by foreground/background rather than exact sample value. |
| `func EulerNumber(src *cv.Mat, conn Connectivity) int` | EulerNumber returns the Euler number of the binary image: the number of connected foreground components minus the number of holes. |
| `func FillHoles(src *cv.Mat, conn Connectivity) *cv.Mat` | FillHoles fills the interior holes of the foreground: every background region not connected to the image border is turned into foreground. |
| `func FilterByArea(src *cv.Mat, conn Connectivity, minArea, maxArea int) *cv.Mat` | FilterByArea returns a binary image keeping only components whose area is in the inclusive range [minArea, maxArea]. |
| `func FloodFill(m *cv.Mat, seedX, seedY int, newVal uint8, conn Connectivity) int` | FloodFill fills the connected region of m that contains the seed pixel and shares its exact value, replacing it with newVal. |
| `func FloodFillMask(m *cv.Mat, seedX, seedY int, conn Connectivity) (*cv.Mat, int)` | FloodFillMask flood-fills the connected region of m sharing the seed pixel's value and returns a binary mask marking that region (255 inside, 0… |
| `func FloodFillTolerance(m *cv.Mat, seedX, seedY int, newVal uint8, tolerance int, conn Connectivity) int` | FloodFillTolerance fills the region reachable from the seed whose samples lie within tolerance of the seed's original value, replacing them with… |
| `func HolesMask(src *cv.Mat, conn Connectivity) *cv.Mat` | HolesMask returns a binary image marking only the filled-in holes: background pixels that are enclosed by the foreground and therefore not connected… |
| `func Invert(src *cv.Mat) *cv.Mat` | Invert returns the complement of a binary image: foreground pixels (non-zero) become 0 and background pixels become 255. |
| `func IsBinary(src *cv.Mat) bool` | IsBinary reports whether every sample of src is either 0 or 255. |
| `func KeepLargest(src *cv.Mat, conn Connectivity) *cv.Mat` | KeepLargest is an alias for `LargestComponentMask`: it removes every component except the single largest one. |
| `func KeepLargestN(src *cv.Mat, conn Connectivity, n int) *cv.Mat` | KeepLargestN returns a binary image retaining only the n largest components by area (255 on kept components, 0 elsewhere). |
| `func LabelWithStats(src *cv.Mat, conn Connectivity) (*Labels, []Component)` | LabelWithStats labels src like `Label` and additionally returns per-component statistics (area, bounding box and centroid), mirroring OpenCV's… |
| `func LargestComponentMask(src *cv.Mat, conn Connectivity) *cv.Mat` | LargestComponentMask returns a binary image containing only the largest component (255 on that component, 0 elsewhere). |
| `func Or(a, b *cv.Mat) *cv.Mat` | Or returns the pixelwise union of two binary masks: a pixel is 255 wherever either a or b is non-zero. |
| `func OuterBoundary(src *cv.Mat, conn Connectivity) *cv.Mat` | OuterBoundary returns the outer boundary of the foreground: every background pixel that has at least one foreground neighbour under conn is marked… |
| `func Perimeter(src *cv.Mat, conn Connectivity) int` | Perimeter returns the number of inner boundary pixels of the foreground, a simple pixel-count estimate of the total contour length under conn. |
| `func RegionPoints(m *cv.Mat, seedX, seedY int, conn Connectivity) []cv.Point` | RegionPoints returns the coordinates of every pixel in the connected region of m that contains the seed and shares its value, in raster order. |
| `func RemoveSmallComponents(src *cv.Mat, conn Connectivity, minArea int) *cv.Mat` | RemoveSmallComponents returns a binary image with every component smaller than minArea pixels deleted; the survivors are rendered at 255. |
| `func Subtract(a, b *cv.Mat) *cv.Mat` | Subtract returns the set difference a minus b: a pixel is 255 where a is non-zero and b is zero. |
| `func TraceBoundary(src *cv.Mat, startX, startY int) []cv.Point` | TraceBoundary follows the outer contour of the connected component that contains the foreground pixel (startX, startY) using Moore-neighbour tracing… |
| `func Xor(a, b *cv.Mat) *cv.Mat` | Xor returns the pixelwise symmetric difference of two binary masks: a pixel is 255 where exactly one of a and b is non-zero. |

### Types

| Type | What it is |
| --- | --- |
| `Blob` | Blob is a richer descriptor for one connected component, adding shape measurements to the basic statistics of a `Component`. |
| `Component` | Component holds summary statistics for one connected component, mirroring the per-label output of OpenCV's connectedComponentsWithStats. |
| `Connectivity` | Connectivity selects the pixel neighbourhood used when deciding whether two foreground pixels belong to the same component. |
| `Labels` | Labels is the result of connected-component labelling: a dense, row-major map assigning each pixel an integer label. |

<details>
<summary><code>Blob</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func AnalyzeBlobs(src *cv.Mat, conn Connectivity) []Blob` | AnalyzeBlobs labels src and returns a `Blob` descriptor for every foreground component, ordered by label. |
| `func (b Blob) AspectRatio() float64` | AspectRatio returns the bounding box aspect ratio (width divided by height), or 0 for a degenerate zero-height box. |
| `func (b Blob) Circularity() float64` | Circularity returns the isoperimetric shape factor 4*pi*Area/Perimeter^2, a value approaching 1 for a filled disc and tending to 0 for elongated or… |
| `func (b Blob) EquivalentDiameter() float64` | EquivalentDiameter returns the diameter of the circle whose area equals the blob's area, i.e. |
| `func (b Blob) Extent() float64` | Extent returns the ratio of blob area to bounding-box area, in (0, 1]. |

</details>

<details>
<summary><code>Component</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ComponentStats(src *cv.Mat, conn Connectivity) []Component` | ComponentStats labels src and returns the statistics of every foreground component, ordered by label. |
| `func LargestComponent(src *cv.Mat, conn Connectivity) (Component, bool)` | LargestComponent returns the component with the greatest area and true, or a zero Component and false when the image has no foreground. |
| `func SmallestComponent(src *cv.Mat, conn Connectivity) (Component, bool)` | SmallestComponent returns the component with the least area and true, or a zero Component and false when the image has no foreground. |
| `func (c Component) AspectRatio() float64` | AspectRatio returns the bounding box aspect ratio (width divided by height). |
| `func (c Component) Centroid() (x, y float64)` | Centroid returns the component's centre of mass as (x, y). |
| `func (c Component) Contains(x, y int) bool` | Contains reports whether pixel (x, y) lies inside the component's bounding box. |
| `func (c Component) Extent() float64` | Extent returns the ratio of the component's area to its bounding-box area, a value in (0, 1] measuring how completely the component fills its box. |
| `func (c Component) Height() int` | Height returns the height in pixels of the component's bounding box. |
| `func (c Component) Width() int` | Width returns the width in pixels of the component's bounding box. |

</details>

<details>
<summary><code>Labels</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Label(src *cv.Mat, conn Connectivity) *Labels` | Label assigns a unique label to every connected foreground region of a binary single-channel image using a two-pass union-find algorithm. |
| `func (l *Labels) Areas() []int` | Areas returns a slice of length Count+1 giving the pixel area of each label; element 0 is the background area and element k the area of label k. |
| `func (l *Labels) At(x, y int) int` | At returns the label of pixel (x, y). |
| `func (l *Labels) BoundingBox(label int) cv.Rect` | BoundingBox returns the tightest upright rectangle enclosing the pixels of the given label. |
| `func (l *Labels) Centroid(label int) (cx, cy float64)` | Centroid returns the area-weighted centre of mass (mean x, mean y) of the given label's pixels. |
| `func (l *Labels) ColorImage() *cv.Mat` | ColorImage renders the label map as a 3-channel RGB image, assigning each component a distinct, deterministic colour and painting the background… |
| `func (l *Labels) Component(label int) Component` | Component computes the `Component` statistics for a single label. |
| `func (l *Labels) Components() []Component` | Components computes `Component` statistics for every foreground label in a single pass and returns them ordered by label (element 0 is label 1). |
| `func (l *Labels) Mask(label int) *cv.Mat` | Mask returns a binary single-channel image in which pixels carrying the given label are 255 and all others 0. |
| `func (l *Labels) NumComponents() int` | NumComponents returns the number of foreground components, excluding the background. |
| `func (l *Labels) Size() (w, h int)` | Size returns the map dimensions as (width, height). |
| `func (l *Labels) ToMat() *cv.Mat` | ToMat renders the label map as a single-channel image, storing each label modulo 256. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/connected).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
