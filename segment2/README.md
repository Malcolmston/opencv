# segment2 — provides image-segmentation algorithms built on the standard-library-only OpenCV port

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/segment2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/segment2)

Package segment2 provides image-segmentation algorithms built on the
standard-library-only OpenCV port github.com/malcolmston/opencv (imported here
as cv). Every routine consumes and produces the library's native `cv.Mat`
image type; no competing image representation is introduced.

- Clustering in colour space: `KMeansSegment`, `QuantizeColors`,
`MeanShiftFilter` and `MeanShiftSegment`. - Marker-controlled flooding:
`Watershed`, `WatershedFromMarkers` and the `DistanceTransform` /
`GradientMagnitude` helpers used to build markers. - Region-based growing and
partitioning: `RegionGrow`, `SeededRegionGrow`, `SplitAndMerge`. - Graph-based
segmentation: `Felzenszwalb` minimum-spanning-forest. - Superpixels: `SLIC`
and `SLICO`. - Foreground extraction by graph cut: `GrabCut`, backed by a
genuine max-flow / min-cut (`MaxFlow`). - Boundary evolution: `ActiveContour`
parametric snakes. - Thresholded connected components: `ConnectedComponents`,
`ThresholdComponents` and the `ComponentStats` measurements.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/segment2"
```

## Usage

This is the package's own `ExampleComponentStats`, so it compiles and its output is
asserted on every `go test ./segment2/`.

```go
lm := segment2.ConnectedComponents(twoBlobs(), segment2.Conn8)

	for _, s := range segment2.ComponentStats(lm) {
		if s.Label == 0 {
			continue
		}
		fmt.Printf("label %d: area=%d box=%dx%d at (%d,%d) centroid=(%d,%d)\n",
			s.Label, s.Area, s.Rect.Width, s.Rect.Height, s.Rect.X, s.Rect.Y,
			s.Centroid.X, s.Centroid.Y)
	}
```

```
label 1: area=25 box=5x5 at (2,2) centroid=(4,4)
label 2: area=30 box=5x6 at (13,12) centroid=(15,15)
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ActiveContour(img *cv.Mat, initial []cv.Point, params ActiveContourParams) []cv.Point` | ActiveContour evolves a closed parametric snake (Kass, Witkin & Terzopoulos) toward image edges. |
| `func ContourArea(pts []cv.Point) float64` | ContourArea returns the absolute area enclosed by the ordered point ring using the shoelace formula. |
| `func ContourPerimeter(pts []cv.Point) float64` | ContourPerimeter returns the closed-polygon perimeter (summed edge lengths, including the closing edge) of the ordered point ring. |
| `func DistanceTransform(img *cv.Mat) []float64` | DistanceTransform returns, for every pixel of a binary single-channel image, the exact Euclidean distance to the nearest zero (background) pixel. |
| `func GrabCut(img *cv.Mat, rect cv.Rect, iterations int) *cv.Mat` | GrabCut segments the foreground object contained in rect from img. |
| `func GrabCutWithMask(img *cv.Mat, mask []byte, iterations, numComponents int) []byte` | GrabCutWithMask runs GrabCut foreground extraction driven by an initial mask. |
| `func GradientMagnitude(img *cv.Mat) []float64` | GradientMagnitude returns the Sobel gradient magnitude of img as a flat row-major []float64 in the image's own coordinate order. |
| `func LargestComponent(lm *LabelMap) int` | LargestComponent returns the label of the largest non-background component of lm, or 0 if there are no foreground pixels. |
| `func MeanShiftFilter(img *cv.Mat, spatialRadius int, colorRadius float64, maxIter int) *cv.Mat` | MeanShiftFilter is `MeanShiftFilterParams` with the scalar parameters spelled out; it smooths img with the given spatial radius, colour radius and… |
| `func MeanShiftFilterParams(img *cv.Mat, p MeanShiftParams) *cv.Mat` | MeanShiftFilterParams performs edge-preserving mean-shift smoothing of img in the joint spatial-range domain using the supplied parameters, returning… |
| `func OtsuThreshold(img *cv.Mat) float64` | OtsuThreshold computes the global intensity threshold that maximises between-class variance (Otsu's method) over the luminance histogram of img. |
| `func QuantizeColors(img *cv.Mat, k, iterations int) *cv.Mat` | QuantizeColors reduces img to k colours by k-means clustering and returns a new `cv.Mat` of the same shape in which every pixel is replaced by the… |
| `func RegionGrow(img *cv.Mat, seed cv.Point, tolerance float64, conn Connectivity) *cv.Mat` | RegionGrow grows a single connected region outward from seed, adding a neighbouring pixel whenever its colour lies within tolerance (Euclidean colour… |
| `func RegionGrowMean(img *cv.Mat, seed cv.Point, tolerance float64, conn Connectivity) *cv.Mat` | RegionGrowMean is like `RegionGrow` but tests each candidate pixel against the running mean colour of the region grown so far rather than the fixed… |
| `func Watershed(img *cv.Mat, markers []int) []int` | Watershed performs marker-controlled watershed segmentation with Meyer's priority-flooding algorithm over the gradient relief of img. |

### Types

| Type | What it is |
| --- | --- |
| `ActiveContourParams` | ActiveContourParams configures the `ActiveContour` parametric snake. |
| `ComponentStat` | ComponentStat summarises one connected component of a `LabelMap`. |
| `Connectivity` | Connectivity selects the pixel neighbourhood used by region-based routines. |
| `FlowGraph` | FlowGraph is a directed graph with real-valued edge capacities used for max-flow / min-cut computations. |
| `KMeansResult` | KMeansResult holds the outcome of colour-space k-means clustering. |
| `LabelMap` | LabelMap is a dense, row-major integer labelling of an image. |
| `MeanShiftParams` | MeanShiftParams configures the joint spatial-range mean-shift routines. |
| `SLICParams` | SLICParams configures the `SLICWithParams` superpixel segmenter. |

<details>
<summary><code>ActiveContourParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultActiveContourParams() ActiveContourParams` | DefaultActiveContourParams returns reasonable defaults (Alpha 0.1, Beta 0.1, Gamma 1.0, EdgeWeight 2.0, Sigma 1.0, Iterations 100). |

</details>

<details>
<summary><code>ComponentStat</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ComponentStats(lm *LabelMap) []ComponentStat` | ComponentStats measures every non-background label (1..NumLabels-1) of lm and returns their `ComponentStat` records ordered by label. |

</details>

<details>
<summary><code>FlowGraph</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewFlowGraph(n, source, sink int) *FlowGraph` | NewFlowGraph creates an empty flow graph with n vertices and the given source and sink. |
| `func (g *FlowGraph) AddEdge(u, v int, capacity, revCap float64)` | AddEdge adds a directed edge from u to v with the given forward capacity and a reverse edge of capacity revCap (use 0 for a purely directed edge, or… |
| `func (g *FlowGraph) MaxFlow() (float64, []bool)` | MaxFlow computes the maximum flow from source to sink with Dinic's algorithm and returns the flow value together with the min-cut: a boolean slice of… |

</details>

<details>
<summary><code>KMeansResult</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func KMeansSegment(img *cv.Mat, k, iterations int) *KMeansResult` | KMeansSegment clusters the pixels of img into k colour groups with deterministic Lloyd's k-means and returns a `KMeansResult` whose Labels give each… |

</details>

<details>
<summary><code>LabelMap</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ConnectedComponents(binary *cv.Mat, conn Connectivity) *LabelMap` | ConnectedComponents labels the connected non-zero (foreground) regions of a binary single-channel image with a two-pass union-find algorithm. |
| `func Felzenszwalb(img *cv.Mat, sigma, k float64, minSize int) *LabelMap` | Felzenszwalb segments img with the Felzenszwalb-Huttenlocher graph-based method — a minimum-spanning-forest over the 8-connected pixel grid whose… |
| `func FilterComponentsBySize(lm *LabelMap, minArea int) *LabelMap` | FilterComponentsBySize returns a copy of lm in which every non-background component smaller than minArea pixels is reset to the background label 0. |
| `func MeanShiftSegment(img *cv.Mat, p MeanShiftParams, minRegionSize int) *LabelMap` | MeanShiftSegment performs mean-shift segmentation: every pixel is driven to its mode in the joint spatial-range domain and pixels whose modes lie… |
| `func NewLabelMap(rows, cols int) *LabelMap` | NewLabelMap allocates a zero-filled LabelMap of the given size. |
| `func SLIC(img *cv.Mat, regionSize int, compactness float64, iterations int) *LabelMap` | SLIC is `SLICWithParams` with the fixed-compactness variant selected. |
| `func SLICO(img *cv.Mat, regionSize, iterations int) *LabelMap` | SLICO is the zero-parameter SLIC variant that adapts each cluster's compactness automatically, so only the region size need be chosen. |
| `func SLICWithParams(img *cv.Mat, p SLICParams) *LabelMap` | SLICWithParams computes superpixels with Simple Linear Iterative Clustering (Achanta et al., 2012) using the supplied `SLICParams`, returning a… |
| `func SeededRegionGrow(img *cv.Mat, seeds *LabelMap, conn Connectivity) *LabelMap` | SeededRegionGrow grows several regions simultaneously from labelled seeds, producing a full partition of the image. |
| `func SplitAndMerge(img *cv.Mat, threshold float64, minSize int) *LabelMap` | SplitAndMerge segments img with the classic quadtree split-and-merge method. |
| `func ThresholdComponents(img *cv.Mat, thresh float64, conn Connectivity) *LabelMap` | ThresholdComponents thresholds img at the given intensity (pixels with luminance greater than thresh become foreground) and returns the connected… |
| `func WatershedFromMarkers(img *cv.Mat, seeds *LabelMap) *LabelMap` | WatershedFromMarkers is a convenience wrapper over `Watershed` that takes and returns a `LabelMap`. |
| `func (lm *LabelMap) At(y, x int) int` | At returns the label of pixel (x, y). |
| `func (lm *LabelMap) BoundaryMask() *cv.Mat` | BoundaryMask returns a single-channel `cv.Mat` whose pixels are 255 wherever a pixel's label differs from that of its right or bottom neighbour, i.e. |
| `func (lm *LabelMap) BoundingRects() []cv.Rect` | BoundingRects returns the axis-aligned bounding rectangle of every label slot, indexed by label. |
| `func (lm *LabelMap) Clone() *LabelMap` | Clone returns a deep copy of the LabelMap. |
| `func (lm *LabelMap) Colorize() *cv.Mat` | Colorize renders the labelling as a three-channel `cv.Mat`, assigning each label a distinct colour from a fixed palette (cycled if there are more… |
| `func (lm *LabelMap) Compact() int` | Compact renumbers the labels so the distinct values present become the contiguous range 0..k-1, preserving first-appearance order in row-major scan. |
| `func (lm *LabelMap) CountRegions() int` | CountRegions returns the number of distinct non-negative labels present. |
| `func (lm *LabelMap) InBounds(y, x int) bool` | InBounds reports whether pixel (x, y) lies inside the labelling. |
| `func (lm *LabelMap) MeanColors(img *cv.Mat) [][]float64` | MeanColors returns the mean colour of every label slot over img, each a channels-length slice, indexed by label. |
| `func (lm *LabelMap) RegionCentroids() []cv.Point` | RegionCentroids returns the (x, y) centroid of every label slot as a `cv.Point` with rounded integer coordinates, indexed by label. |
| `func (lm *LabelMap) RegionMask(label int) *cv.Mat` | RegionMask returns a single-channel `cv.Mat` mask that is 255 where the labelling equals label and 0 elsewhere. |
| `func (lm *LabelMap) RegionSizes() []int` | RegionSizes returns the pixel count of each label slot, indexed by label. |
| `func (lm *LabelMap) Set(y, x, label int)` | Set stores label at pixel (x, y). |

</details>

<details>
<summary><code>MeanShiftParams</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultMeanShiftParams() MeanShiftParams` | DefaultMeanShiftParams returns reasonable defaults (SpatialRadius 7, ColorRadius 20, MaxIter 5, Epsilon 1). |

</details>

### Constants

`GCBackground`, `GCForeground`, `GCProbablyBackground`, `GCProbablyForeground`, `WatershedLine`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/segment2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
