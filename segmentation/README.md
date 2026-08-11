# segmentation — provides region-based image segmentation built on the stdlib-only OpenCV port

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/segmentation.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/segmentation)

Package segmentation provides region-based image segmentation built on the
stdlib-only OpenCV port github.com/malcolmston/opencv (imported here as cv).

- `FloodFill` grows a connected region from a seed point, adding pixels whose
colour lies within a tolerance of an already-filled neighbour (OpenCV's
floating-range flood fill). It reports the pixel count and the bounding
rectangle of the filled region and, like cv2.floodFill, mutates the input
image in place. - `Watershed` performs marker-controlled watershed
segmentation using Meyer's priority-flooding algorithm. Basins grow outward
from labelled markers in order of increasing image-gradient magnitude, driven
by a min-heap (container/heap). Pixels where two basins meet form the
watershed line. - `GrabCut` separates foreground from background with an
iterative Gaussian-mixture colour model. Each iteration re-fits per-class GMMs
and relabels the soft-labelled pixels. The global min-cut of the classic
GrabCut energy is approximated by iterated conditional modes (ICM): a
per-pixel MAP data term combined with a Potts smoothness term over the
4-neighbourhood. See `GrabCut` for the details of this approximation. -
`MeanShiftFiltering` and `PyrMeanShiftFiltering` perform edge-preserving
mean-shift smoothing in the joint spatial-range domain, collapsing each pixel
onto the mode of its local colour distribution.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/segmentation"
```

## Usage

This is the package's own `ExampleDistanceTransformWatershed`, so it compiles and its output is
asserted on every `go test ./segmentation/`.

```go
m := cv.NewMat(24, 30, 1)
	disc := func(cx, cy, r int) {
		for y := cy - r; y <= cy+r; y++ {
			for x := cx - r; x <= cx+r; x++ {
				if x >= 0 && x < 30 && y >= 0 && y < 24 && (x-cx)*(x-cx)+(y-cy)*(y-cy) <= r*r {
					m.Set(y, x, 0, 255)
				}
			}
		}
	}
	disc(8, 12, 6)
	disc(18, 12, 6)
	lm := segmentation.DistanceTransformWatershed(m, 0.7)

	fmt.Println(lm.Count, lm.At(8, 12) != lm.At(18, 12))
```

```
3 true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func DistanceTransform(binary *cv.Mat) []float64` | DistanceTransform returns the exact Euclidean distance from every foreground pixel of binary to the nearest background pixel, as a flat row-major… |
| `func FloodFill(img *cv.Mat, seed cv.Point, newVal cv.Scalar, loDiff, hiDiff cv.Scalar, connectivity int) (count int, rect cv.Rect)` | FloodFill fills a connected region of img, starting at seed, with newVal and returns the number of pixels filled together with the bounding rectangle… |
| `func GrabCut(img *cv.Mat, mask *cv.Mat, rect cv.Rect, iters int) *cv.Mat` | GrabCut segments the three-channel image img into foreground and background using an iterative Gaussian-mixture colour model, returning a new… |
| `func MeanShiftFiltering(img *cv.Mat, sp, sr float64) *cv.Mat` | MeanShiftFiltering performs edge-preserving mean-shift smoothing of the three-channel image img and returns a new Mat of the same size, mirroring a… |
| `func MultiOtsu(img *cv.Mat, classes int) []int` | MultiOtsu computes classes-1 grey-level thresholds that partition the intensity histogram of img into classes groups by maximising the between-class… |
| `func MultiOtsuThreshold(img *cv.Mat, classes int) (*cv.Mat, []int)` | MultiOtsuThreshold applies `MultiOtsu` and returns both the chosen thresholds and a single-channel `cv.Mat` whose pixels hold the class index (0 for… |
| `func PyrMeanShiftFiltering(img *cv.Mat, sp, sr float64, maxLevel int) *cv.Mat` | PyrMeanShiftFiltering applies mean-shift smoothing over a Gaussian pyramid, the multi-scale variant of `MeanShiftFiltering` modelled on… |
| `func SelectiveSearchSegmentation(img *cv.Mat, sigma, k float64, minSize int) []cv.Rect` | SelectiveSearchSegmentation generates object-region proposals from img using the selective-search strategy of Uijlings et al. |
| `func Watershed(img *cv.Mat, markers *cv.Mat) *cv.Mat` | Watershed performs marker-controlled watershed segmentation of img using the seed labels in markers, returning a new single-channel `cv.Mat` of… |

### Types

| Type | What it is |
| --- | --- |
| `IntelligentScissors` | IntelligentScissors implements the live-wire boundary tool of Mortensen & Barrett ("Intelligent Scissors for Image Composition", 1995), matching the… |
| `LabelMap` | LabelMap is a dense, per-pixel region labelling produced by the region-based segmenters in this package. |
| `RAG` | RAG is a Region Adjacency Graph: the nodes are the regions of a `LabelMap` and the edges connect regions that touch, weighted by the Euclidean… |

<details>
<summary><code>IntelligentScissors</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewIntelligentScissors(img *cv.Mat) *IntelligentScissors` | NewIntelligentScissors builds the cost field of img for live-wire tracing. |
| `func (s *IntelligentScissors) BuildMap(seed cv.Point)` | BuildMap runs Dijkstra from seed over the 8-connected pixel graph, recording the minimum-cost predecessor of every reachable pixel so that any number… |
| `func (s *IntelligentScissors) Trace(target cv.Point) []cv.Point` | Trace returns the minimum-cost contour from the current seed (set by `BuildMap`) to target, as an ordered slice of points running seed -> target. |

</details>

<details>
<summary><code>LabelMap</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DistanceTransformWatershed(binary *cv.Mat, seedRatio float64) *LabelMap` | DistanceTransformWatershed splits the foreground of a binary mask into separate regions using distance-transform-seeded watershed, the standard… |
| `func EfficientGraphSegmentation(img *cv.Mat, sigma, k float64, minSize int) *LabelMap` | EfficientGraphSegmentation segments img with the Felzenszwalb-Huttenlocher graph-based algorithm ("Efficient Graph-Based Image Segmentation", 2004),… |
| `func KMeansSegmentation(img *cv.Mat, k, iterations int) (*LabelMap, [][]float64)` | KMeansSegmentation clusters the pixels of img into k colour groups with deterministic Lloyd's k-means and returns a `LabelMap` whose label is the… |
| `func MeanShiftSegmentation(img *cv.Mat, sp, sr float64, minSize int) *LabelMap` | MeanShiftSegmentation turns the edge-preserving smoothing of `MeanShiftFiltering` into a full region labelling, the segmentation stage that follows… |
| `func RegionGrowing(img *cv.Mat, seeds []cv.Point, threshold float64) *LabelMap` | RegionGrowing performs seeded region growing on img: each seed point starts a region and neighbouring pixels are absorbed while their colour stays… |
| `func SLIC(img *cv.Mat, regionSize int, compactness float64, iterations int) *LabelMap` | SLIC computes superpixels with the Simple Linear Iterative Clustering algorithm of Achanta et al. |
| `func (l *LabelMap) At(x, y int) int` | At returns the region label of pixel (x, y). |
| `func (l *LabelMap) BoundaryMask() *cv.Mat` | BoundaryMask returns a single-channel `cv.Mat` whose pixels are 255 on region boundaries and 0 elsewhere. |
| `func (l *LabelMap) BoundingRects() []cv.Rect` | BoundingRects returns the tight bounding rectangle of every region, indexed by label (length Count). |
| `func (l *LabelMap) Colorize() *cv.Mat` | Colorize renders the labelling as a three-channel `cv.Mat`, assigning each region a distinct, deterministic pseudo-random colour. |
| `func (l *LabelMap) RegionSizes() []int` | RegionSizes returns the pixel count of every region, indexed by label. |

</details>

<details>
<summary><code>RAG</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BuildRAG(lm *LabelMap, img *cv.Mat) *RAG` | BuildRAG constructs the region adjacency graph of lm over the colours of img. |
| `func (r *RAG) EdgeWeight(a, b int) float64` | EdgeWeight returns the Euclidean distance between the mean colours of regions a and b, the weight used for merging. |
| `func (r *RAG) MeanColor(i int) []float64` | MeanColor returns a copy of the mean colour of region i. |
| `func (r *RAG) MergeByColor(threshold float64) *LabelMap` | MergeByColor greedily merges adjacent regions whose mean-colour distance is at most threshold, the classic RAG threshold-merging step. |
| `func (r *RAG) MergeBySize(minSize int) *LabelMap` | MergeBySize merges every region smaller than minSize pixels into the adjacent region whose mean colour is closest, repeating until no region below… |
| `func (r *RAG) Neighbors(i int) []int` | Neighbors returns the labels of the regions adjacent to region i, in ascending order. |
| `func (r *RAG) Regions() int` | Regions returns the number of regions (nodes) in the graph. |

</details>

### Constants

`GcBgd`, `GcFgd`, `GcPrBgd`, `GcPrFgd`, `WatershedMarker`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/segmentation).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
