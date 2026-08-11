# morph2 — standard-library-only toolkit of advanced mathematical morphology built on top

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/morph2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/morph2)

Package morph2 is a standard-library-only toolkit of advanced mathematical
morphology built on top of the parent package's
`github.com/malcolmston/opencv.Mat` image type.

It complements the elementary erosion/dilation offered by the root package
with the harder machinery of grey-scale and binary morphology: compound
operators (opening, closing, gradients, top-/black-hat), the hit-or-miss
transform, Zhang-Suen and Guo-Hall thinning, the morphological skeleton,
grey-scale morphological reconstruction and its many derivatives (hole
filling, border clearing, opening/closing by reconstruction, regional and
extended extrema, h-maxima/minima, minima imposition), exact and chamfer
distance transforms, marker-controlled watershed segmentation and granulometry
/ pattern spectra.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/morph2"
```

## Usage

This is the package's own `ExampleDilate`, so it compiles and its output is
asserted on every `go test ./morph2/`.

```go
src := mat(
		".....",
		".....",
		"..#..",
		".....",
		".....",
	)
	render(morph2.Dilate(src, morph2.CrossElement(3, 3)))
```

```
.....
..#..
.###.
..#..
.....
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func Binarize(src *cv.Mat, thresh uint8) *cv.Mat` | Binarize returns a binary image in which every input sample strictly greater than thresh becomes foreground (255) and every other sample becomes… |
| `func BlackHat(src *cv.Mat, e *Element) *cv.Mat` | BlackHat returns the black top-hat, the closing minus the source, which isolates dark details smaller than the structuring element. |
| `func ClearBorder(src *cv.Mat, conn Connectivity) *cv.Mat` | ClearBorder removes connected foreground components that touch the image border, leaving interior components untouched. |
| `func Close(src *cv.Mat, e *Element) *cv.Mat` | Close performs a morphological closing (dilation followed by erosion). |
| `func CloseByReconstruction(src *cv.Mat, e *Element, conn Connectivity) *cv.Mat` | CloseByReconstruction returns the closing by reconstruction of src: a dilation by e used as the marker, reconstructed by erosion under src as the… |
| `func Complement(src *cv.Mat) *cv.Mat` | Complement returns the pointwise complement 255 - src of a grey-scale image; for a binary image this swaps foreground and background. |
| `func CountForeground(src *cv.Mat) int` | CountForeground returns the number of non-zero (foreground) samples. |
| `func Difference(a, b *cv.Mat) *cv.Mat` | Difference returns the binary set difference a \ b: foreground where a is foreground and b is background. |
| `func Dilate(src *cv.Mat, e *Element) *cv.Mat` | Dilate performs a single grey-scale dilation: each output sample is the maximum input sample over the reflected structuring element. |
| `func DilateN(src *cv.Mat, e *Element, iterations int) *cv.Mat` | DilateN applies `Dilate` the given number of times (values < 1 are treated as 1). |
| `func Equal(a, b *cv.Mat) bool` | Equal reports whether two images have identical dimensions and identical samples. |
| `func Erode(src *cv.Mat, e *Element) *cv.Mat` | Erode performs a single grey-scale erosion: each output sample is the minimum input sample over the set cells of the structuring element. |
| `func ErodeN(src *cv.Mat, e *Element, iterations int) *cv.Mat` | ErodeN applies `Erode` the given number of times (values < 1 are treated as 1). |
| `func ExtendedMaxima(src *cv.Mat, h uint8, conn Connectivity) *cv.Mat` | ExtendedMaxima returns the regional maxima of the h-maxima transform of src, i.e. |
| `func ExtendedMinima(src *cv.Mat, h uint8, conn Connectivity) *cv.Mat` | ExtendedMinima returns the regional minima of the h-minima transform of src, i.e. |
| `func ExternalGradient(src *cv.Mat, e *Element) *cv.Mat` | ExternalGradient returns the external morphological gradient, the dilation minus the source, marking the outer boundary of bright objects. |
| `func FillHoles(src *cv.Mat, conn Connectivity) *cv.Mat` | FillHoles fills holes — background regions not connected to the image border — in a grey-scale or binary image. |
| `func FindBranchPoints(src *cv.Mat) *cv.Mat` | FindBranchPoints returns a binary image marking the branch points of a one-pixel-wide skeleton: foreground pixels with three or more foreground… |
| `func FindEndpoints(src *cv.Mat) *cv.Mat` | FindEndpoints returns a binary image marking the endpoints of a one-pixel-wide skeleton: foreground pixels with exactly one foreground neighbour. |
| `func Gradient(src *cv.Mat, e *Element) *cv.Mat` | Gradient returns the morphological gradient, the dilation minus the erosion, which highlights object boundaries. |
| `func GuoHallThinning(src *cv.Mat) *cv.Mat` | GuoHallThinning reduces the foreground of a binary image to a one-pixel-wide skeleton using the Guo-Hall (1989) parallel thinning algorithm, which… |
| `func HMaxima(src *cv.Mat, h uint8, conn Connectivity) *cv.Mat` | HMaxima suppresses all bright domes of dynamic (height) less than h by reconstructing src from src minus h. |
| `func HMinima(src *cv.Mat, h uint8, conn Connectivity) *cv.Mat` | HMinima fills all dark basins of depth less than h by reconstructing src (by erosion) from src plus h. |
| `func HitOrMiss(src *cv.Mat, hit, miss *Element) *cv.Mat` | HitOrMiss computes the binary hit-or-miss transform of src using two disjoint structuring elements: hit specifies cells that must lie on foreground… |
| `func ImposeMinima(src, markerMask *cv.Mat, conn Connectivity) *cv.Mat` | ImposeMinima forces regional minima of value 0 at every non-zero pixel of the binary markerMask and raises the rest of src so that no other minima… |
| `func InternalGradient(src *cv.Mat, e *Element) *cv.Mat` | InternalGradient returns the internal morphological gradient, the source minus its erosion, marking the inner boundary of bright objects. |
| `func Intersection(a, b *cv.Mat) *cv.Mat` | Intersection returns the pointwise minimum of two identically sized images; for binary images this is the set intersection. |
| `func MorphologyEx(src *cv.Mat, e *Element, op Op, iterations int) *cv.Mat` | MorphologyEx dispatches to the compound operator named by op, applying the given number of iterations to the primitive erosion/dilation stages… |
| `func Open(src *cv.Mat, e *Element) *cv.Mat` | Open performs a morphological opening (erosion followed by dilation). |
| `func OpenByReconstruction(src *cv.Mat, e *Element, conn Connectivity) *cv.Mat` | OpenByReconstruction returns the opening by reconstruction of src: an erosion by e used as the marker, reconstructed under src as the mask. |
| `func Prune(src *cv.Mat, iterations int) *cv.Mat` | Prune removes spur pixels (short branches) from a one-pixel-wide skeleton by iteratively deleting endpoints — foreground pixels with a single… |
| `func ReconstructByDilation(marker, mask *cv.Mat, conn Connectivity) *cv.Mat` | ReconstructByDilation performs grey-scale morphological reconstruction of the mask from the marker by iterated geodesic dilation until stability. |
| `func ReconstructByErosion(marker, mask *cv.Mat, conn Connectivity) *cv.Mat` | ReconstructByErosion performs grey-scale morphological reconstruction by erosion: the dual of `ReconstructByDilation`. |
| `func RegionalMaxima(src *cv.Mat, conn Connectivity) *cv.Mat` | RegionalMaxima returns a binary image marking the regional maxima of src: connected plateaus whose neighbours are all strictly lower. |
| `func RegionalMinima(src *cv.Mat, conn Connectivity) *cv.Mat` | RegionalMinima returns a binary image marking the regional minima of src: connected plateaus whose neighbours are all strictly higher. |
| `func Skeleton(src *cv.Mat, e *Element) *cv.Mat` | Skeleton computes the morphological skeleton of a binary image by Lantuejoul's method: the union over successive erosions f⊖nB of the residual… |
| `func Subtract(a, b *cv.Mat) *cv.Mat` | Subtract returns the per-sample saturating difference a - b, clamped at zero. |
| `func TopHat(src *cv.Mat, e *Element) *cv.Mat` | TopHat returns the white top-hat, the source minus its opening, which isolates bright details smaller than the structuring element. |
| `func Union(a, b *cv.Mat) *cv.Mat` | Union returns the pointwise maximum of two identically sized images; for binary images this is the set union. |
| `func ZhangSuenThinning(src *cv.Mat) *cv.Mat` | ZhangSuenThinning reduces the foreground of a binary image to a one-pixel-wide skeleton using the Zhang-Suen (1984) parallel thinning algorithm. |

### Types

| Type | What it is |
| --- | --- |
| `Connectivity` | Connectivity selects the neighbourhood used by binary and reconstruction routines. |
| `DistanceMask` | DistanceMask selects the neighbourhood mask used to approximate a distance transform. |
| `DistanceType` | DistanceType selects the metric approximated by a distance transform. |
| `Element` | Element is a flat structuring element: a rectangular grid of boolean cells with an explicit anchor. |
| `FloatGrid` | FloatGrid is a dense row-major grid of float64 values used to return real-valued morphological results such as distance maps. |
| `LabelMap` | LabelMap is a dense row-major grid of integer labels used for markers and watershed segmentation. |
| `Op` | Op selects the compound operation performed by `MorphologyEx`. |
| `PatternSpectrum` | PatternSpectrum holds the result of a granulometric analysis: how the image volume is distributed across structuring-element sizes. |
| `Shape` | Shape selects the geometry of a structuring element built by `NewElement`. |

<details>
<summary><code>Element</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CrossElement(rows, cols int) *Element` | CrossElement returns a cross-shaped structuring element of the given size. |
| `func DiamondElement(radius int) *Element` | DiamondElement returns a Manhattan (L1) disk of the given radius; the element is a square of side 2*radius+1 anchored at its centre. |
| `func DiskElement(radius int) *Element` | DiskElement returns a Euclidean disk of the given radius; the element is a square of side 2*radius+1 anchored at its centre. |
| `func ElementFromMat(kernel *cv.Mat, anchorY, anchorX int) *Element` | ElementFromMat builds a structuring element from a single-channel `cv.Mat` kernel: a cell is set where the corresponding sample is non-zero. |
| `func EllipseElement(rows, cols int) *Element` | EllipseElement returns an elliptical structuring element of the given size. |
| `func NewElement(shape Shape, rows, cols int) *Element` | NewElement builds a structuring element of the given shape and size. |
| `func RectElement(rows, cols int) *Element` | RectElement returns a solid rectangular structuring element of the given size. |
| `func (e *Element) At(y, x int) bool` | At reports whether the cell at (y, x) is set. |
| `func (e *Element) Clone() *Element` | Clone returns an independent deep copy of the element. |
| `func (e *Element) Count() int` | Count returns the number of set cells in the element. |
| `func (e *Element) Reflect() *Element` | Reflect returns the point reflection of the element through its anchor, i.e. |
| `func (e *Element) Set(y, x int, on bool)` | Set marks the cell at (y, x) as set or unset. |
| `func (e *Element) ToMat() *cv.Mat` | ToMat converts the element to a single-channel `cv.Mat` whose set cells hold 1 and unset cells hold 0, matching the root package's… |

</details>

<details>
<summary><code>FloatGrid</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ChamferDistance(src *cv.Mat, a, b float64) *FloatGrid` | ChamferDistance computes a chamfer distance transform with caller-supplied 3x3 weights: a for orthogonal steps and b for diagonal steps. |
| `func DistanceTransform(src *cv.Mat, dt DistanceType, mask DistanceMask) *FloatGrid` | DistanceTransform computes, for every foreground (non-zero) pixel, the distance to the nearest background (zero) pixel under the chosen metric and… |
| `func DistanceTransformChebyshev(src *cv.Mat) *FloatGrid` | DistanceTransformChebyshev is shorthand for the exact chessboard distance transform. |
| `func DistanceTransformExact(src *cv.Mat) *FloatGrid` | DistanceTransformExact is shorthand for the exact Euclidean distance transform (Felzenszwalb-Huttenlocher). |
| `func DistanceTransformL1(src *cv.Mat) *FloatGrid` | DistanceTransformL1 is shorthand for the exact city-block distance transform. |
| `func NewFloatGrid(rows, cols int) *FloatGrid` | NewFloatGrid allocates a zero-filled grid of the given size. |
| `func (g *FloatGrid) At(y, x int) float64` | At returns the value at (y, x). |
| `func (g *FloatGrid) Max() float64` | Max returns the largest value in the grid, or 0 for an empty grid. |
| `func (g *FloatGrid) Min() float64` | Min returns the smallest value in the grid, or 0 for an empty grid. |
| `func (g *FloatGrid) Set(y, x int, value float64)` | Set stores value at (y, x). |
| `func (g *FloatGrid) ToMat(normalize bool) *cv.Mat` | ToMat converts the grid to a single-channel `cv.Mat`. |

</details>

<details>
<summary><code>LabelMap</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ConnectedComponentMarkers(src *cv.Mat, conn Connectivity) *LabelMap` | ConnectedComponentMarkers labels the connected foreground (non-zero) components of a binary image with consecutive positive integers starting at 1,… |
| `func NewLabelMap(rows, cols int) *LabelMap` | NewLabelMap allocates a zero-filled label map of the given size. |
| `func Watershed(image *cv.Mat, markers *LabelMap, conn Connectivity) *LabelMap` | Watershed performs marker-controlled watershed segmentation of a grey-scale image using Meyer's flooding algorithm. |
| `func (l *LabelMap) At(y, x int) int32` | At returns the label at (y, x). |
| `func (l *LabelMap) Clone() *LabelMap` | Clone returns an independent deep copy of the label map. |
| `func (l *LabelMap) NumLabels() int` | NumLabels returns the largest positive label present, which for a dense 1..N labelling equals the number of regions. |
| `func (l *LabelMap) Ridges() *cv.Mat` | Ridges returns a binary image marking the watershed line pixels (label `WatershedRidge`). |
| `func (l *LabelMap) Set(y, x int, label int32)` | Set stores label at (y, x). |
| `func (l *LabelMap) ToMat() *cv.Mat` | ToMat renders the label map as a single-channel `cv.Mat` for visualisation: background (0) maps to 0, watershed ridges map to 255, and each positive… |

</details>

<details>
<summary><code>PatternSpectrum</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Granulometry(src *cv.Mat, shape Shape, maxRadius, step int) *PatternSpectrum` | Granulometry performs a granulometric analysis of a grey-scale or binary image by opening it with structuring elements of the given shape and… |
| `func (ps *PatternSpectrum) Mean() float64` | Mean returns the volume-weighted mean structuring-element size of the pattern spectrum, a scalar summary of the characteristic object size in the… |
| `func (ps *PatternSpectrum) Volume() float64` | Volume returns the total sample volume of the unopened source (Volumes[0]). |

</details>

### Constants

`WatershedRidge`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/morph2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
