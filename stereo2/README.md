# stereo2 — self-contained, standard-library-only toolkit for binocular stereo vision and depth

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/stereo2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/stereo2)

Package stereo2 is a self-contained, standard-library-only toolkit for
binocular stereo vision and depth reconstruction, built on top of the core
`github.com/malcolmston/opencv` package.

- matching cost construction (SAD/SSD/NCC block matching, census-based
matching) and the raw per-pixel cost volume (`CostVolume`); - cost
aggregation, both local box aggregation (`AggregateBoxFilter`) and the
dynamic-programming path aggregation of semi-global matching
(`SemiGlobalAggregate`, `SGMMatcher`); - disparity extraction with
winner-take-all and uniqueness testing; - disparity refinement — left/right
consistency checking (`LeftRightCheck`), parabola sub-pixel interpolation
(`RefineSubpixelParabola`), speckle removal (`SpeckleFilter`), median
smoothing and invalid-pixel filling; - geometry — depth from disparity
(`DepthFromDisparity`), reprojection to a metric 3-D point cloud
(`PointCloudFromDepth`, `ReprojectImageTo3D`) and least-squares / RANSAC plane
fitting in the reconstructed cloud (`FitPlane`, `FitPlaneRANSAC`).

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/stereo2"
```

## Usage

This is the package's own `ExampleBlockMatchSAD`, so it compiles and its output is
asserted on every `go test ./stereo2/`.

```go
left, right := stereoPair(24, 40, 5)

	disp := stereo2.BlockMatchSAD(left, right, 0, 12, 5)
	fmt.Printf("%dx%d, disparity at (12,20) = %.0f\n", disp.Cols, disp.Rows, disp.At(12, 20))
```

```
40x24, disparity at (12,20) = 5
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func EightPaths() [][2]int` | EightPaths returns the eight directions (the four of `FourPaths` plus the four diagonals) used by `SemiGlobalAggregate` for the full SGM… |
| `func FourPaths() [][2]int` | FourPaths returns the four axis-aligned scanline directions (left-to-right, right-to-left, top-to-bottom, bottom-to-top) used by… |
| `func HammingDistance(a, b uint64) int` | HammingDistance returns the number of differing bits between two census signatures, i.e. |
| `func QMatrix(cx, cy, f, baseline float64) [4][4]float64` | QMatrix builds the 4x4 disparity-to-depth reprojection matrix Q for a rectified stereo pair with principal point (cx, cy), focal length f (pixels)… |

### Types

| Type | What it is |
| --- | --- |
| `BlockMatcher` | BlockMatcher is a local ("winner-take-all") block matcher, the counterpart to OpenCV's cv::StereoBM. |
| `Camera` | Camera holds the pinhole intrinsics needed to back-project image pixels into 3-D: the focal lengths Fx, Fy (in pixels) and the principal point (Cx,… |
| `CensusMap` | CensusMap holds the census-transform signature of every pixel of an image. |
| `CensusMatcher` | CensusMatcher is a stereo matcher that scores disparities by aggregated Hamming distance between census signatures. |
| `CostVolume` | CostVolume is a dense disparity-space image (DSI): for every pixel it holds the matching cost of each candidate disparity. |
| `DepthMap` | DepthMap is a single-channel, row-major map of metric depths (typically in the same length unit as the stereo baseline) held as float32 values. |
| `DisparityMap` | DisparityMap is a single-channel, row-major map of sub-pixel disparities held as float32 values. |
| `MatchCost` | MatchCost selects the pointwise window dissimilarity measure used by a `BlockMatcher`. |
| `Plane` | Plane is an oriented plane in 3-D space, the set of points satisfying A*x + B*y + C*z + D = 0. |
| `Point3D` | Point3D is a point (or vector) in 3-D space using float64 coordinates. |
| `PointCloud` | PointCloud is an unordered set of reconstructed 3-D points, optionally carrying a per-point RGB colour sampled from a reference image. |
| `SGMMatcher` | SGMMatcher is a semi-global matcher. |

<details>
<summary><code>BlockMatcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBlockMatcher(minDisparity, numDisparities, blockSize int) *BlockMatcher` | NewBlockMatcher returns a BlockMatcher with the given search range and window size and SAD cost. |
| `func (bm *BlockMatcher) Compute(left, right *cv.Mat) *DisparityMap` | Compute matches left against right and returns the left-referenced disparity map: for each left pixel it searches the corresponding right pixel at… |
| `func (bm *BlockMatcher) ComputeRight(left, right *cv.Mat) *DisparityMap` | ComputeRight matches right against left and returns the right-referenced disparity map: for each right pixel it searches the corresponding left pixel… |

</details>

<details>
<summary><code>Camera</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewCamera(fx, fy, cx, cy float64) Camera` | NewCamera returns a Camera with the given focal lengths and principal point. |

</details>

<details>
<summary><code>CensusMap</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CensusTransform(img *cv.Mat, windowW, windowH int) *CensusMap` | CensusTransform computes the census signature of every pixel using a windowW-by-windowH neighbourhood (both rounded up to odd). |
| `func (c *CensusMap) At(y, x int) uint64` | At returns the census signature at row y, column x. |
| `func (c *CensusMap) Size() (rows, cols int)` | Size returns the map dimensions as (rows, cols). |

</details>

<details>
<summary><code>CensusMatcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewCensusMatcher(minDisparity, numDisparities, censusW, censusH, aggRadius int) *CensusMatcher` | NewCensusMatcher returns a CensusMatcher with the given search range, a censusW-by-censusH census window and the given aggregation radius. |
| `func (cm *CensusMatcher) Compute(left, right *cv.Mat) *DisparityMap` | Compute matches left against right and returns the left-referenced disparity map. |
| `func (cm *CensusMatcher) ComputeRight(left, right *cv.Mat) *DisparityMap` | ComputeRight matches right against left and returns the right-referenced disparity map for use with `LeftRightCheck`. |

</details>

<details>
<summary><code>CostVolume</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func AggregateBoxFilter(vol *CostVolume, radius int) *CostVolume` | AggregateBoxFilter sums the matching cost of each disparity candidate over a square (2*radius+1) window, turning a per-pixel cost volume into an… |
| `func BuildCensusCostVolume(left, right *CensusMap, minDisparity, numDisparities int) *CostVolume` | BuildCensusCostVolume constructs a per-pixel matching cost volume whose cost is the Hamming distance between census signatures. |
| `func BuildCostVolume(left, right *cv.Mat, minDisparity, numDisparities int, squared bool) *CostVolume` | BuildCostVolume constructs the raw, un-aggregated per-pixel matching cost volume for a rectified stereo pair. |
| `func NewCostVolume(rows, cols, disparities, minDisparity int) *CostVolume` | NewCostVolume allocates a zero-filled cost volume. |
| `func SemiGlobalAggregate(vol *CostVolume, p1, p2 float32, dirs [][2]int) *CostVolume` | SemiGlobalAggregate performs Hirschmüller's semi-global cost aggregation. |
| `func (v *CostVolume) ArgMin(y, x int) (k int, cost float32)` | ArgMin returns the candidate disparity index with the smallest cost at pixel (y, x) together with that cost. |
| `func (v *CostVolume) At(y, x, k int) float32` | At returns the cost of candidate disparity index k at pixel (y, x). |
| `func (v *CostVolume) Dims() (rows, cols, disparities int)` | Dims returns the volume dimensions as (rows, cols, disparities). |
| `func (v *CostVolume) Set(y, x, k int, c float32)` | Set stores the cost c for candidate disparity index k at pixel (y, x). |
| `func (v *CostVolume) ToDisparity(uniquenessRatio int) *DisparityMap` | ToDisparity extracts an integer winner-take-all disparity map from the volume. |

</details>

<details>
<summary><code>DepthMap</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DepthFromDisparity(disp *DisparityMap, focalLength, baseline float64) *DepthMap` | DepthFromDisparity converts a disparity map to a metric depth map using the standard relation Z = focalLength * baseline / d, valid for a rectified… |
| `func NewDepthMap(rows, cols int) *DepthMap` | NewDepthMap allocates a DepthMap whose samples are all `InvalidDepth`. |
| `func (m *DepthMap) At(y, x int) float32` | At returns the depth at row y, column x. |
| `func (m *DepthMap) Clone() *DepthMap` | Clone returns a deep copy of the map. |
| `func (m *DepthMap) MinMax() (min, max float32, ok bool)` | MinMax returns the smallest and largest genuine depths and whether any valid pixel was found. |
| `func (m *DepthMap) Set(y, x int, v float32)` | Set stores the depth v at row y, column x. |
| `func (m *DepthMap) Size() (rows, cols int)` | Size returns the map dimensions as (rows, cols). |
| `func (m *DepthMap) ToMat(near, far float32) *cv.Mat` | ToMat renders the depth map to a viewable single-channel 8-bit github.com/malcolmston/opencv.Mat. |
| `func (m *DepthMap) Valid(y, x int) bool` | Valid reports whether the pixel at row y, column x holds a genuine (positive) depth. |

</details>

<details>
<summary><code>DisparityMap</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BlockMatchNCC(left, right *cv.Mat, minDisparity, numDisparities, blockSize int) *DisparityMap` | BlockMatchNCC is a convenience wrapper that runs a normalized-cross-correlation `BlockMatcher` over the pair. |
| `func BlockMatchSAD(left, right *cv.Mat, minDisparity, numDisparities, blockSize int) *DisparityMap` | BlockMatchSAD is a convenience wrapper that runs a sum-of-absolute-differences `BlockMatcher` over the pair. |
| `func BlockMatchSSD(left, right *cv.Mat, minDisparity, numDisparities, blockSize int) *DisparityMap` | BlockMatchSSD is a convenience wrapper that runs a sum-of-squared-differences `BlockMatcher` over the pair. |
| `func DisparityFromDepth(depth *DepthMap, focalLength, baseline float64) *DisparityMap` | DisparityFromDepth is the inverse of `DepthFromDisparity`: it converts a metric depth map back to a disparity map with d = focalLength * baseline /… |
| `func FillInvalidHorizontal(disp *DisparityMap) *DisparityMap` | FillInvalidHorizontal fills each invalid pixel by propagating disparities horizontally: it takes the nearest valid disparity to the left and to the… |
| `func LeftRightCheck(left, right *DisparityMap, maxDiff float32) *DisparityMap` | LeftRightCheck enforces left/right consistency between a left-referenced disparity map and a right-referenced one. |
| `func MedianFilterDisparity(disp *DisparityMap, radius int) *DisparityMap` | MedianFilterDisparity replaces each valid disparity by the median of the valid disparities in its (2*radius+1) square neighbourhood, suppressing… |
| `func NewDisparityMap(rows, cols int) *DisparityMap` | NewDisparityMap allocates a DisparityMap whose samples are all `InvalidDisparity`. |
| `func RefineSubpixelParabola(vol *CostVolume, uniquenessRatio int) *DisparityMap` | RefineSubpixelParabola extracts a sub-pixel disparity map from an aggregated cost volume. |
| `func SpeckleFilter(disp *DisparityMap, maxSpeckleSize int, maxDiff float32) *DisparityMap` | SpeckleFilter removes small isolated disparity blobs ("speckles"). |
| `func (d *DisparityMap) At(y, x int) float32` | At returns the disparity at row y, column x. |
| `func (d *DisparityMap) Clone() *DisparityMap` | Clone returns a deep copy of the map. |
| `func (d *DisparityMap) CountValid() int` | CountValid returns the number of pixels holding a genuine disparity. |
| `func (d *DisparityMap) Fill(v float32)` | Fill sets every sample to v. |
| `func (d *DisparityMap) MinMax() (min, max float32, ok bool)` | MinMax returns the smallest and largest genuine disparities in the map and whether any valid pixel was found. |
| `func (d *DisparityMap) Set(y, x int, v float32)` | Set stores the disparity v at row y, column x. |
| `func (d *DisparityMap) Size() (rows, cols int)` | Size returns the map dimensions as (rows, cols). |
| `func (d *DisparityMap) ToMat() *cv.Mat` | ToMat renders the map to a viewable single-channel 8-bit github.com/malcolmston/opencv.Mat. |
| `func (d *DisparityMap) Valid(y, x int) bool` | Valid reports whether the pixel at row y, column x holds a genuine (non-sentinel) disparity. |

</details>

<details>
<summary><code>Plane</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FitPlane(points []Point3D) (Plane, error)` | FitPlane fits the total-least-squares plane to a set of 3-D points: the plane through their centroid whose normal is the direction of least variance… |
| `func FitPlaneRANSAC(points []Point3D, iterations int, threshold float64, seed int64) (Plane, []int, error)` | FitPlaneRANSAC robustly fits a plane to points that may contain outliers. |
| `func (pl Plane) DistanceTo(p Point3D) float64` | DistanceTo returns the signed perpendicular distance from point p to the plane. |
| `func (pl Plane) Normal() Point3D` | Normal returns the plane normal (A, B, C) as a `Point3D`. |

</details>

<details>
<summary><code>Point3D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (p Point3D) Add(q Point3D) Point3D` | Add returns the vector sum p+q. |
| `func (p Point3D) Cross(q Point3D) Point3D` | Cross returns the cross product p×q. |
| `func (p Point3D) Dot(q Point3D) float64` | Dot returns the dot product p·q. |
| `func (p Point3D) Norm() float64` | Norm returns the Euclidean length of p. |
| `func (p Point3D) Scale(s float64) Point3D` | Scale returns p scaled by the scalar s. |
| `func (p Point3D) Sub(q Point3D) Point3D` | Sub returns the vector difference p-q. |

</details>

<details>
<summary><code>PointCloud</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func PointCloudFromDepth(depth *DepthMap, cam Camera, color *cv.Mat) *PointCloud` | PointCloudFromDepth back-projects a depth map into a metric point cloud using the camera intrinsics: for pixel (y,x) with depth Z the point is… |
| `func PointCloudFromDisparity(disp *DisparityMap, cam Camera, baseline float64, color *cv.Mat) *PointCloud` | PointCloudFromDisparity converts a disparity map to depth (Fx as focal length and the given baseline) and then back-projects it with… |
| `func ReprojectImageTo3D(disp *DisparityMap, Q [4][4]float64, color *cv.Mat) *PointCloud` | ReprojectImageTo3D maps every valid disparity to a 3-D point using the 4x4 reprojection matrix Q (see `QMatrix`), the standard cv::reprojectImageTo3D… |
| `func (pc *PointCloud) Len() int` | Len returns the number of points in the cloud. |

</details>

<details>
<summary><code>SGMMatcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewSGMMatcher(minDisparity, numDisparities int) *SGMMatcher` | NewSGMMatcher returns an SGMMatcher with sensible defaults: a 5x5 census window, eight aggregation paths, P1=10, P2=120 and sub-pixel enabled. |
| `func (s *SGMMatcher) Compute(left, right *cv.Mat) *DisparityMap` | Compute matches left against right with semi-global aggregation and returns the left-referenced disparity map, sub-pixel refined when Subpixel is set. |
| `func (s *SGMMatcher) ComputeCostVolume(left, right *cv.Mat) *CostVolume` | ComputeCostVolume builds the census matching cost for the pair and returns the semi-globally aggregated cost volume, before disparity extraction. |

</details>

### Constants

`InvalidDepth`, `InvalidDisparity`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/stereo2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
