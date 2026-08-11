# rapid — standard-library-only port of OpenCV's rapid module for real-time silhouette-based 3D

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/rapid.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/rapid)

Package rapid is a standard-library-only port of OpenCV's rapid module for
real-time silhouette-based 3D pose tracking, built on top of the root cv
package (github.com/malcolmston/opencv).

RAPID (Real-time Attitude and Position Determination) tracks the 6-DOF pose of
a known rigid object by aligning the projected silhouette edges of its 3D mesh
with intensity edges in the image. Given an approximate pose it performs the
classic one-dimensional search: control points are sampled along the projected
contour, a short image line ("line bundle") is read out along each contour
normal, the strongest edge along every line is located, and the resulting
perpendicular displacements drive a Gauss-Newton update of the pose.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/rapid"
```

## Usage

This is the package's own `ExampleExtractControlPoints`, so it compiles and its output is
asserted on every `go test ./rapid/`.

```go
k := rapid.NewCamera(500, 500, 240, 240)
	pose := rapid.Pose{Rvec: [3]float64{0.15, -0.2, 0.1}, Tvec: [3]float64{0, 0, 6}}
	cps := rapid.ExtractControlPoints(64, 15, cube(), pose, k, 480, 480)
	fmt.Println(len(cps) > 0)
```

```
true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func DrawCorrespondencies(bundle *cv.Mat, cols []int, colors []cv.Scalar)` | DrawCorrespondencies marks the found edge column on each row of the bundle image produced by `ExtractLineBundle`. |
| `func DrawSearchLines(img *cv.Mat, srcLocations [][]cv.Point, color cv.Scalar)` | DrawSearchLines draws the search line of every control point onto img by connecting the first and last recorded sample location for each row of… |
| `func DrawWireframe(img *cv.Mat, pts2d []Point2f, tris [][3]int, color cv.Scalar, cullBackface bool)` | DrawWireframe draws the mesh edges onto img using the projected vertex positions pts2d and the triangle list tris. |
| `func ExtractLineBundle(length int, ctl []ControlPoint, img *cv.Mat) (bundle *cv.Mat, srcLocations [][]cv.Point)` | ExtractLineBundle reads a one-dimensional intensity profile of length 2*length+1 along each control point's search normal into a bundle image. |
| `func FindCorrespondencies(bundle *cv.Mat) (cols []int, response []float64)` | FindCorrespondencies locates, for every row of the bundle, the column of the strongest intensity edge. |
| `func NewCamera(fx, fy, cx, cy float64) [3][3]float64` | NewCamera builds a 3×3 pinhole intrinsic matrix K from focal lengths (fx, fy) and principal point (cx, cy): |

### Types

| Type | What it is |
| --- | --- |
| `ControlPoint` | ControlPoint is a point sampled on the projected silhouette of the mesh. |
| `Correspondence` | Correspondence pairs a matched 2D image edge point with its 3D model point and the search normal along which the match was found. |
| `GOSTracker` | GOSTracker is the global-optimal-search RAPID variant. |
| `Mesh` | Mesh is a triangulated 3D model expressed in object (world) coordinates. |
| `OLSTracker` | OLSTracker is the optimized-local-search RAPID variant. |
| `Point2f` | Point2f is a sub-pixel image coordinate (X is the column, Y is the row). |
| `Pose` | Pose is a rigid transform from object to camera coordinates, stored as an axis-angle rotation vector (Rvec, a Rodrigues vector) and a translation… |
| `Rapid` | Rapid is the basic RAPID tracker. |
| `TermCriteria` | TermCriteria bounds the iteration of `Rapid.Compute`. |
| `Tracker` | Tracker is the common interface implemented by all RAPID trackers: `Rapid`, `OLSTracker` and `GOSTracker`. |

<details>
<summary><code>ControlPoint</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ExtractControlPoints(num, length int, mesh *Mesh, pose Pose, k [3][3]float64, width, height int) []ControlPoint` | ExtractControlPoints projects the mesh at the given pose, extracts its silhouette (contour) edges, and samples up to num control points spread along… |

</details>

<details>
<summary><code>Correspondence</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ConvertCorrespondencies(cols []int, response []float64, srcLocations [][]cv.Point, ctl []ControlPoint, minResponse float64) (matches []Correspondence, mask []bool)` | ConvertCorrespondencies converts found bundle columns into `Correspondence` pairs. |

</details>

<details>
<summary><code>GOSTracker</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewGOSTracker(mesh *Mesh, histBins int, sobelThresh float64) *GOSTracker` | NewGOSTracker creates a `GOSTracker` for the given mesh. |

</details>

<details>
<summary><code>OLSTracker</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewOLSTracker(mesh *Mesh, histBins int, sobelThresh float64) *OLSTracker` | NewOLSTracker creates an `OLSTracker` for the given mesh. |

</details>

<details>
<summary><code>Point2f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ProjectVertices(mesh *Mesh, pose Pose, k [3][3]float64) []Point2f` | ProjectVertices projects every vertex of the mesh through the given pose and intrinsics, returning image coordinates in vertex order. |

</details>

<details>
<summary><code>Rapid</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRapid(mesh *Mesh) *Rapid` | NewRapid creates a `Rapid` tracker for the given mesh. |
| `func (r *Rapid) ClearState()` | ClearState resets the tracker's search state. |
| `func (r *Rapid) Compute(img *cv.Mat, num, length int, k [3][3]float64, pose Pose, term TermCriteria) (Pose, float64)` | Compute iterates `Rapid.Proceed` until the pose increment falls below term.Epsilon or term.MaxCount iterations have run. |
| `func (r *Rapid) Proceed(img *cv.Mat, num, length int, k [3][3]float64, pose Pose) (Pose, float64, float64)` | Proceed performs a single RAPID iteration: extract control points, read the line bundles, find the strongest edges, build correspondences, and apply… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/rapid).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
