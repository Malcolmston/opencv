# rgbd — standard-library-only implementation of the depth-map and point-cloud routines

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/rgbd.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/rgbd)

Package rgbd is a standard-library-only implementation of the depth-map and
point-cloud routines from OpenCV's contrib rgbd module, built on top of the
root cv package (github.com/malcolmston/opencv).

It depends only on the Go standard library and the root cv package: no cgo, no
third-party code, and no dependency on any sibling cv/* subpackage. Where the
root package offers a building block — notably
`github.com/malcolmston/opencv.FloatMat` for single-channel float grids —
rgbd reuses it; the numerical kernels it needs beyond that (a symmetric 3×3
Jacobi eigensolver and a 3×3 singular value decomposition) are implemented
locally in this package.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/rgbd"
```

## Usage

This is the package's own `ExampleDepthCleaner`, so it compiles and its output is
asserted on every `go test ./rgbd/`.

```go
depth := cv.NewFloatMat(5, 5)
	for i := range depth.Data {
		depth.Data[i] = 2.0
	}
	depth.Data[2*5+2] = 0
	out := rgbd.NewDepthCleaner(1).Clean(depth)
	fmt.Printf("filled=%.1f\n", out.At(2, 2))
```

```
filled=2.0
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func BilateralDepthFilter(depth *cv.FloatMat, radius int, sigmaSpace, sigmaDepth float64) *cv.FloatMat` | BilateralDepthFilter applies an edge-preserving bilateral filter to a depth map. |
| `func Compute3DNormals(depth *cv.FloatMat, k [3][3]float64) [][3]float64` | Compute3DNormals estimates a unit surface normal at every pixel of a depth map from the local tangent plane spanned by neighbouring back-projected… |
| `func DepthTo3D(depth *cv.FloatMat, k [3][3]float64) [][3]float64` | DepthTo3D back-projects every pixel of a depth map into a 3-D point using the pinhole intrinsics K. The result is a dense, row-major slice of length… |
| `func DepthTo3dSparse(depth *cv.FloatMat, k [3][3]float64, pixels [][2]int) (points [][3]float64, valid []bool)` | DepthTo3dSparse back-projects a chosen set of pixels of a depth map into 3-D points, the sparse counterpart of `DepthTo3D`. |
| `func ICP(src, dst [][3]float64, iters int) (r [3][3]float64, t [3]float64, err float64)` | ICP aligns the src point cloud to the dst point cloud with point-to-point iterative closest point. |
| `func InverseRodrigues(r [3][3]float64) [3]float64` | InverseRodrigues converts a 3×3 rotation matrix to its rotation vector (axis-angle) representation, the inverse of `Rodrigues`. |
| `func RegisterDepth(depth *cv.FloatMat, kd, kc [3][3]float64, r [3][3]float64, t [3]float64, outRows, outCols int) *cv.FloatMat` | RegisterDepth warps a depth map captured by one camera into the image frame of a second (colour) camera. |
| `func RegisterDepthDistorted(depth *cv.FloatMat, kd, kc [3][3]float64, dist [5]float64, r [3][3]float64, t [3]float64, outRows, outCols int) *cv.FloatMat` | RegisterDepthDistorted warps a depth map from a depth camera into the image frame of a colour camera, exactly like `RegisterDepth`, but additionally… |
| `func RescaleDepth(depth *cv.FloatMat, scale float64) *cv.FloatMat` | RescaleDepth returns a copy of depth with every valid measurement multiplied by scale, the standard way to convert integer depth units (for example… |
| `func Rodrigues(rvec [3]float64) [3][3]float64` | Rodrigues converts a rotation vector (axis-angle, whose direction is the rotation axis and whose length is the rotation angle in radians) to a 3×3… |
| `func ValidPoints(points [][3]float64) [][3]float64` | ValidPoints returns the subset of points whose depth (Z component) is strictly positive, in their original order. |
| `func VoxelDownsample(points [][3]float64, leaf float64) [][3]float64` | VoxelDownsample reduces a point cloud by overlaying a regular grid of cubic cells with side length leaf and replacing all points that fall in a cell… |
| `func WarpFrame(image, depth *cv.FloatMat, k [3][3]float64, pose Pose, rows, cols int) (warpedImage, warpedDepth *cv.FloatMat, mask []bool)` | WarpFrame renders how an intensity image and its depth map would look from a second camera pose, the operation OpenCV calls warpFrame. |

### Types

| Type | What it is |
| --- | --- |
| `Camera` | Camera holds the intrinsic parameters of a pinhole camera: the focal lengths Fx, Fy (in pixels) and the principal point (Cx, Cy). |
| `DepthCleaner` | DepthCleaner removes the two dominant artefacts of consumer depth sensors: small holes (missing measurements) and speckle noise. |
| `NormalsMethod` | NormalsMethod selects the estimator used by `RgbdNormals`. |
| `OdometryOptions` | OdometryOptions configures the frame-to-frame odometry solvers (`ICPOdometry`, `RgbdOdometry` and `RgbdICPOdometry`). |
| `OdometryResult` | OdometryResult reports the outcome of an odometry solve. |
| `Plane` | Plane is an oriented plane in 3-D given in Hessian normal form: a unit Normal n and an offset D such that a point p lies on the plane when n·p + D =… |
| `PlaneOptions` | PlaneOptions configures `PlaneSegmentation`. |
| `Pose` | Pose is a rigid body transform in 3-D: a rotation R and a translation T that act on a point as p' = R·p + T. It is the transform type used by the… |
| `RgbdNormals` | RgbdNormals estimates per-pixel surface normals from a depth map using one of three methods (FALS, LINE-MOD or SRI), following OpenCV's… |
| `TSDFVolume` | TSDFVolume is a lightweight truncated signed distance function volume, the core data structure of KinectFusion-style dense reconstruction. |

<details>
<summary><code>Camera</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (c Camera) K() [3][3]float64` | K returns the intrinsic parameters as the 3×3 row-major matrix |

</details>

<details>
<summary><code>DepthCleaner</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDepthCleaner(window int) *DepthCleaner` | NewDepthCleaner returns a `DepthCleaner` with the given neighbourhood half-size and no maximum-depth clamp. |
| `func (c *DepthCleaner) Clean(depth *cv.FloatMat) *cv.FloatMat` | Clean returns a cleaned copy of depth. |

</details>

<details>
<summary><code>OdometryOptions</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultOdometryOptions() OdometryOptions` | DefaultOdometryOptions returns sensible defaults: 20 iterations, a 0.1–4 m depth band, a 0.07 m association gate, a 1e-8 convergence threshold,… |

</details>

<details>
<summary><code>OdometryResult</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ICPOdometry(srcDepth, dstDepth *cv.FloatMat, k [3][3]float64, opts OdometryOptions) OdometryResult` | ICPOdometry estimates the rigid motion between two depth frames with dense projective point-to-plane ICP, the geometric arm of OpenCV's rgbd odometry. |
| `func RgbdICPOdometry(srcGray, srcDepth, dstGray, dstDepth *cv.FloatMat, k [3][3]float64, opts OdometryOptions) OdometryResult` | RgbdICPOdometry estimates motion by jointly minimising the photometric and the geometric point-to-plane residuals, the full RGB-D + ICP odometry. |
| `func RgbdOdometry(srcGray, srcDepth, dstGray, dstDepth *cv.FloatMat, k [3][3]float64, opts OdometryOptions) OdometryResult` | RgbdOdometry estimates the rigid motion between two frames from photometric (brightness-constancy) constraints, the direct/dense arm of RGB-D… |

</details>

<details>
<summary><code>Plane</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func PlaneSegmentation(points [][3]float64, opts PlaneOptions) ([]Plane, []int)` | PlaneSegmentation extracts one or more dominant planes from a point cloud with sequential RANSAC. |
| `func (pl Plane) Distance(p [3]float64) float64` | Distance returns the absolute perpendicular distance from p to the plane. |

</details>

<details>
<summary><code>PlaneOptions</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultPlaneOptions() PlaneOptions` | DefaultPlaneOptions returns reasonable defaults: a 0.02-unit inlier band, 200 hypotheses per plane, up to 4 planes, an automatic minimum inlier count… |

</details>

<details>
<summary><code>Pose</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ColoredICP(src, dst [][3]float64, srcColor, dstColor []float64, colorWeight float64, iters int) (Pose, float64)` | ColoredICP aligns two coloured point clouds, using per-point colour (a scalar intensity in [0,1], say) to disambiguate correspondences that geometry… |
| `func ICPPointToPlane(src, dst [][3]float64, dstNormals [][3]float64, iters int) (Pose, float64)` | ICPPointToPlane aligns the src point cloud to the dst point cloud with point-to-plane iterative closest point. |
| `func IdentityPose() Pose` | IdentityPose returns the identity transform (R = I, T = 0), which maps every point to itself. |
| `func PoseFromRt(r [3][3]float64, t [3]float64) Pose` | PoseFromRt builds a `Pose` from an explicit rotation matrix and translation vector. |
| `func (p Pose) Apply(x [3]float64) [3]float64` | Apply transforms the point p by the pose, returning R·p + T. |
| `func (p Pose) Compose(b Pose) Pose` | Compose returns the composition p∘b, the transform that applies b first and then p: (p∘b)(x) = p.Apply(b.Apply(x)). |
| `func (p Pose) Inverse() Pose` | Inverse returns the inverse transform, which undoes p: its rotation is Rᵀ and its translation is −Rᵀ·T, so p.Inverse().Apply(p.Apply(x)) == x… |

</details>

<details>
<summary><code>RgbdNormals</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRgbdNormals(rows, cols int, k [3][3]float64, window int, method NormalsMethod) *RgbdNormals` | NewRgbdNormals returns an `RgbdNormals` for rows×cols depth maps with the given intrinsics, neighbourhood half-size and method. |
| `func (rn *RgbdNormals) Compute(depth *cv.FloatMat) [][3]float64` | Compute returns a dense, row-major slice of unit normals, one per pixel, in the same layout as `DepthTo3D`. |

</details>

<details>
<summary><code>TSDFVolume</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTSDFVolume(dims [3]int, voxelSize float64, origin [3]float64, truncation float64) *TSDFVolume` | NewTSDFVolume allocates an empty volume of dims = {nx, ny, nz} voxels of side voxelSize (metres), whose (0,0,0) corner sits at origin in world… |
| `func (vol *TSDFVolume) FetchPointCloud() [][3]float64` | FetchPointCloud extracts a surface point cloud from the volume by scanning for sign changes of the TSDF between adjacent voxels along each axis and… |
| `func (vol *TSDFVolume) Integrate(depth *cv.FloatMat, k [3][3]float64, camPose Pose)` | Integrate fuses one depth frame into the volume. |
| `func (vol *TSDFVolume) Raycast(k [3][3]float64, camPose Pose, rows, cols int) *cv.FloatMat` | Raycast renders a depth map by marching a ray from the camera through each pixel and reporting the depth at which the interpolated TSDF crosses zero… |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/rgbd).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
