# transforms2 — implements pure-Go geometric image transformations for the parent cv package

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/transforms2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/transforms2)

Package transforms2 implements pure-Go geometric image transformations for the
parent cv package, operating on its `cv.Mat` (8-bit, channel-interleaved) and
`cv.FloatMat` types.

- Affine and projective warps with builders, composition and inversion. -
High-level helpers: resize, rotate (in-place or bound), scale, translate and
shear. - Arbitrary remapping via coordinate maps or a mapping function. -
Polar and log-polar (un)warping. - Piecewise-affine warping with Delaunay
triangulation. - Thin-plate-spline (TPS) warping for smooth non-rigid
deformation. - Brown-Conrady lens distortion modelling and (un)distortion. -
Enhanced Correlation Coefficient (ECC) image registration.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/transforms2"
```

## Usage

This is the package's own `ExampleComposeAffine`, so it compiles and its output is
asserted on every `go test ./transforms2/`.

```go
m := transforms2.ComposeAffine(
		transforms2.AffineTranslation(5, 5),
		transforms2.AffineScaling(2, 2),
	)
	x, y := transforms2.ApplyAffine(m, 3, 4)
	fmt.Printf("(3,4) -> (%.0f,%.0f)\n", x, y)

	inv, ok := transforms2.InvertAffine(m)
	if !ok {
		panic("singular")
	}
	bx, by := transforms2.ApplyAffine(inv, x, y)
	fmt.Printf("back    -> (%.0f,%.0f)\n", bx, by)

	_, ok = transforms2.InvertAffine(transforms2.AffineScaling(0, 1))
	fmt.Println("degenerate invertible:", ok)
```

```
(3,4) -> (11,13)
back    -> (3,4)
degenerate invertible: false
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AffineIdentity() cv.AffineMatrix` | AffineIdentity returns the affine transform that leaves points unchanged. |
| `func AffineRotation(angleDeg float64) cv.AffineMatrix` | AffineRotation returns the affine transform that rotates points about the origin by angleDeg degrees (counter-clockwise in OpenCV's convention). |
| `func AffineRotationAround(cx, cy, angleDeg, scale float64) cv.AffineMatrix` | AffineRotationAround returns the affine transform that rotates about the point (cx, cy) by angleDeg degrees (counter-clockwise) and uniformly scales… |
| `func AffineScaling(sx, sy float64) cv.AffineMatrix` | AffineScaling returns the affine transform that scales points by sx along x and sy along y about the origin. |
| `func AffineShear(shx, shy float64) cv.AffineMatrix` | AffineShear returns the affine transform that shears by shx along x (as a function of y) and shy along y (as a function of x). |
| `func AffineToPerspective(a cv.AffineMatrix) cv.PerspectiveMatrix` | AffineToPerspective embeds a 2x3 affine transform into a 3x3 homography. |
| `func AffineTranslation(tx, ty float64) cv.AffineMatrix` | AffineTranslation returns the affine transform that shifts points by (tx, ty). |
| `func ApplyAffine(m cv.AffineMatrix, x, y float64) (float64, float64)` | ApplyAffine maps the point (x, y) through the affine transform m. |
| `func ApplyPerspective(m cv.PerspectiveMatrix, x, y float64) (float64, float64)` | ApplyPerspective maps the point (x, y) through the homography m. |
| `func ComposeAffine(a, b cv.AffineMatrix) cv.AffineMatrix` | ComposeAffine returns the affine transform equivalent to applying b first and then a (a after b). |
| `func ComposePerspective(a, b cv.PerspectiveMatrix) cv.PerspectiveMatrix` | ComposePerspective returns the homography equivalent to applying b first and then a (a after b), i.e. |
| `func DelaunayTriangulation(pts []cv.Point2f) [][3]int` | DelaunayTriangulation computes a Delaunay triangulation of the given points using the Bowyer-Watson algorithm and returns the triangles as triples of… |
| `func DistortPoint(cam CameraMatrix, dist DistortionCoeffs, x, y float64) (float64, float64)` | DistortPoint maps an ideal (undistorted) pixel (x, y) to the pixel where the lens actually images it, given the camera intrinsics and distortion. |
| `func DistortPoints(cam CameraMatrix, dist DistortionCoeffs, pts []cv.Point2f) []cv.Point2f` | DistortPoints applies `DistortPoint` to every point in pts, returning a fresh slice. |
| `func EnhancedCorrelationCoefficient(a, b *cv.Mat) float64` | EnhancedCorrelationCoefficient returns the zero-mean normalised correlation coefficient between a and b (each flattened over all samples), a value in… |
| `func FindTransformECC(template, image *cv.Mat, model MotionModel, criteria ECCCriteria) (cv.AffineMatrix, float64, error)` | FindTransformECC estimates the geometric transform that best aligns image to template by maximising the Enhanced Correlation Coefficient (Evangelidis… |
| `func GetAffineTransform(src, dst [3]cv.Point2f) cv.AffineMatrix` | GetAffineTransform computes the affine transform that maps the three source points src to the three destination points dst. |
| `func GetPerspectiveTransform(src, dst [4]cv.Point2f) cv.PerspectiveMatrix` | GetPerspectiveTransform computes the homography mapping the four source points src to the four destination points dst, in corresponding order and… |
| `func InitUndistortRectifyMap(cam CameraMatrix, dist DistortionCoeffs, width, height int) (mapX, mapY *cv.FloatMat)` | InitUndistortRectifyMap builds coordinate maps of the given size such that Remap with them removes lens distortion: destination pixel (x, y) in the… |
| `func InvertAffine(m cv.AffineMatrix) (cv.AffineMatrix, bool)` | InvertAffine returns the inverse of m and reports whether its 2x2 linear part is invertible. |
| `func InvertPerspective(m cv.PerspectiveMatrix) (cv.PerspectiveMatrix, bool)` | InvertPerspective returns the inverse homography and reports whether m is invertible. |
| `func LinearPolar(src *cv.Mat, centerX, centerY, maxRadius float64, width, height int, interp Interpolation) *cv.Mat` | LinearPolar is a convenience wrapper for `WarpPolar` with `PolarLinear`. |
| `func LogPolar(src *cv.Mat, centerX, centerY, maxRadius float64, width, height int, interp Interpolation) *cv.Mat` | LogPolar is a convenience wrapper for `WarpPolar` with `PolarLog`. |
| `func MakeAffineMaps(m cv.AffineMatrix, width, height int) (mapX, mapY *cv.FloatMat)` | MakeAffineMaps returns coordinate maps of the given size that realise the inverse of the affine transform m, so that Remap with them reproduces… |
| `func MakeIdentityMaps(width, height int) (mapX, mapY *cv.FloatMat)` | MakeIdentityMaps returns coordinate maps of the given size for which destination pixel (x, y) maps to source (x, y). |
| `func MakePerspectiveMaps(m cv.PerspectiveMatrix, width, height int) (mapX, mapY *cv.FloatMat)` | MakePerspectiveMaps returns coordinate maps of the given size that realise the inverse of the homography m, so that Remap with them reproduces… |
| `func PerspectiveIdentity() cv.PerspectiveMatrix` | PerspectiveIdentity returns the homography that leaves points unchanged. |
| `func PiecewiseAffineWarp(src *cv.Mat, srcPts, dstPts []cv.Point2f, tris [][3]int, width, height int, interp Interpolation, border BorderMode, fill float64) *cv.Mat` | PiecewiseAffineWarp warps src by moving the control points srcPts to dstPts under a piecewise-affine deformation. |
| `func Remap(src *cv.Mat, mapX, mapY *cv.FloatMat, interp Interpolation, border BorderMode, fill float64) *cv.Mat` | Remap resamples src at the per-pixel source coordinates given by mapX and mapY: destination pixel (x, y) is taken from src at (mapX[y,x], mapY[y,x])… |
| `func RemapFunc(src *cv.Mat, width, height int, fn func(x, y float64) (sx, sy float64), interp Interpolation, border BorderMode, fill float64) *cv.Mat` | RemapFunc resamples src into a width x height output where the source coordinate for each destination pixel (x, y) is returned by fn. |
| `func Resize(src *cv.Mat, width, height int, interp Interpolation) *cv.Mat` | Resize returns src resampled to the given width and height using the chosen interpolation. |
| `func Rotate(src *cv.Mat, angleDeg, scale float64, interp Interpolation, border BorderMode, fill float64) *cv.Mat` | Rotate returns src rotated about its centre by angleDeg degrees (counter-clockwise) and uniformly scaled by scale, keeping the output the same size. |
| `func RotateBound(src *cv.Mat, angleDeg float64, interp Interpolation, border BorderMode, fill float64) *cv.Mat` | RotateBound returns src rotated about its centre by angleDeg degrees (counter-clockwise), enlarging the output canvas so that no content is clipped. |
| `func SampleChannel(src *cv.Mat, x, y float64, c int, interp Interpolation, border BorderMode, fill float64) float64` | SampleChannel returns the interpolated value of channel c of src at the fractional coordinate (x, y) using the given interpolation and border mode. |
| `func SamplePixel(src *cv.Mat, x, y float64, interp Interpolation, border BorderMode, fill float64) []uint8` | SamplePixel returns every channel of src sampled at the fractional coordinate (x, y) as freshly allocated, clamped 8-bit values. |
| `func Scale(src *cv.Mat, fx, fy float64, interp Interpolation) *cv.Mat` | Scale returns src scaled by the factors fx and fy about the origin, with the output size derived by rounding src's dimensions. |
| `func Shear(src *cv.Mat, shx, shy float64, interp Interpolation, border BorderMode, fill float64) *cv.Mat` | Shear returns src sheared by shx (along x, proportional to y) and shy (along y, proportional to x) while keeping the output the same size. |
| `func Translate(src *cv.Mat, tx, ty float64, interp Interpolation, border BorderMode, fill float64) *cv.Mat` | Translate returns src shifted by (tx, ty) pixels while keeping the output the same size, filling exposed areas per the border mode. |
| `func Undistort(src *cv.Mat, cam CameraMatrix, dist DistortionCoeffs, interp Interpolation, border BorderMode, fill float64) *cv.Mat` | Undistort removes lens distortion from src, returning an image of the same size in which straight lines are straight again. |
| `func UndistortPoint(cam CameraMatrix, dist DistortionCoeffs, u, v float64) (float64, float64)` | UndistortPoint maps a distorted pixel (u, v), as recorded by the lens, to the ideal pinhole pixel, given the camera intrinsics and distortion. |
| `func UndistortPoints(cam CameraMatrix, dist DistortionCoeffs, pts []cv.Point2f) []cv.Point2f` | UndistortPoints applies `UndistortPoint` to every point in pts, returning a fresh slice. |
| `func WarpAffine(src *cv.Mat, m cv.AffineMatrix, width, height int, interp Interpolation, border BorderMode, fill float64) *cv.Mat` | WarpAffine applies the affine transform m to src, producing a width x height output. |
| `func WarpPerspective(src *cv.Mat, m cv.PerspectiveMatrix, width, height int, interp Interpolation, border BorderMode, fill float64) *cv.Mat` | WarpPerspective applies the homography m to src, producing a width x height output. |
| `func WarpPolar(src *cv.Mat, centerX, centerY, maxRadius float64, width, height int, mode PolarMode, interp Interpolation, border BorderMode, fill float64) *cv.Mat` | WarpPolar remaps the Cartesian image src into a polar representation of size width (radius axis) by height (angle axis) about the centre (centerX,… |
| `func WarpPolarInverse(polar *cv.Mat, centerX, centerY, maxRadius float64, width, height int, mode PolarMode, interp Interpolation, border BorderMode, fill float64) *cv.Mat` | WarpPolarInverse reconstructs a Cartesian image of the given width and height from the polar image polar (as produced by `WarpPolar` with the same… |
| `func WarpTriangle(src, dst *cv.Mat, srcTri, dstTri [3]cv.Point2f, interp Interpolation, border BorderMode, fill float64)` | WarpTriangle warps the source triangle srcTri onto the destination triangle dstTri, compositing the result into dst in place (overwriting the covered… |

### Types

| Type | What it is |
| --- | --- |
| `BorderMode` | BorderMode selects how out-of-range coordinates are handled while sampling. |
| `CameraMatrix` | CameraMatrix holds the pinhole intrinsic parameters: focal lengths (Fx, Fy) in pixels and the principal point (Cx, Cy). |
| `DistortionCoeffs` | DistortionCoeffs holds Brown-Conrady lens distortion coefficients: radial terms K1, K2, K3 and tangential terms P1, P2, matching OpenCV's ordering… |
| `ECCCriteria` | ECCCriteria controls the `FindTransformECC` iteration: it stops after MaxCount iterations or once the parameter update magnitude falls below Epsilon,… |
| `Interpolation` | Interpolation selects the resampling kernel used when a transform samples the source image at fractional coordinates. |
| `MotionModel` | MotionModel selects the parametric warp family estimated by `FindTransformECC`. |
| `PolarMode` | PolarMode selects the radial mapping used by `WarpPolar` and `WarpPolarInverse`. |
| `ThinPlateSpline` | ThinPlateSpline is a smooth non-rigid coordinate mapping fitted from a set of control-point correspondences. |

<details>
<summary><code>ECCCriteria</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultECCCriteria() ECCCriteria` | DefaultECCCriteria returns criteria suitable for typical registration tasks (200 iterations, epsilon 1e-6). |

</details>

<details>
<summary><code>ThinPlateSpline</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewThinPlateSpline(from, to []cv.Point2f, lambda float64) *ThinPlateSpline` | NewThinPlateSpline fits a thin-plate spline that maps the points in from to the corresponding points in to. |
| `func (t *ThinPlateSpline) Transform(x, y float64) (float64, float64)` | Transform maps a single point (x, y) through the spline, returning its image in the target coordinate system. |
| `func (t *ThinPlateSpline) Warp(src *cv.Mat, width, height int, interp Interpolation, border BorderMode, fill float64) *cv.Mat` | Warp renders src deformed by the spline into a width x height output. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/transforms2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
