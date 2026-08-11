# calib2 — pure standard-library implementation of camera calibration and multi-view 3D geometry

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/calib2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/calib2)

Package calib2 is a pure standard-library implementation of camera calibration
and multi-view 3D geometry, built on top of the root cv package
(github.com/malcolmston/opencv).

It depends only on the Go standard library and the root cv package for its
image and floating-point point types: there is no cgo, no third-party code and
no GPU dependency. All numerical kernels — a symmetric Jacobi eigensolver,
Gaussian-elimination linear solves, an RQ decomposition and the small
fixed-size 3×3 / 3-vector helpers — are implemented from scratch and are
fully deterministic.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/calib2"
```

## Usage

This is the package's own `ExampleCameraMatrix`, so it compiles and its output is
asserted on every `go test ./calib2/`.

```go
k := vgaCamera()
	m := k.Matrix()
	fmt.Println(m[0], m[1], m[2])

	inv := k.Inverse()
	fmt.Printf("%.5f %.4f %.4f\n", inv[0][0], inv[0][2], inv[1][2])

	fmt.Println(calib2.NewCameraMatrix(m) == k)
```

```
[800 0 320] [0 800 240] [0 0 1]
0.00125 -0.4000 -0.3000
true
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ApplyHomography(h [3][3]float64, p cv.Point2f) cv.Point2f` | ApplyHomography maps a single point through a 3×3 homography, performing the perspective division. |
| `func CalibrateCamera(objectPoints [][][3]float64, imagePoints [][]cv.Point2f) (k CameraMatrix, d DistortionCoeffs, poses []Pose, rmsError float64, err error)` | CalibrateCamera performs intrinsic camera calibration from several views of a planar target using Zhang's method. |
| `func ComposeRotations(a, b [3][3]float64) [3][3]float64` | ComposeRotations returns the rotation matrix equivalent to applying rotation b first and then a, i.e. |
| `func ComputeCorrespondEpilines(points []cv.Point2f, whichImage int, f [3][3]float64) [][3]float64` | ComputeCorrespondEpilines maps image points from one view to their epipolar lines in the other view via the fundamental matrix f. |
| `func DecomposeEssentialMatrix(e [3][3]float64) (r1, r2 [3][3]float64, t [3]float64)` | DecomposeEssentialMatrix factors an essential matrix into the two possible rotations and the (sign-ambiguous) unit translation direction of the… |
| `func DepthFromDisparity(disparity, focal, baseline float64) float64` | DepthFromDisparity returns the metric depth Z = focal·baseline/disparity for a rectified stereo pair. |
| `func DisparityTo3D(u, v, disparity float64, q [4][4]float64) [3]float64` | DisparityTo3D reprojects a single rectified pixel (u, v) with its disparity to a 3D point using the disparity-to-depth matrix q from `StereoRectify`. |
| `func DistortPoint(pt cv.Point2f, k CameraMatrix, d DistortionCoeffs) cv.Point2f` | DistortPoint maps an ideal (undistortion-corrected) image pixel to the pixel where the lens would actually image that ray, applying the… |
| `func DistortPoints(pts []cv.Point2f, k CameraMatrix, d DistortionCoeffs) []cv.Point2f` | DistortPoints applies `DistortPoint` to every point, returning a new slice in the same order. |
| `func EssentialMatrix(pose Pose) [3][3]float64` | EssentialMatrix returns the essential matrix E = `T`ₓ·R for the relative pose (R, T) that maps the first camera's coordinates to the second… |
| `func EulerToRotation(roll, pitch, yaw float64) [3][3]float64` | EulerToRotation builds a rotation matrix from intrinsic Tait–Bryan angles in the Z-Y-X convention: R = Rz(yaw)·Ry(pitch)·Rx(roll). |
| `func FindHomography(src, dst []cv.Point2f) (h [3][3]float64, ok bool)` | FindHomography estimates the 3×3 projective transform H that maps the source points to the destination points, using the normalized direct linear… |
| `func FundamentalMatrix(k1, k2 CameraMatrix, pose Pose) [3][3]float64` | FundamentalMatrix returns the fundamental matrix F = K2⁻ᵀ·E·K1⁻¹ relating pixel coordinates in the two views, where E is the essential… |
| `func GenerateChessboardCorners(patternRows, patternCols int, squareSize float64) [][3]float64` | GenerateChessboardCorners returns the 3D object coordinates of the inner corners of a chessboard calibration target lying in the z = 0 plane. |
| `func InitUndistortRectifyMap(k CameraMatrix, d DistortionCoeffs, rect [3][3]float64, newK CameraMatrix, rows, cols int) (mapX, mapY [][]float64)` | InitUndistortRectifyMap builds the per-pixel resampling maps that convert a distorted source image into an undistorted, rectified image of size… |
| `func IsRotationMatrix(r [3][3]float64, tol float64) bool` | IsRotationMatrix reports whether r is a proper rotation matrix to the given tolerance: its transpose is its inverse (rᵀr = I) and its determinant… |
| `func Mat3Det(a [3][3]float64) float64` | Mat3Det returns the determinant of a 3×3 matrix. |
| `func Mat3Identity() [3][3]float64` | Mat3Identity returns the 3×3 identity matrix. |
| `func Mat3Inverse(a [3][3]float64) (inv [3][3]float64, ok bool)` | Mat3Inverse returns the inverse of a 3×3 matrix and reports ok=false when the matrix is singular to working precision. |
| `func Mat3Mul(a, b [3][3]float64) [3][3]float64` | Mat3Mul returns the matrix product a·b of two 3×3 matrices. |
| `func Mat3Transpose(a [3][3]float64) [3][3]float64` | Mat3Transpose returns the transpose of a 3×3 matrix. |
| `func Mat3VecMul(a [3][3]float64, v [3]float64) [3]float64` | Mat3VecMul returns the matrix–vector product a·v for a 3×3 matrix and a 3-vector. |
| `func NormalizeQuaternion(q [4]float64) [4]float64` | NormalizeQuaternion returns q scaled to unit length in (w, x, y, z) order. |
| `func ProjectPoint(objectPoint [3]float64, pose Pose, k CameraMatrix, d DistortionCoeffs) cv.Point2f` | ProjectPoint projects a single 3D world point into the image through the full pinhole model: it is transformed by the pose, distorted by the lens… |
| `func ProjectPoints(objectPoints [][3]float64, pose Pose, k CameraMatrix, d DistortionCoeffs) []cv.Point2f` | ProjectPoints projects a slice of 3D world points into the image through the full pinhole model, applying the pose, distortion and intrinsics to each… |
| `func ProjectionMatrix(k CameraMatrix, pose Pose) [3][4]float64` | ProjectionMatrix returns the 3×4 camera projection matrix P = K·[R\|T] that maps homogeneous world points to homogeneous image points. |
| `func QuaternionToRotation(q [4]float64) [3][3]float64` | QuaternionToRotation converts a quaternion in (w, x, y, z) order into a 3×3 rotation matrix. |
| `func Remap(src *cv.Mat, mapX, mapY [][]float64) *cv.Mat` | Remap resamples src according to the coordinate maps produced by `InitUndistortRectifyMap`, using bilinear interpolation. |
| `func ReprojectImageTo3D(disparity [][]float64, q [4][4]float64) [][][3]float64` | ReprojectImageTo3D reprojects an entire disparity map to 3D points using the disparity-to-depth matrix q. |
| `func ReprojectionError(objectPoints [][3]float64, imagePoints []cv.Point2f, pose Pose, k CameraMatrix, d DistortionCoeffs) float64` | ReprojectionError returns the mean Euclidean reprojection error, in pixels, between the observed image points and the projections of the object… |
| `func ReprojectionErrorRMS(objectPoints [][3]float64, imagePoints []cv.Point2f, pose Pose, k CameraMatrix, d DistortionCoeffs) float64` | ReprojectionErrorRMS returns the root-mean-square reprojection error, in pixels, between the observed image points and the projections of the object… |
| `func Rodrigues(rvec [3]float64) [3][3]float64` | Rodrigues converts a rotation vector (axis-angle, whose direction is the rotation axis and whose magnitude is the rotation angle in radians) into the… |
| `func RodriguesInverse(r [3][3]float64) [3]float64` | RodriguesInverse converts a 3×3 rotation matrix into the equivalent rotation vector (axis-angle) via the inverse of Rodrigues' formula. |
| `func RotationToEuler(r [3][3]float64) (roll, pitch, yaw float64)` | RotationToEuler extracts intrinsic Tait–Bryan angles (roll about x, pitch about y, yaw about z) from a rotation matrix, inverting `EulerToRotation`. |
| `func RotationToQuaternion(r [3][3]float64) [4]float64` | RotationToQuaternion converts a rotation matrix into a unit quaternion in (w, x, y, z) order, using the numerically stable branch selection of… |
| `func RotationX(theta float64) [3][3]float64` | RotationX returns the 3×3 matrix of a rotation by theta radians about the x-axis. |
| `func RotationY(theta float64) [3][3]float64` | RotationY returns the 3×3 matrix of a rotation by theta radians about the y-axis. |
| `func RotationZ(theta float64) [3][3]float64` | RotationZ returns the 3×3 matrix of a rotation by theta radians about the z-axis. |
| `func Triangulate(p1, p2 [3][4]float64, pt1, pt2 cv.Point2f) [3]float64` | Triangulate reconstructs a single 3D point from its projections in two views by the homogeneous linear (DLT) method. |
| `func TriangulatePoints(p1, p2 [3][4]float64, pts1, pts2 []cv.Point2f) [][3]float64` | TriangulatePoints reconstructs many 3D points from two views by applying `Triangulate` to each correspondence. |
| `func UndistortImage(src *cv.Mat, k CameraMatrix, d DistortionCoeffs) *cv.Mat` | UndistortImage removes lens distortion from src, producing a new image of the same size in which straight lines in the scene appear straight. |
| `func UndistortPoint(pt cv.Point2f, k CameraMatrix, d DistortionCoeffs) cv.Point2f` | UndistortPoint maps a distorted image pixel to the ideal pixel it would occupy under a distortion-free pinhole camera with the same intrinsics,… |
| `func UndistortPoints(pts []cv.Point2f, k CameraMatrix, d DistortionCoeffs) []cv.Point2f` | UndistortPoints applies `UndistortPoint` to every point, returning a new slice in the same order. |
| `func Vec3Add(a, b [3]float64) [3]float64` | Vec3Add returns the element-wise sum a+b. |
| `func Vec3Cross(a, b [3]float64) [3]float64` | Vec3Cross returns the cross product a×b. |
| `func Vec3Dot(a, b [3]float64) float64` | Vec3Dot returns the dot product a·b. |
| `func Vec3Norm(a [3]float64) float64` | Vec3Norm returns the Euclidean length of a. |
| `func Vec3Normalize(a [3]float64) [3]float64` | Vec3Normalize returns a unit vector in the direction of a. |
| `func Vec3Scale(a [3]float64, s float64) [3]float64` | Vec3Scale returns the vector a scaled by the scalar s. |
| `func Vec3Sub(a, b [3]float64) [3]float64` | Vec3Sub returns the element-wise difference a-b. |

### Types

| Type | What it is |
| --- | --- |
| `CameraMatrix` | CameraMatrix holds the intrinsic parameters of a pinhole camera: the focal lengths Fx, Fy in pixels, the principal point (Cx, Cy) in pixels and an… |
| `DistortionCoeffs` | DistortionCoeffs holds the Brown–Conrady lens-distortion coefficients: the radial terms K1, K2, K3 and the tangential terms P1, P2. |
| `Matrix` | Matrix is a dense, row-major matrix of float64 values, used throughout the package for the linear-algebra steps of calibration and triangulation. |
| `Pose` | Pose is a rigid-body transform from world (object) coordinates to camera coordinates, expressed as a 3×3 rotation R and a translation T such that a… |
| `StereoRectification` | StereoRectification holds the result of `StereoRectify`: the rectifying rotations R1, R2 to apply to each camera (via `InitUndistortRectifyMap`), the… |

<details>
<summary><code>CameraMatrix</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DecomposeProjectionMatrix(p [3][4]float64) (k CameraMatrix, r [3][3]float64, t [3]float64)` | DecomposeProjectionMatrix factors a 3×4 projection matrix P into an intrinsic `CameraMatrix`, a rotation R and a translation T such that P is… |
| `func IntrinsicsFromHomographies(homographies [][3][3]float64) (CameraMatrix, error)` | IntrinsicsFromHomographies recovers the camera intrinsics from a set of plane-to-image homographies using Zhang's closed-form solution. |
| `func NewCameraMatrix(k [3][3]float64) CameraMatrix` | NewCameraMatrix builds a `CameraMatrix` from a 3×3 intrinsic matrix, extracting the focal lengths, skew and principal point. |
| `func (c CameraMatrix) Inverse() [3][3]float64` | Inverse returns the inverse of the intrinsic matrix. |
| `func (c CameraMatrix) Matrix() [3][3]float64` | Matrix returns the intrinsic parameters as a 3×3 row-major matrix. |

</details>

<details>
<summary><code>DistortionCoeffs</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDistortionCoeffs(s []float64) DistortionCoeffs` | NewDistortionCoeffs builds a `DistortionCoeffs` from a coefficient slice in OpenCV's [K1, K2, P1, P2, K3] ordering. |
| `func (d DistortionCoeffs) IsZero() bool` | IsZero reports whether all coefficients are zero, i.e. |
| `func (d DistortionCoeffs) Slice() []float64` | Slice returns the coefficients in OpenCV's canonical ordering [K1, K2, P1, P2, K3]. |

</details>

<details>
<summary><code>Matrix</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func IdentityMatrix(n int) *Matrix` | IdentityMatrix returns an n×n identity matrix. |
| `func NewMatrix(rows, cols int) *Matrix` | NewMatrix allocates a zero-filled Matrix with the given dimensions. |
| `func ZerosMatrix(rows, cols int) *Matrix` | ZerosMatrix returns a rows×cols matrix filled with zeros. |
| `func (m *Matrix) At(i, j int) float64` | At returns the element at row i, column j. |
| `func (m *Matrix) Clone() *Matrix` | Clone returns an independent deep copy of the matrix. |
| `func (m *Matrix) FrobeniusNorm() float64` | FrobeniusNorm returns the Frobenius norm (square root of the sum of squared elements) of the matrix. |
| `func (m *Matrix) Mul(other *Matrix) *Matrix` | Mul returns the matrix product m·other. |
| `func (m *Matrix) MulVec(v []float64) []float64` | MulVec returns the matrix–vector product m·v as a new slice. |
| `func (m *Matrix) Set(i, j int, v float64)` | Set stores v at row i, column j. |
| `func (m *Matrix) Transpose() *Matrix` | Transpose returns a new matrix that is the transpose of the receiver. |

</details>

<details>
<summary><code>Pose</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func ExtrinsicsFromHomography(h [3][3]float64, k CameraMatrix) Pose` | ExtrinsicsFromHomography recovers the camera pose (rotation and translation) for a planar target from its plane-to-image homography and the known… |
| `func NewPoseFromRvec(rvec, tvec [3]float64) Pose` | NewPoseFromRvec builds a `Pose` from an axis-angle rotation vector and a translation vector. |
| `func (p Pose) Apply(x [3]float64) [3]float64` | Apply transforms a world point into camera coordinates: R·X + T. |
| `func (p Pose) Inverse() Pose` | Inverse returns the inverse rigid transform, mapping camera coordinates back to world coordinates. |
| `func (p Pose) Matrix() [3][4]float64` | Matrix returns the pose as a 3×4 [R\|T] matrix in row-major order. |
| `func (p Pose) Rvec() [3]float64` | Rvec returns the rotation of the pose as an axis-angle rotation vector. |

</details>

<details>
<summary><code>StereoRectification</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func StereoRectify(k1, k2 CameraMatrix, pose Pose) StereoRectification` | StereoRectify computes the rectification transforms for a calibrated stereo pair. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/calib2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
