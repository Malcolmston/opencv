# calib3d — standard-library-only implementation of the classic camera-geometry and multi-view

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/calib3d.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/calib3d)

Package calib3d is a standard-library-only implementation of the classic
camera-geometry and multi-view routines from OpenCV's calib3d module, built on
top of the root cv package (github.com/malcolmston/opencv).

It depends only on the Go standard library and the root cv package: no cgo, no
third-party code. Where the root package offers a building block — notably
`github.com/malcolmston/opencv.GetPerspectiveTransform` and
`github.com/malcolmston/opencv.WarpPerspective` — calib3d reuses it; the
numerical kernels it needs beyond that (a symmetric Jacobi eigensolver, a 3×3
SVD, and small matrix helpers) are implemented from scratch.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/calib3d"
```

## Usage

This is the package's own `ExampleFindChessboardCorners`, so it compiles and its output is
asserted on every `go test ./calib3d/`.

```go
// Build a 4x4-square board (3x3 inner corners) with a white quiet zone.
	const sq, margin = 20, 20
	w := 4*sq + 2*margin
	img := cv.NewMat(w, w, 1)
	img.SetTo(255)
	for sy := 0; sy < 4; sy++ {
		for sx := 0; sx < 4; sx++ {
			if (sx+sy)%2 == 0 {
				continue
			}
			for y := 0; y < sq; y++ {
				for x := 0; x < sq; x++ {
					img.Set(margin+sy*sq+y, margin+sx*sq+x, 0, 0)
				}
			}
		}
	}
	corners, found := calib3d.FindChessboardCorners(img, [2]int{3, 3})
	fmt.Printf("found=%v n=%d\n", found, len(corners))
```

```
found=true n=9
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func CalibrateCamera(objectPoints [][][3]float64, imagePoints [][][2]float64, imageWidth, imageHeight int) (K [3][3]float64, dist []float64, rvecs, tvecs [][3]float64, rms float64, ok bool)` | CalibrateCamera estimates the intrinsic parameters and per-view extrinsic poses of a camera from several views of a planar calibration target, using… |
| `func ComputeCorrespondEpilines(pts [][2]float64, whichImage int, F [3][3]float64) [][3]float64` | ComputeCorrespondEpilines computes, for each input image point, the epipolar line it induces in the other image of a stereo pair described by the… |
| `func ConvertPointsFromHomogeneous(pts [][3]float64) [][2]float64` | ConvertPointsFromHomogeneous performs the perspective division that maps each homogeneous 3-vector (x, y, w) back to the inhomogeneous 2D point (x/w,… |
| `func ConvertPointsToHomogeneous(pts [][2]float64) [][3]float64` | ConvertPointsToHomogeneous appends a unit homogeneous coordinate to each 2D point, converting an inhomogeneous point set (x, y) into its homogeneous… |
| `func DecomposeEssentialMat(E [3][3]float64) (R1, R2 [3][3]float64, t [3]float64)` | DecomposeEssentialMat factorises an essential matrix into the two possible rotations and the (sign-ambiguous, unit-norm) translation direction that… |
| `func DecomposeHomographyMat(H [3][3]float64, K [3][3]float64) (rotations [][3][3]float64, translations [][3]float64, normals [][3]float64, ok bool)` | DecomposeHomographyMat factorises a plane-induced image homography H into the rigid motions and plane normals consistent with it. |
| `func DrawChessboardCorners(img *cv.Mat, patternSize [2]int, corners [][2]float64, found bool)` | DrawChessboardCorners renders detected chessboard corners onto img in place, mirroring OpenCV's visualisation. |
| `func FindChessboardCorners(img *cv.Mat, patternSize [2]int) (corners [][2]float64, found bool)` | FindChessboardCorners locates the inner corners of a chessboard calibration pattern in an image. |
| `func FindEssentialMat(pts1, pts2 [][2]float64, K [3][3]float64) (E [3][3]float64, ok bool)` | FindEssentialMat estimates the 3×3 essential matrix relating two calibrated views from corresponding image points. |
| `func FindFundamentalMat(pts1, pts2 []cv.Point) (F [3][3]float64, ok bool)` | FindFundamentalMat estimates the 3×3 fundamental matrix F relating two views from corresponding points, using the normalized eight-point algorithm. |
| `func FindHomography(src, dst []cv.Point, method int, ransacThresh float64) (H [3][3]float64, inliers []bool)` | FindHomography estimates the 3×3 projective transform H that maps the source points src onto the destination points dst, so that for each i the… |
| `func GetOptimalNewCameraMatrix(K [3][3]float64, dist []float64, imageSize [2]int, alpha float64) (newK [3][3]float64, roi [4]int)` | GetOptimalNewCameraMatrix computes a new camera matrix that controls how much of the distorted image is retained after undistortion, via the… |
| `func InitUndistortRectifyMap(K [3][3]float64, dist []float64, R [3][3]float64, newK [3][3]float64, imageSize [2]int) (mapX, mapY [][]float64)` | InitUndistortRectifyMap builds the per-pixel remap tables that jointly undo lens distortion and apply a rectification rotation. |
| `func ProjectPoints(objPts [][3]float64, rvec, tvec [3]float64, K [3][3]float64, dist []float64) []cv.Point` | ProjectPoints projects a set of 3D object points into the image plane of a camera. |
| `func RecoverPose(E [3][3]float64, pts1, pts2 [][2]float64, K [3][3]float64) (R [3][3]float64, t [3]float64, good int)` | RecoverPose recovers the relative camera pose (rotation R and unit-norm translation direction t) between two calibrated views from an essential… |
| `func RodriguesToMatrix(rvec [3]float64) [3][3]float64` | RodriguesToMatrix converts a rotation vector (axis-angle, whose direction is the rotation axis and whose magnitude is the rotation angle in radians)… |
| `func RodriguesToVector(r [3][3]float64) [3]float64` | RodriguesToVector converts a 3×3 rotation matrix into the equivalent rotation vector (axis-angle) via the inverse of Rodrigues' formula. |
| `func SolvePnP(objPts [][3]float64, imgPts [][2]float64, K [3][3]float64, dist []float64) (rvec, tvec [3]float64, ok bool)` | SolvePnP recovers the pose (rotation vector rvec and translation tvec) of a calibrated camera from 3D object points and their 2D image projections. |
| `func SolvePnPPlanar(objPts [][3]float64, imgPts []cv.Point, K [3][3]float64) (rvec, tvec [3]float64, ok bool)` | SolvePnPPlanar recovers the pose (rotation vector and translation) of a calibrated camera observing a planar object. |
| `func SolvePnPRansac(objPts [][3]float64, imgPts [][2]float64, K [3][3]float64, dist []float64, reprojThresh float64) (rvec, tvec [3]float64, inliers []bool, ok bool)` | SolvePnPRansac robustly recovers camera pose from 3D–2D correspondences that may contain gross outliers, using Random Sample Consensus around… |
| `func StereoCalibrate(objectPoints [][][3]float64, imagePoints1, imagePoints2 [][][2]float64, K1 [3][3]float64, d1 []float64, K2 [3][3]float64, d2 []float64) (R [3][3]float64, T [3]float64, rms float64, ok bool)` | StereoCalibrate estimates the relative pose (rotation R and translation T) that maps points from the first camera's coordinate frame to the second's,… |
| `func StereoRectify(K1 [3][3]float64, d1 []float64, K2 [3][3]float64, d2 []float64, imageSize [2]int, R [3][3]float64, T [3]float64) (R1, R2 [3][3]float64, P1, P2 [3][4]float64, Q [4][4]float64)` | StereoRectify computes the rectification rotations and rectified projection matrices that make a calibrated stereo pair row-aligned, so that a… |
| `func TriangulatePoints(P1, P2 [3][4]float64, pts1, pts2 []cv.Point) [][3]float64` | TriangulatePoints reconstructs 3D points from their projections in two views by linear (Direct Linear Transform) triangulation. |
| `func Undistort(img *cv.Mat, K [3][3]float64, dist []float64) *cv.Mat` | Undistort removes lens distortion from an image, returning a new `cv.Mat` of the same size and channel count as img. |

### Types

| Type | What it is |
| --- | --- |
| `CameraMatrix` | CameraMatrix holds the intrinsic parameters of a pinhole camera: the focal lengths Fx, Fy (in pixels) and the principal point (Cx, Cy). |
| `DistCoeffs` | DistCoeffs holds the Brown–Conrady lens distortion coefficients: the radial terms K1, K2, K3 and the tangential terms P1, P2. |

<details>
<summary><code>CameraMatrix</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewCameraMatrix(k [3][3]float64) CameraMatrix` | NewCameraMatrix builds a `CameraMatrix` from a 3×3 intrinsic matrix, extracting the focal lengths and principal point. |
| `func (c CameraMatrix) Matrix() [3][3]float64` | Matrix returns the intrinsic parameters as a 3×3 matrix in row-major order. |

</details>

<details>
<summary><code>DistCoeffs</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func (d DistCoeffs) Slice() []float64` | Slice returns the coefficients in OpenCV's canonical ordering [K1, K2, P1, P2, K3], suitable for passing to `ProjectPoints` and `Undistort`. |

</details>

### Constants

`Image1`, `Image2`, `MethodDirect`, `MethodRANSAC`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/calib3d).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
