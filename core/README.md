# core — provides the small fixed-size value types from OpenCV's core module

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/core.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/core)

Package core provides the small fixed-size value types from OpenCV's core
module — the concrete point, vector, matrix, size, rectangle and helper
types that cv2 exposes as templated typedefs (cv::Point_, cv::Point3_,
cv::Size_, cv::Rect_, cv::Vec, cv::Matx, cv::Scalar, cv::Complex, cv::Range,
cv::RotatedRect, cv::KeyPoint, cv::DMatch and cv::TermCriteria).

The root cv package carries a handful of these (Point, Rect, Scalar) tuned for
image drawing; this package supplies the full family of numeric variants with
their arithmetic, so numerical and geometric code can port from OpenCV with
the same type names and operations. Everything here is written against the Go
standard library only — no cgo and no third-party dependencies — and is
deterministic and allocation-light: the vector and matrix types are plain
fixed-length arrays passed by value.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/core"
```

## Usage

This is the package's own `ExampleAffine3d`, so it compiles and its output is
asserted on every `go test ./core/`.

```go
a := core.NewAffine3d(
		core.RodriguesToMatrixd(core.Vec3d{0, 0, math.Pi / 2}),
		core.Vec3d{1, 2, 3},
	)
	p := a.TransformPoint(core.Pt3d(1, 0, 0))
	fmt.Printf("%.3f %.3f %.3f\n", p.X, p.Y, p.Z)

	q := a.Inv().TransformPoint(p)
	fmt.Printf("%.3f %.3f %.3f\n", q.X, q.Y, q.Z)
```

```
1.000 3.000 3.000
1.000 0.000 0.000
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func Angle2f(a, b Point2f) float64` | Angle2f returns the direction of the vector from a to b in radians, measured with math.Atan2. |
| `func Dist2d(a, b Point2d) float64` | Dist2d returns the Euclidean distance between two float64 points. |
| `func Dist2f(a, b Point2f) float64` | Dist2f returns the Euclidean distance between two float32 points. |
| `func Dist2i(a, b Point2i) float64` | Dist2i returns the Euclidean distance between two integer points. |
| `func Dist3d(a, b Point3d) float64` | Dist3d returns the Euclidean distance between two 3D float64 points. |
| `func IsConvex2f(pts []Point2f) bool` | IsConvex2f reports whether the closed polygon is convex, i.e. |
| `func ManhattanDist2i(a, b Point2i) int` | ManhattanDist2i returns the L1 (city-block) distance between two integer points. |
| `func Perimeter2f(pts []Point2f, closed bool) float64` | Perimeter2f returns the total edge length of a point sequence. |
| `func PointInTriangle2f(p, a, b, c Point2f) bool` | PointInTriangle2f reports whether point p lies inside (or on the boundary of) triangle a,b,c. |
| `func PointToSegmentDist2f(p, a, b Point2f) float64` | PointToSegmentDist2f returns the shortest distance from point p to the segment a-b. |
| `func PolygonArea2f(pts []Point2f) float64` | PolygonArea2f returns the absolute area of the polygon. |
| `func SignedPolygonArea2f(pts []Point2f) float64` | SignedPolygonArea2f returns the signed area of the polygon by the shoelace formula; the sign is positive when the vertices wind counter-clockwise in… |

### Types

| Type | What it is |
| --- | --- |
| `Affine3d` | Affine3d is a 3D affine transform stored as a 3x3 rotation/scale matrix and a translation vector, mirroring cv::Affine3d. |
| `Affine3f` | Affine3f is the float32 analogue of Affine3d, mirroring cv::Affine3f. |
| `Complexd` | Complexd is a double-precision complex number, mirroring cv::Complexd. |
| `Complexf` | Complexf is a single-precision complex number, mirroring cv::Complexf. |
| `DMatch` | DMatch links a query descriptor to a train descriptor with a distance, mirroring cv::DMatch. |
| `KeyPoint` | KeyPoint is a salient image point produced by a feature detector, mirroring cv::KeyPoint. |
| `Matx12d` | Matx12d is a 1x2 matrix of float64 stored row-major, mirroring cv::Matx12d. |
| `Matx12f` | Matx12f is a 1x2 matrix of float32 stored row-major, mirroring cv::Matx12f. |
| `Matx13d` | Matx13d is a 1x3 matrix of float64 stored row-major, mirroring cv::Matx13d. |
| `Matx13f` | Matx13f is a 1x3 matrix of float32 stored row-major, mirroring cv::Matx13f. |
| `Matx14d` | Matx14d is a 1x4 matrix of float64 stored row-major, mirroring cv::Matx14d. |
| `Matx14f` | Matx14f is a 1x4 matrix of float32 stored row-major, mirroring cv::Matx14f. |
| `Matx16d` | Matx16d is a 1x6 matrix of float64 stored row-major, mirroring cv::Matx16d. |
| `Matx16f` | Matx16f is a 1x6 matrix of float32 stored row-major, mirroring cv::Matx16f. |
| `Matx21d` | Matx21d is a 2x1 matrix of float64 stored row-major, mirroring cv::Matx21d. |
| `Matx21f` | Matx21f is a 2x1 matrix of float32 stored row-major, mirroring cv::Matx21f. |
| `Matx22d` | Matx22d is a 2x2 matrix of float64 stored row-major, mirroring cv::Matx22d. |
| `Matx22f` | Matx22f is a 2x2 matrix of float32 stored row-major, mirroring cv::Matx22f. |
| `Matx23d` | Matx23d is a 2x3 matrix of float64 stored row-major, mirroring cv::Matx23d. |
| `Matx23f` | Matx23f is a 2x3 matrix of float32 stored row-major, mirroring cv::Matx23f. |
| `Matx31d` | Matx31d is a 3x1 matrix of float64 stored row-major, mirroring cv::Matx31d. |
| `Matx31f` | Matx31f is a 3x1 matrix of float32 stored row-major, mirroring cv::Matx31f. |
| `Matx32d` | Matx32d is a 3x2 matrix of float64 stored row-major, mirroring cv::Matx32d. |
| `Matx32f` | Matx32f is a 3x2 matrix of float32 stored row-major, mirroring cv::Matx32f. |
| `Matx33d` | Matx33d is a 3x3 matrix of float64 stored row-major, mirroring cv::Matx33d. |
| `Matx33f` | Matx33f is a 3x3 matrix of float32 stored row-major, mirroring cv::Matx33f. |
| `Matx34d` | Matx34d is a 3x4 matrix of float64 stored row-major, mirroring cv::Matx34d. |
| `Matx34f` | Matx34f is a 3x4 matrix of float32 stored row-major, mirroring cv::Matx34f. |
| `Matx41d` | Matx41d is a 4x1 matrix of float64 stored row-major, mirroring cv::Matx41d. |
| `Matx41f` | Matx41f is a 4x1 matrix of float32 stored row-major, mirroring cv::Matx41f. |
| `Matx43d` | Matx43d is a 4x3 matrix of float64 stored row-major, mirroring cv::Matx43d. |
| `Matx43f` | Matx43f is a 4x3 matrix of float32 stored row-major, mirroring cv::Matx43f. |
| `Matx44d` | Matx44d is a 4x4 matrix of float64 stored row-major, mirroring cv::Matx44d. |
| `Matx44f` | Matx44f is a 4x4 matrix of float32 stored row-major, mirroring cv::Matx44f. |
| `Matx61d` | Matx61d is a 6x1 matrix of float64 stored row-major, mirroring cv::Matx61d. |
| `Matx61f` | Matx61f is a 6x1 matrix of float32 stored row-major, mirroring cv::Matx61f. |
| `Matx66d` | Matx66d is a 6x6 matrix of float64 stored row-major, mirroring cv::Matx66d. |
| `Matx66f` | Matx66f is a 6x6 matrix of float32 stored row-major, mirroring cv::Matx66f. |
| `Point2d` | Point2d is a 2D point with float64 coordinates, mirroring cv::Point2d. |
| `Point2f` | Point2f is a 2D point with float32 coordinates, mirroring cv::Point2f. |
| `Point2i` | Point2i is a 2D point with integer coordinates, mirroring cv::Point2i (a.k.a. |
| `Point3d` | Point3d is a 3D point with float64 coordinates, mirroring cv::Point3d. |
| `Point3f` | Point3f is a 3D point with float32 coordinates, mirroring cv::Point3f. |
| `Point3i` | Point3i is a 3D point with integer coordinates, mirroring cv::Point3i. |
| `Quatd` | Quatd is a double-precision quaternion w + xi + yj + zk, mirroring cv::Quatd. |
| `Quatf` | Quatf is the float32 analogue of Quatd, mirroring cv::Quatf. |
| `RNG` | RNG is a fast pseudo-random number generator using the multiply-with-carry scheme of OpenCV's cv::RNG. |
| `RNGMT19937` | RNGMT19937 is a Mersenne Twister generator matching OpenCV's cv::RNG_MT19937, offering a longer period than `RNG`. |
| `Range` | Range is a half-open integer interval [Start, End), mirroring cv::Range. |
| `Rect2d` | Rect2d is an axis-aligned rectangle with float64 fields, mirroring cv::Rect2d. |
| `Rect2f` | Rect2f is an axis-aligned rectangle with float32 fields, mirroring cv::Rect2f. |
| `Rect2i` | Rect2i is an axis-aligned rectangle with an integer top-left corner and integer dimensions, mirroring cv::Rect. |
| `RotatedRect` | RotatedRect is a rectangle rotated about its centre, mirroring cv::RotatedRect. |
| `Scalar` | Scalar is a 4-element double vector, mirroring cv::Scalar. |
| `Size2d` | Size2d is a width/height pair with float64 dimensions, mirroring cv::Size2d. |
| `Size2f` | Size2f is a width/height pair with float32 dimensions, mirroring cv::Size2f. |
| `Size2i` | Size2i is a width/height pair with integer dimensions, mirroring cv::Size. |
| `TermCriteria` | TermCriteria describes when an iterative algorithm should stop, mirroring cv::TermCriteria. |
| `TermCriteriaType` | TermCriteriaType is a bit flag selecting the stopping conditions of an iterative algorithm, mirroring cv::TermCriteria::Type. |
| `Vec2b` | Vec2b is a fixed-length vector of 2 uint8 values, mirroring OpenCV's cv::Vec2b. |
| `Vec2d` | Vec2d is a fixed-length vector of 2 float64 values, mirroring OpenCV's cv::Vec2d. |
| `Vec2f` | Vec2f is a fixed-length vector of 2 float32 values, mirroring OpenCV's cv::Vec2f. |
| `Vec2i` | Vec2i is a fixed-length vector of 2 int values, mirroring OpenCV's cv::Vec2i. |
| `Vec2s` | Vec2s is a fixed-length vector of 2 int16 values, mirroring OpenCV's cv::Vec2s. |
| `Vec2w` | Vec2w is a fixed-length vector of 2 uint16 values, mirroring OpenCV's cv::Vec2w. |
| `Vec3b` | Vec3b is a fixed-length vector of 3 uint8 values, mirroring OpenCV's cv::Vec3b. |
| `Vec3d` | Vec3d is a fixed-length vector of 3 float64 values, mirroring OpenCV's cv::Vec3d. |
| `Vec3f` | Vec3f is a fixed-length vector of 3 float32 values, mirroring OpenCV's cv::Vec3f. |
| `Vec3i` | Vec3i is a fixed-length vector of 3 int values, mirroring OpenCV's cv::Vec3i. |
| `Vec3s` | Vec3s is a fixed-length vector of 3 int16 values, mirroring OpenCV's cv::Vec3s. |
| `Vec3w` | Vec3w is a fixed-length vector of 3 uint16 values, mirroring OpenCV's cv::Vec3w. |
| `Vec4b` | Vec4b is a fixed-length vector of 4 uint8 values, mirroring OpenCV's cv::Vec4b. |
| `Vec4d` | Vec4d is a fixed-length vector of 4 float64 values, mirroring OpenCV's cv::Vec4d. |
| `Vec4f` | Vec4f is a fixed-length vector of 4 float32 values, mirroring OpenCV's cv::Vec4f. |
| `Vec4i` | Vec4i is a fixed-length vector of 4 int values, mirroring OpenCV's cv::Vec4i. |
| `Vec4s` | Vec4s is a fixed-length vector of 4 int16 values, mirroring OpenCV's cv::Vec4s. |
| `Vec4w` | Vec4w is a fixed-length vector of 4 uint16 values, mirroring OpenCV's cv::Vec4w. |
| `Vec6d` | Vec6d is a fixed-length vector of 6 float64 values, mirroring OpenCV's cv::Vec6d. |
| `Vec6f` | Vec6f is a fixed-length vector of 6 float32 values, mirroring OpenCV's cv::Vec6f. |
| `Vec6i` | Vec6i is a fixed-length vector of 6 int values, mirroring OpenCV's cv::Vec6i. |
| `Vec8i` | Vec8i is a fixed-length vector of 8 int values, mirroring OpenCV's cv::Vec8i. |

<details>
<summary><code>Affine3d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Affine3dIdentity() Affine3d` | Affine3dIdentity returns the identity transform. |
| `func NewAffine3d(r Matx33d, t Vec3d) Affine3d` | NewAffine3d builds an Affine3d from a linear matrix and a translation. |
| `func (a Affine3d) Concatenate(o Affine3d) Affine3d` | Concatenate returns the composition a∘o, the transform that first applies o then a. |
| `func (a Affine3d) Inv() Affine3d` | Inv returns the inverse transform. |
| `func (a Affine3d) Rotate(r Matx33d) Affine3d` | Rotate returns the transform with an extra rotation r applied before it. |
| `func (a Affine3d) Rotation() Matx33d` | Rotation returns the linear (rotation/scale) part of the transform. |
| `func (a Affine3d) String() string` | String renders the transform. |
| `func (a Affine3d) ToMatx34d() Matx34d` | ToMatx34d returns the transform as a 3x4 matrix [R \| T]. |
| `func (a Affine3d) TransformPoint(p Point3d) Point3d` | TransformPoint applies the transform to point p (R*p + T). |
| `func (a Affine3d) TransformVec(v Vec3d) Vec3d` | TransformVec applies only the linear part to v (no translation). |
| `func (a Affine3d) Translate(t Vec3d) Affine3d` | Translate returns the transform with an extra translation applied after it. |
| `func (a Affine3d) Translation() Vec3d` | Translation returns the translation part of the transform. |

</details>

<details>
<summary><code>Affine3f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Affine3fIdentity() Affine3f` | Affine3fIdentity returns the identity transform. |
| `func NewAffine3f(r Matx33f, t Vec3f) Affine3f` | NewAffine3f builds an Affine3f from a linear matrix and a translation. |
| `func (a Affine3f) Concatenate(o Affine3f) Affine3f` | Concatenate returns the composition a∘o (apply o then a). |
| `func (a Affine3f) Inv() Affine3f` | Inv returns the inverse transform, or the identity when singular. |
| `func (a Affine3f) Rotation() Matx33f` | Rotation returns the linear part of the transform. |
| `func (a Affine3f) String() string` | String renders the transform. |
| `func (a Affine3f) ToMatx34f() Matx34f` | ToMatx34f returns the transform as a 3x4 matrix [R \| T]. |
| `func (a Affine3f) TransformPoint(p Point3f) Point3f` | TransformPoint applies the transform to point p (R*p + T). |
| `func (a Affine3f) TransformVec(v Vec3f) Vec3f` | TransformVec applies only the linear part to v. |
| `func (a Affine3f) Translation() Vec3f` | Translation returns the translation part of the transform. |

</details>

<details>
<summary><code>Complexd</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewComplexd(re, im float64) Complexd` | NewComplexd builds a Complexd from real and imaginary parts. |
| `func (c Complexd) Abs() float64` | Abs returns the modulus \|c\|. |
| `func (c Complexd) Add(o Complexd) Complexd` | Add returns c+o. |
| `func (c Complexd) Arg() float64` | Arg returns the argument (phase angle) of c in radians. |
| `func (c Complexd) Conj() Complexd` | Conj returns the complex conjugate. |
| `func (c Complexd) Div(o Complexd) Complexd` | Div returns the complex quotient c/o. |
| `func (c Complexd) Mul(o Complexd) Complexd` | Mul returns the complex product c*o. |
| `func (c Complexd) String() string` | String renders the number as re+imi. |
| `func (c Complexd) Sub(o Complexd) Complexd` | Sub returns c-o. |

</details>

<details>
<summary><code>Complexf</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewComplexf(re, im float32) Complexf` | NewComplexf builds a Complexf from real and imaginary parts. |
| `func (c Complexf) Abs() float64` | Abs returns the modulus \|c\|. |
| `func (c Complexf) Add(o Complexf) Complexf` | Add returns c+o. |
| `func (c Complexf) Arg() float64` | Arg returns the argument (phase angle) of c in radians. |
| `func (c Complexf) Conj() Complexf` | Conj returns the complex conjugate. |
| `func (c Complexf) Div(o Complexf) Complexf` | Div returns the complex quotient c/o. |
| `func (c Complexf) Mul(o Complexf) Complexf` | Mul returns the complex product c*o. |
| `func (c Complexf) String() string` | String renders the number as re+imi. |
| `func (c Complexf) Sub(o Complexf) Complexf` | Sub returns c-o. |

</details>

<details>
<summary><code>DMatch</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewDMatch(queryIdx, trainIdx, imgIdx int, distance float64) DMatch` | NewDMatch builds a DMatch from the query and train indices and a distance. |
| `func (m DMatch) Less(o DMatch) bool` | Less reports whether m sorts before o, ordering by ascending distance the way cv::DMatch::operator< does. |

</details>

<details>
<summary><code>KeyPoint</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewKeyPoint(x, y, size, angle, response float64, octave, classID int) KeyPoint` | NewKeyPoint builds a KeyPoint at (x, y) with the given diameter, angle, response, octave and class id. |
| `func (k KeyPoint) Overlap(o KeyPoint) float64` | Overlap returns the intersection-over-union of the two keypoint regions, treated as discs of radius Size/2, matching cv::KeyPoint::overlap. |

</details>

<details>
<summary><code>Matx12d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx12d(e0 float64, e1 float64) Matx12d` | NewMatx12d builds a Matx12d from 2 row-major elements. |
| `func (m Matx12d) Add(o Matx12d) Matx12d` | Add returns the element-wise sum m+o. |
| `func (m Matx12d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx12d) Cols() int` | Cols returns the column count (2). |
| `func (m Matx12d) Rows() int` | Rows returns the row count (1). |
| `func (m Matx12d) Scale(s float64) Matx12d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx12d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx12d) Sub(o Matx12d) Matx12d` | Sub returns the element-wise difference m-o. |
| `func (m Matx12d) Transpose() Matx21d` | Transpose returns the 2x1 transpose as a Matx21d. |

</details>

<details>
<summary><code>Matx12f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx12f(e0 float32, e1 float32) Matx12f` | NewMatx12f builds a Matx12f from 2 row-major elements. |
| `func (m Matx12f) Add(o Matx12f) Matx12f` | Add returns the element-wise sum m+o. |
| `func (m Matx12f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx12f) Cols() int` | Cols returns the column count (2). |
| `func (m Matx12f) Rows() int` | Rows returns the row count (1). |
| `func (m Matx12f) Scale(s float32) Matx12f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx12f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx12f) Sub(o Matx12f) Matx12f` | Sub returns the element-wise difference m-o. |
| `func (m Matx12f) Transpose() Matx21f` | Transpose returns the 2x1 transpose as a Matx21f. |

</details>

<details>
<summary><code>Matx13d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx13d(e0 float64, e1 float64, e2 float64) Matx13d` | NewMatx13d builds a Matx13d from 3 row-major elements. |
| `func (m Matx13d) Add(o Matx13d) Matx13d` | Add returns the element-wise sum m+o. |
| `func (m Matx13d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx13d) Cols() int` | Cols returns the column count (3). |
| `func (m Matx13d) Rows() int` | Rows returns the row count (1). |
| `func (m Matx13d) Scale(s float64) Matx13d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx13d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx13d) Sub(o Matx13d) Matx13d` | Sub returns the element-wise difference m-o. |
| `func (m Matx13d) Transpose() Matx31d` | Transpose returns the 3x1 transpose as a Matx31d. |

</details>

<details>
<summary><code>Matx13f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx13f(e0 float32, e1 float32, e2 float32) Matx13f` | NewMatx13f builds a Matx13f from 3 row-major elements. |
| `func (m Matx13f) Add(o Matx13f) Matx13f` | Add returns the element-wise sum m+o. |
| `func (m Matx13f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx13f) Cols() int` | Cols returns the column count (3). |
| `func (m Matx13f) Rows() int` | Rows returns the row count (1). |
| `func (m Matx13f) Scale(s float32) Matx13f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx13f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx13f) Sub(o Matx13f) Matx13f` | Sub returns the element-wise difference m-o. |
| `func (m Matx13f) Transpose() Matx31f` | Transpose returns the 3x1 transpose as a Matx31f. |

</details>

<details>
<summary><code>Matx14d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx14d(e0 float64, e1 float64, e2 float64, e3 float64) Matx14d` | NewMatx14d builds a Matx14d from 4 row-major elements. |
| `func (m Matx14d) Add(o Matx14d) Matx14d` | Add returns the element-wise sum m+o. |
| `func (m Matx14d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx14d) Cols() int` | Cols returns the column count (4). |
| `func (m Matx14d) Rows() int` | Rows returns the row count (1). |
| `func (m Matx14d) Scale(s float64) Matx14d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx14d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx14d) Sub(o Matx14d) Matx14d` | Sub returns the element-wise difference m-o. |
| `func (m Matx14d) Transpose() Matx41d` | Transpose returns the 4x1 transpose as a Matx41d. |

</details>

<details>
<summary><code>Matx14f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx14f(e0 float32, e1 float32, e2 float32, e3 float32) Matx14f` | NewMatx14f builds a Matx14f from 4 row-major elements. |
| `func (m Matx14f) Add(o Matx14f) Matx14f` | Add returns the element-wise sum m+o. |
| `func (m Matx14f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx14f) Cols() int` | Cols returns the column count (4). |
| `func (m Matx14f) Rows() int` | Rows returns the row count (1). |
| `func (m Matx14f) Scale(s float32) Matx14f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx14f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx14f) Sub(o Matx14f) Matx14f` | Sub returns the element-wise difference m-o. |
| `func (m Matx14f) Transpose() Matx41f` | Transpose returns the 4x1 transpose as a Matx41f. |

</details>

<details>
<summary><code>Matx16d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx16d(e0 float64, e1 float64, e2 float64, e3 float64, e4 float64, e5 float64) Matx16d` | NewMatx16d builds a Matx16d from 6 row-major elements. |
| `func (m Matx16d) Add(o Matx16d) Matx16d` | Add returns the element-wise sum m+o. |
| `func (m Matx16d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx16d) Cols() int` | Cols returns the column count (6). |
| `func (m Matx16d) Rows() int` | Rows returns the row count (1). |
| `func (m Matx16d) Scale(s float64) Matx16d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx16d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx16d) Sub(o Matx16d) Matx16d` | Sub returns the element-wise difference m-o. |
| `func (m Matx16d) Transpose() Matx61d` | Transpose returns the 6x1 transpose as a Matx61d. |

</details>

<details>
<summary><code>Matx16f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx16f(e0 float32, e1 float32, e2 float32, e3 float32, e4 float32, e5 float32) Matx16f` | NewMatx16f builds a Matx16f from 6 row-major elements. |
| `func (m Matx16f) Add(o Matx16f) Matx16f` | Add returns the element-wise sum m+o. |
| `func (m Matx16f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx16f) Cols() int` | Cols returns the column count (6). |
| `func (m Matx16f) Rows() int` | Rows returns the row count (1). |
| `func (m Matx16f) Scale(s float32) Matx16f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx16f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx16f) Sub(o Matx16f) Matx16f` | Sub returns the element-wise difference m-o. |
| `func (m Matx16f) Transpose() Matx61f` | Transpose returns the 6x1 transpose as a Matx61f. |

</details>

<details>
<summary><code>Matx21d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx21d(e0 float64, e1 float64) Matx21d` | NewMatx21d builds a Matx21d from 2 row-major elements. |
| `func (m Matx21d) Add(o Matx21d) Matx21d` | Add returns the element-wise sum m+o. |
| `func (m Matx21d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx21d) Cols() int` | Cols returns the column count (1). |
| `func (m Matx21d) Rows() int` | Rows returns the row count (2). |
| `func (m Matx21d) Scale(s float64) Matx21d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx21d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx21d) Sub(o Matx21d) Matx21d` | Sub returns the element-wise difference m-o. |
| `func (m Matx21d) Transpose() Matx12d` | Transpose returns the 1x2 transpose as a Matx12d. |

</details>

<details>
<summary><code>Matx21f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx21f(e0 float32, e1 float32) Matx21f` | NewMatx21f builds a Matx21f from 2 row-major elements. |
| `func (m Matx21f) Add(o Matx21f) Matx21f` | Add returns the element-wise sum m+o. |
| `func (m Matx21f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx21f) Cols() int` | Cols returns the column count (1). |
| `func (m Matx21f) Rows() int` | Rows returns the row count (2). |
| `func (m Matx21f) Scale(s float32) Matx21f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx21f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx21f) Sub(o Matx21f) Matx21f` | Sub returns the element-wise difference m-o. |
| `func (m Matx21f) Transpose() Matx12f` | Transpose returns the 1x2 transpose as a Matx12f. |

</details>

<details>
<summary><code>Matx22d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Matx22dIdentity() Matx22d` | Matx22dIdentity returns the 2x2 identity matrix. |
| `func NewMatx22d(e0 float64, e1 float64, e2 float64, e3 float64) Matx22d` | NewMatx22d builds a Matx22d from 4 row-major elements. |
| `func (m Matx22d) Add(o Matx22d) Matx22d` | Add returns the element-wise sum m+o. |
| `func (m Matx22d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx22d) Cols() int` | Cols returns the column count (2). |
| `func (m Matx22d) Determinant() float64` | Determinant returns the determinant computed in float64 by Gaussian elimination with partial pivoting. |
| `func (m Matx22d) Inverse() (Matx22d, bool)` | Inverse returns the matrix inverse and true, or the zero matrix and false when the matrix is singular. |
| `func (m Matx22d) Mul(o Matx22d) Matx22d` | Mul returns the matrix product m*o. |
| `func (m Matx22d) MulVec(v Vec2d) Vec2d` | MulVec returns the matrix-vector product m*v. |
| `func (m Matx22d) Rows() int` | Rows returns the row count (2). |
| `func (m Matx22d) Scale(s float64) Matx22d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx22d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx22d) Sub(o Matx22d) Matx22d` | Sub returns the element-wise difference m-o. |
| `func (m Matx22d) Trace() float64` | Trace returns the sum of the diagonal elements as float64. |
| `func (m Matx22d) Transpose() Matx22d` | Transpose returns the 2x2 transpose as a Matx22d. |

</details>

<details>
<summary><code>Matx22f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Matx22fIdentity() Matx22f` | Matx22fIdentity returns the 2x2 identity matrix. |
| `func NewMatx22f(e0 float32, e1 float32, e2 float32, e3 float32) Matx22f` | NewMatx22f builds a Matx22f from 4 row-major elements. |
| `func (m Matx22f) Add(o Matx22f) Matx22f` | Add returns the element-wise sum m+o. |
| `func (m Matx22f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx22f) Cols() int` | Cols returns the column count (2). |
| `func (m Matx22f) Determinant() float64` | Determinant returns the determinant computed in float64 by Gaussian elimination with partial pivoting. |
| `func (m Matx22f) Inverse() (Matx22f, bool)` | Inverse returns the matrix inverse and true, or the zero matrix and false when the matrix is singular. |
| `func (m Matx22f) Mul(o Matx22f) Matx22f` | Mul returns the matrix product m*o. |
| `func (m Matx22f) MulVec(v Vec2f) Vec2f` | MulVec returns the matrix-vector product m*v. |
| `func (m Matx22f) Rows() int` | Rows returns the row count (2). |
| `func (m Matx22f) Scale(s float32) Matx22f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx22f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx22f) Sub(o Matx22f) Matx22f` | Sub returns the element-wise difference m-o. |
| `func (m Matx22f) Trace() float64` | Trace returns the sum of the diagonal elements as float64. |
| `func (m Matx22f) Transpose() Matx22f` | Transpose returns the 2x2 transpose as a Matx22f. |

</details>

<details>
<summary><code>Matx23d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx23d(e0 float64, e1 float64, e2 float64, e3 float64, e4 float64, e5 float64) Matx23d` | NewMatx23d builds a Matx23d from 6 row-major elements. |
| `func (m Matx23d) Add(o Matx23d) Matx23d` | Add returns the element-wise sum m+o. |
| `func (m Matx23d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx23d) Cols() int` | Cols returns the column count (3). |
| `func (m Matx23d) Rows() int` | Rows returns the row count (2). |
| `func (m Matx23d) Scale(s float64) Matx23d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx23d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx23d) Sub(o Matx23d) Matx23d` | Sub returns the element-wise difference m-o. |
| `func (m Matx23d) Transpose() Matx32d` | Transpose returns the 3x2 transpose as a Matx32d. |

</details>

<details>
<summary><code>Matx23f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx23f(e0 float32, e1 float32, e2 float32, e3 float32, e4 float32, e5 float32) Matx23f` | NewMatx23f builds a Matx23f from 6 row-major elements. |
| `func (m Matx23f) Add(o Matx23f) Matx23f` | Add returns the element-wise sum m+o. |
| `func (m Matx23f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx23f) Cols() int` | Cols returns the column count (3). |
| `func (m Matx23f) Rows() int` | Rows returns the row count (2). |
| `func (m Matx23f) Scale(s float32) Matx23f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx23f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx23f) Sub(o Matx23f) Matx23f` | Sub returns the element-wise difference m-o. |
| `func (m Matx23f) Transpose() Matx32f` | Transpose returns the 3x2 transpose as a Matx32f. |

</details>

<details>
<summary><code>Matx31d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx31d(e0 float64, e1 float64, e2 float64) Matx31d` | NewMatx31d builds a Matx31d from 3 row-major elements. |
| `func (m Matx31d) Add(o Matx31d) Matx31d` | Add returns the element-wise sum m+o. |
| `func (m Matx31d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx31d) Cols() int` | Cols returns the column count (1). |
| `func (m Matx31d) Rows() int` | Rows returns the row count (3). |
| `func (m Matx31d) Scale(s float64) Matx31d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx31d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx31d) Sub(o Matx31d) Matx31d` | Sub returns the element-wise difference m-o. |
| `func (m Matx31d) Transpose() Matx13d` | Transpose returns the 1x3 transpose as a Matx13d. |

</details>

<details>
<summary><code>Matx31f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx31f(e0 float32, e1 float32, e2 float32) Matx31f` | NewMatx31f builds a Matx31f from 3 row-major elements. |
| `func (m Matx31f) Add(o Matx31f) Matx31f` | Add returns the element-wise sum m+o. |
| `func (m Matx31f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx31f) Cols() int` | Cols returns the column count (1). |
| `func (m Matx31f) Rows() int` | Rows returns the row count (3). |
| `func (m Matx31f) Scale(s float32) Matx31f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx31f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx31f) Sub(o Matx31f) Matx31f` | Sub returns the element-wise difference m-o. |
| `func (m Matx31f) Transpose() Matx13f` | Transpose returns the 1x3 transpose as a Matx13f. |

</details>

<details>
<summary><code>Matx32d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx32d(e0 float64, e1 float64, e2 float64, e3 float64, e4 float64, e5 float64) Matx32d` | NewMatx32d builds a Matx32d from 6 row-major elements. |
| `func (m Matx32d) Add(o Matx32d) Matx32d` | Add returns the element-wise sum m+o. |
| `func (m Matx32d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx32d) Cols() int` | Cols returns the column count (2). |
| `func (m Matx32d) Rows() int` | Rows returns the row count (3). |
| `func (m Matx32d) Scale(s float64) Matx32d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx32d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx32d) Sub(o Matx32d) Matx32d` | Sub returns the element-wise difference m-o. |
| `func (m Matx32d) Transpose() Matx23d` | Transpose returns the 2x3 transpose as a Matx23d. |

</details>

<details>
<summary><code>Matx32f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx32f(e0 float32, e1 float32, e2 float32, e3 float32, e4 float32, e5 float32) Matx32f` | NewMatx32f builds a Matx32f from 6 row-major elements. |
| `func (m Matx32f) Add(o Matx32f) Matx32f` | Add returns the element-wise sum m+o. |
| `func (m Matx32f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx32f) Cols() int` | Cols returns the column count (2). |
| `func (m Matx32f) Rows() int` | Rows returns the row count (3). |
| `func (m Matx32f) Scale(s float32) Matx32f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx32f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx32f) Sub(o Matx32f) Matx32f` | Sub returns the element-wise difference m-o. |
| `func (m Matx32f) Transpose() Matx23f` | Transpose returns the 2x3 transpose as a Matx23f. |

</details>

<details>
<summary><code>Matx33d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Matx33dIdentity() Matx33d` | Matx33dIdentity returns the 3x3 identity matrix. |
| `func NewMatx33d(e0 float64, e1 float64, e2 float64, e3 float64, e4 float64, e5 float64, e6 float64, e7 float64, e8 float64) Matx33d` | NewMatx33d builds a Matx33d from 9 row-major elements. |
| `func RodriguesToMatrixd(rvec Vec3d) Matx33d` | RodriguesToMatrixd converts a rotation vector (axis scaled by angle in radians) to a 3x3 rotation matrix using Rodrigues' formula, mirroring… |
| `func (m Matx33d) Add(o Matx33d) Matx33d` | Add returns the element-wise sum m+o. |
| `func (m Matx33d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx33d) Cols() int` | Cols returns the column count (3). |
| `func (m Matx33d) Determinant() float64` | Determinant returns the determinant computed in float64 by Gaussian elimination with partial pivoting. |
| `func (m Matx33d) Inverse() (Matx33d, bool)` | Inverse returns the matrix inverse and true, or the zero matrix and false when the matrix is singular. |
| `func (m Matx33d) Mul(o Matx33d) Matx33d` | Mul returns the matrix product m*o. |
| `func (m Matx33d) MulVec(v Vec3d) Vec3d` | MulVec returns the matrix-vector product m*v. |
| `func (m Matx33d) Rows() int` | Rows returns the row count (3). |
| `func (m Matx33d) Scale(s float64) Matx33d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx33d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx33d) Sub(o Matx33d) Matx33d` | Sub returns the element-wise difference m-o. |
| `func (m Matx33d) Trace() float64` | Trace returns the sum of the diagonal elements as float64. |
| `func (m Matx33d) Transpose() Matx33d` | Transpose returns the 3x3 transpose as a Matx33d. |

</details>

<details>
<summary><code>Matx33f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Matx33fIdentity() Matx33f` | Matx33fIdentity returns the 3x3 identity matrix. |
| `func NewMatx33f(e0 float32, e1 float32, e2 float32, e3 float32, e4 float32, e5 float32, e6 float32, e7 float32, e8 float32) Matx33f` | NewMatx33f builds a Matx33f from 9 row-major elements. |
| `func (m Matx33f) Add(o Matx33f) Matx33f` | Add returns the element-wise sum m+o. |
| `func (m Matx33f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx33f) Cols() int` | Cols returns the column count (3). |
| `func (m Matx33f) Determinant() float64` | Determinant returns the determinant computed in float64 by Gaussian elimination with partial pivoting. |
| `func (m Matx33f) Inverse() (Matx33f, bool)` | Inverse returns the matrix inverse and true, or the zero matrix and false when the matrix is singular. |
| `func (m Matx33f) Mul(o Matx33f) Matx33f` | Mul returns the matrix product m*o. |
| `func (m Matx33f) MulVec(v Vec3f) Vec3f` | MulVec returns the matrix-vector product m*v. |
| `func (m Matx33f) Rows() int` | Rows returns the row count (3). |
| `func (m Matx33f) Scale(s float32) Matx33f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx33f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx33f) Sub(o Matx33f) Matx33f` | Sub returns the element-wise difference m-o. |
| `func (m Matx33f) Trace() float64` | Trace returns the sum of the diagonal elements as float64. |
| `func (m Matx33f) Transpose() Matx33f` | Transpose returns the 3x3 transpose as a Matx33f. |

</details>

<details>
<summary><code>Matx34d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx34d(e0 float64, e1 float64, e2 float64, e3 float64, e4 float64, e5 float64, e6 float64, e7 float64, e8 float64, e9 float64, e10 float64, e11 float64) Matx34d` | NewMatx34d builds a Matx34d from 12 row-major elements. |
| `func (m Matx34d) Add(o Matx34d) Matx34d` | Add returns the element-wise sum m+o. |
| `func (m Matx34d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx34d) Cols() int` | Cols returns the column count (4). |
| `func (m Matx34d) Rows() int` | Rows returns the row count (3). |
| `func (m Matx34d) Scale(s float64) Matx34d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx34d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx34d) Sub(o Matx34d) Matx34d` | Sub returns the element-wise difference m-o. |
| `func (m Matx34d) Transpose() Matx43d` | Transpose returns the 4x3 transpose as a Matx43d. |

</details>

<details>
<summary><code>Matx34f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx34f(e0 float32, e1 float32, e2 float32, e3 float32, e4 float32, e5 float32, e6 float32, e7 float32, e8 float32, e9 float32, e10 float32, e11 float32) Matx34f` | NewMatx34f builds a Matx34f from 12 row-major elements. |
| `func (m Matx34f) Add(o Matx34f) Matx34f` | Add returns the element-wise sum m+o. |
| `func (m Matx34f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx34f) Cols() int` | Cols returns the column count (4). |
| `func (m Matx34f) Rows() int` | Rows returns the row count (3). |
| `func (m Matx34f) Scale(s float32) Matx34f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx34f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx34f) Sub(o Matx34f) Matx34f` | Sub returns the element-wise difference m-o. |
| `func (m Matx34f) Transpose() Matx43f` | Transpose returns the 4x3 transpose as a Matx43f. |

</details>

<details>
<summary><code>Matx41d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx41d(e0 float64, e1 float64, e2 float64, e3 float64) Matx41d` | NewMatx41d builds a Matx41d from 4 row-major elements. |
| `func (m Matx41d) Add(o Matx41d) Matx41d` | Add returns the element-wise sum m+o. |
| `func (m Matx41d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx41d) Cols() int` | Cols returns the column count (1). |
| `func (m Matx41d) Rows() int` | Rows returns the row count (4). |
| `func (m Matx41d) Scale(s float64) Matx41d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx41d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx41d) Sub(o Matx41d) Matx41d` | Sub returns the element-wise difference m-o. |
| `func (m Matx41d) Transpose() Matx14d` | Transpose returns the 1x4 transpose as a Matx14d. |

</details>

<details>
<summary><code>Matx41f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx41f(e0 float32, e1 float32, e2 float32, e3 float32) Matx41f` | NewMatx41f builds a Matx41f from 4 row-major elements. |
| `func (m Matx41f) Add(o Matx41f) Matx41f` | Add returns the element-wise sum m+o. |
| `func (m Matx41f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx41f) Cols() int` | Cols returns the column count (1). |
| `func (m Matx41f) Rows() int` | Rows returns the row count (4). |
| `func (m Matx41f) Scale(s float32) Matx41f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx41f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx41f) Sub(o Matx41f) Matx41f` | Sub returns the element-wise difference m-o. |
| `func (m Matx41f) Transpose() Matx14f` | Transpose returns the 1x4 transpose as a Matx14f. |

</details>

<details>
<summary><code>Matx43d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx43d(e0 float64, e1 float64, e2 float64, e3 float64, e4 float64, e5 float64, e6 float64, e7 float64, e8 float64, e9 float64, e10 float64, e11 float64) Matx43d` | NewMatx43d builds a Matx43d from 12 row-major elements. |
| `func (m Matx43d) Add(o Matx43d) Matx43d` | Add returns the element-wise sum m+o. |
| `func (m Matx43d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx43d) Cols() int` | Cols returns the column count (3). |
| `func (m Matx43d) Rows() int` | Rows returns the row count (4). |
| `func (m Matx43d) Scale(s float64) Matx43d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx43d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx43d) Sub(o Matx43d) Matx43d` | Sub returns the element-wise difference m-o. |
| `func (m Matx43d) Transpose() Matx34d` | Transpose returns the 3x4 transpose as a Matx34d. |

</details>

<details>
<summary><code>Matx43f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx43f(e0 float32, e1 float32, e2 float32, e3 float32, e4 float32, e5 float32, e6 float32, e7 float32, e8 float32, e9 float32, e10 float32, e11 float32) Matx43f` | NewMatx43f builds a Matx43f from 12 row-major elements. |
| `func (m Matx43f) Add(o Matx43f) Matx43f` | Add returns the element-wise sum m+o. |
| `func (m Matx43f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx43f) Cols() int` | Cols returns the column count (3). |
| `func (m Matx43f) Rows() int` | Rows returns the row count (4). |
| `func (m Matx43f) Scale(s float32) Matx43f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx43f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx43f) Sub(o Matx43f) Matx43f` | Sub returns the element-wise difference m-o. |
| `func (m Matx43f) Transpose() Matx34f` | Transpose returns the 3x4 transpose as a Matx34f. |

</details>

<details>
<summary><code>Matx44d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Matx44dIdentity() Matx44d` | Matx44dIdentity returns the 4x4 identity matrix. |
| `func NewMatx44d(e0 float64, e1 float64, e2 float64, e3 float64, e4 float64, e5 float64, e6 float64, e7 float64, e8 float64, e9 float64, e10 float64, e11 float64, e12 float64, e13 float64, e14 float64, e15 float64) Matx44d` | NewMatx44d builds a Matx44d from 16 row-major elements. |
| `func (m Matx44d) Add(o Matx44d) Matx44d` | Add returns the element-wise sum m+o. |
| `func (m Matx44d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx44d) Cols() int` | Cols returns the column count (4). |
| `func (m Matx44d) Determinant() float64` | Determinant returns the determinant computed in float64 by Gaussian elimination with partial pivoting. |
| `func (m Matx44d) Inverse() (Matx44d, bool)` | Inverse returns the matrix inverse and true, or the zero matrix and false when the matrix is singular. |
| `func (m Matx44d) Mul(o Matx44d) Matx44d` | Mul returns the matrix product m*o. |
| `func (m Matx44d) MulVec(v Vec4d) Vec4d` | MulVec returns the matrix-vector product m*v. |
| `func (m Matx44d) Rows() int` | Rows returns the row count (4). |
| `func (m Matx44d) Scale(s float64) Matx44d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx44d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx44d) Sub(o Matx44d) Matx44d` | Sub returns the element-wise difference m-o. |
| `func (m Matx44d) Trace() float64` | Trace returns the sum of the diagonal elements as float64. |
| `func (m Matx44d) Transpose() Matx44d` | Transpose returns the 4x4 transpose as a Matx44d. |

</details>

<details>
<summary><code>Matx44f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Matx44fIdentity() Matx44f` | Matx44fIdentity returns the 4x4 identity matrix. |
| `func NewMatx44f(e0 float32, e1 float32, e2 float32, e3 float32, e4 float32, e5 float32, e6 float32, e7 float32, e8 float32, e9 float32, e10 float32, e11 float32, e12 float32, e13 float32, e14 float32, e15 float32) Matx44f` | NewMatx44f builds a Matx44f from 16 row-major elements. |
| `func (m Matx44f) Add(o Matx44f) Matx44f` | Add returns the element-wise sum m+o. |
| `func (m Matx44f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx44f) Cols() int` | Cols returns the column count (4). |
| `func (m Matx44f) Determinant() float64` | Determinant returns the determinant computed in float64 by Gaussian elimination with partial pivoting. |
| `func (m Matx44f) Inverse() (Matx44f, bool)` | Inverse returns the matrix inverse and true, or the zero matrix and false when the matrix is singular. |
| `func (m Matx44f) Mul(o Matx44f) Matx44f` | Mul returns the matrix product m*o. |
| `func (m Matx44f) MulVec(v Vec4f) Vec4f` | MulVec returns the matrix-vector product m*v. |
| `func (m Matx44f) Rows() int` | Rows returns the row count (4). |
| `func (m Matx44f) Scale(s float32) Matx44f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx44f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx44f) Sub(o Matx44f) Matx44f` | Sub returns the element-wise difference m-o. |
| `func (m Matx44f) Trace() float64` | Trace returns the sum of the diagonal elements as float64. |
| `func (m Matx44f) Transpose() Matx44f` | Transpose returns the 4x4 transpose as a Matx44f. |

</details>

<details>
<summary><code>Matx61d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx61d(e0 float64, e1 float64, e2 float64, e3 float64, e4 float64, e5 float64) Matx61d` | NewMatx61d builds a Matx61d from 6 row-major elements. |
| `func (m Matx61d) Add(o Matx61d) Matx61d` | Add returns the element-wise sum m+o. |
| `func (m Matx61d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx61d) Cols() int` | Cols returns the column count (1). |
| `func (m Matx61d) Rows() int` | Rows returns the row count (6). |
| `func (m Matx61d) Scale(s float64) Matx61d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx61d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx61d) Sub(o Matx61d) Matx61d` | Sub returns the element-wise difference m-o. |
| `func (m Matx61d) Transpose() Matx16d` | Transpose returns the 1x6 transpose as a Matx16d. |

</details>

<details>
<summary><code>Matx61f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewMatx61f(e0 float32, e1 float32, e2 float32, e3 float32, e4 float32, e5 float32) Matx61f` | NewMatx61f builds a Matx61f from 6 row-major elements. |
| `func (m Matx61f) Add(o Matx61f) Matx61f` | Add returns the element-wise sum m+o. |
| `func (m Matx61f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx61f) Cols() int` | Cols returns the column count (1). |
| `func (m Matx61f) Rows() int` | Rows returns the row count (6). |
| `func (m Matx61f) Scale(s float32) Matx61f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx61f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx61f) Sub(o Matx61f) Matx61f` | Sub returns the element-wise difference m-o. |
| `func (m Matx61f) Transpose() Matx16f` | Transpose returns the 1x6 transpose as a Matx16f. |

</details>

<details>
<summary><code>Matx66d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Matx66dIdentity() Matx66d` | Matx66dIdentity returns the 6x6 identity matrix. |
| `func NewMatx66d(e0 float64, e1 float64, e2 float64, e3 float64, e4 float64, e5 float64, e6 float64, e7 float64, e8 float64, e9 float64, e10 float64, e11 float64, e12 float64, e13 float64, e14 float64, e15 float64, e16 float64, e17 float64, e18 float64, e19 float64, e20 float64, e21 float64, e22 float64, e23 float64, e24 float64, e25 float64, e26 float64, e27 float64, e28 float64, e29 float64, e30 float64, e31 float64, e32 float64, e33 float64, e34 float64, e35 float64) Matx66d` | NewMatx66d builds a Matx66d from 36 row-major elements. |
| `func (m Matx66d) Add(o Matx66d) Matx66d` | Add returns the element-wise sum m+o. |
| `func (m Matx66d) At(i, j int) float64` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx66d) Cols() int` | Cols returns the column count (6). |
| `func (m Matx66d) Determinant() float64` | Determinant returns the determinant computed in float64 by Gaussian elimination with partial pivoting. |
| `func (m Matx66d) Inverse() (Matx66d, bool)` | Inverse returns the matrix inverse and true, or the zero matrix and false when the matrix is singular. |
| `func (m Matx66d) Mul(o Matx66d) Matx66d` | Mul returns the matrix product m*o. |
| `func (m Matx66d) MulVec(v Vec6d) Vec6d` | MulVec returns the matrix-vector product m*v. |
| `func (m Matx66d) Rows() int` | Rows returns the row count (6). |
| `func (m Matx66d) Scale(s float64) Matx66d` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx66d) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx66d) Sub(o Matx66d) Matx66d` | Sub returns the element-wise difference m-o. |
| `func (m Matx66d) Trace() float64` | Trace returns the sum of the diagonal elements as float64. |
| `func (m Matx66d) Transpose() Matx66d` | Transpose returns the 6x6 transpose as a Matx66d. |

</details>

<details>
<summary><code>Matx66f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Matx66fIdentity() Matx66f` | Matx66fIdentity returns the 6x6 identity matrix. |
| `func NewMatx66f(e0 float32, e1 float32, e2 float32, e3 float32, e4 float32, e5 float32, e6 float32, e7 float32, e8 float32, e9 float32, e10 float32, e11 float32, e12 float32, e13 float32, e14 float32, e15 float32, e16 float32, e17 float32, e18 float32, e19 float32, e20 float32, e21 float32, e22 float32, e23 float32, e24 float32, e25 float32, e26 float32, e27 float32, e28 float32, e29 float32, e30 float32, e31 float32, e32 float32, e33 float32, e34 float32, e35 float32) Matx66f` | NewMatx66f builds a Matx66f from 36 row-major elements. |
| `func (m Matx66f) Add(o Matx66f) Matx66f` | Add returns the element-wise sum m+o. |
| `func (m Matx66f) At(i, j int) float32` | At returns the element at row i, column j (0-indexed). |
| `func (m Matx66f) Cols() int` | Cols returns the column count (6). |
| `func (m Matx66f) Determinant() float64` | Determinant returns the determinant computed in float64 by Gaussian elimination with partial pivoting. |
| `func (m Matx66f) Inverse() (Matx66f, bool)` | Inverse returns the matrix inverse and true, or the zero matrix and false when the matrix is singular. |
| `func (m Matx66f) Mul(o Matx66f) Matx66f` | Mul returns the matrix product m*o. |
| `func (m Matx66f) MulVec(v Vec6f) Vec6f` | MulVec returns the matrix-vector product m*v. |
| `func (m Matx66f) Rows() int` | Rows returns the row count (6). |
| `func (m Matx66f) Scale(s float32) Matx66f` | Scale returns the matrix with every element multiplied by s. |
| `func (m Matx66f) String() string` | String renders the matrix elements in row-major order. |
| `func (m Matx66f) Sub(o Matx66f) Matx66f` | Sub returns the element-wise difference m-o. |
| `func (m Matx66f) Trace() float64` | Trace returns the sum of the diagonal elements as float64. |
| `func (m Matx66f) Transpose() Matx66f` | Transpose returns the 6x6 transpose as a Matx66f. |

</details>

<details>
<summary><code>Point2d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Centroid2d(pts []Point2d) Point2d` | Centroid2d returns the arithmetic mean of the points. |
| `func Lerp2d(a, b Point2d, t float64) Point2d` | Lerp2d linearly interpolates from a to b by parameter t. |
| `func Pt2d(x, y float64) Point2d` | Pt2d builds a Point2d from x and y. |
| `func RotatePoint2d(p, center Point2d, angleDeg float64) Point2d` | RotatePoint2d rotates p about center by angleDeg degrees clockwise in image coordinates (y down) and returns the result. |
| `func (p Point2d) Add(o Point2d) Point2d` | Add returns the vector sum p+o. |
| `func (p Point2d) Cross(o Point2d) float64` | Cross returns the 2D cross product (z-component) p×o. |
| `func (p Point2d) Ddot(o Point2d) float64` | Ddot is an alias for Dot kept for symmetry with the integer types. |
| `func (p Point2d) Dot(o Point2d) float64` | Dot returns the dot product p·o. |
| `func (p Point2d) Equals(o Point2d) bool` | Equals reports whether p and o are the same point. |
| `func (p Point2d) Inside(r Rect2d) bool` | Inside reports whether the point lies within rectangle r. |
| `func (p Point2d) Mul(s float64) Point2d` | Mul returns the point scaled by s. |
| `func (p Point2d) Norm() float64` | Norm returns the Euclidean length of the position vector. |
| `func (p Point2d) Round() Point2i` | Round returns the nearest integer point. |
| `func (p Point2d) String() string` | String renders the point as (x, y). |
| `func (p Point2d) Sub(o Point2d) Point2d` | Sub returns the vector difference p-o. |

</details>

<details>
<summary><code>Point2f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Centroid2f(pts []Point2f) Point2f` | Centroid2f returns the arithmetic mean of the points. |
| `func ConvexHull2f(pts []Point2f) []Point2f` | ConvexHull2f computes the convex hull of the points with Andrew's monotone chain algorithm and returns the hull vertices in counter-clockwise order… |
| `func Lerp2f(a, b Point2f, t float64) Point2f` | Lerp2f linearly interpolates from a to b by parameter t (0 gives a, 1 gives b). |
| `func MidPoint2f(a, b Point2f) Point2f` | MidPoint2f returns the midpoint of segment a-b. |
| `func Pt2f(x, y float32) Point2f` | Pt2f builds a Point2f from x and y. |
| `func RotatePoint2f(p, center Point2f, angleDeg float64) Point2f` | RotatePoint2f rotates p about center by angleDeg degrees clockwise in image coordinates (y down) and returns the result. |
| `func SegmentsIntersect2f(p1, p2, p3, p4 Point2f) (Point2f, bool)` | SegmentsIntersect2f reports whether segments p1-p2 and p3-p4 cross and, when they do, returns their intersection point. |
| `func (p Point2f) Add(o Point2f) Point2f` | Add returns the vector sum p+o. |
| `func (p Point2f) Cross(o Point2f) float64` | Cross returns the 2D cross product (z-component) p×o. |
| `func (p Point2f) Ddot(o Point2f) float64` | Ddot returns the dot product accumulated in float64. |
| `func (p Point2f) Dot(o Point2f) float32` | Dot returns the dot product p·o. |
| `func (p Point2f) Equals(o Point2f) bool` | Equals reports whether p and o are the same point. |
| `func (p Point2f) Inside(r Rect2f) bool` | Inside reports whether the point lies within rectangle r. |
| `func (p Point2f) Mul(s float32) Point2f` | Mul returns the point scaled by s. |
| `func (p Point2f) Norm() float64` | Norm returns the Euclidean length of the position vector. |
| `func (p Point2f) Round() Point2i` | Round returns the nearest integer point. |
| `func (p Point2f) String() string` | String renders the point as (x, y). |
| `func (p Point2f) Sub(o Point2f) Point2f` | Sub returns the vector difference p-o. |
| `func (p Point2f) ToPoint2d() Point2d` | ToPoint2d converts to a float64 point. |

</details>

<details>
<summary><code>Point2i</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Clamp2i(p Point2i, r Rect2i) Point2i` | Clamp2i clamps point p to lie within rectangle r (the returned point is on the closed range [X, X+Width-1] x [Y, Y+Height-1]). |
| `func Pt2i(x, y int) Point2i` | Pt2i builds a Point2i from x and y. |
| `func (p Point2i) Add(o Point2i) Point2i` | Add returns the vector sum p+o. |
| `func (p Point2i) Cross(o Point2i) float64` | Cross returns the 2D cross product (z-component) p×o. |
| `func (p Point2i) Ddot(o Point2i) float64` | Ddot returns the dot product accumulated in float64. |
| `func (p Point2i) Dot(o Point2i) int` | Dot returns the integer dot product p·o. |
| `func (p Point2i) Equals(o Point2i) bool` | Equals reports whether p and o are the same point. |
| `func (p Point2i) Inside(r Rect2i) bool` | Inside reports whether the point lies within rectangle r (edges inclusive on the top-left, exclusive on the bottom-right, matching OpenCV). |
| `func (p Point2i) Mul(s int) Point2i` | Mul returns the point scaled by s. |
| `func (p Point2i) Norm() float64` | Norm returns the Euclidean length of the position vector. |
| `func (p Point2i) String() string` | String renders the point as (x, y). |
| `func (p Point2i) Sub(o Point2i) Point2i` | Sub returns the vector difference p-o. |
| `func (p Point2i) ToPoint2d() Point2d` | ToPoint2d converts to a float64 point. |
| `func (p Point2i) ToPoint2f() Point2f` | ToPoint2f converts to a float32 point. |

</details>

<details>
<summary><code>Point3d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Pt3d(x, y, z float64) Point3d` | Pt3d builds a Point3d from x, y and z. |
| `func (p Point3d) Add(o Point3d) Point3d` | Add returns the vector sum p+o. |
| `func (p Point3d) Cross(o Point3d) Point3d` | Cross returns the 3D cross product p×o. |
| `func (p Point3d) Dot(o Point3d) float64` | Dot returns the dot product p·o. |
| `func (p Point3d) Equals(o Point3d) bool` | Equals reports whether p and o are the same point. |
| `func (p Point3d) Mul(s float64) Point3d` | Mul returns the point scaled by s. |
| `func (p Point3d) Norm() float64` | Norm returns the Euclidean length of the position vector. |
| `func (p Point3d) Normalize() Point3d` | Normalize returns the unit vector along p; a zero vector is returned as-is. |
| `func (p Point3d) String() string` | String renders the point as (x, y, z). |
| `func (p Point3d) Sub(o Point3d) Point3d` | Sub returns the vector difference p-o. |

</details>

<details>
<summary><code>Point3f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Pt3f(x, y, z float32) Point3f` | Pt3f builds a Point3f from x, y and z. |
| `func (p Point3f) Add(o Point3f) Point3f` | Add returns the vector sum p+o. |
| `func (p Point3f) Cross(o Point3f) Point3f` | Cross returns the 3D cross product p×o. |
| `func (p Point3f) Dot(o Point3f) float32` | Dot returns the dot product p·o. |
| `func (p Point3f) Equals(o Point3f) bool` | Equals reports whether p and o are the same point. |
| `func (p Point3f) Mul(s float32) Point3f` | Mul returns the point scaled by s. |
| `func (p Point3f) Norm() float64` | Norm returns the Euclidean length of the position vector. |
| `func (p Point3f) Normalize() Point3f` | Normalize returns the unit vector along p; a zero vector is returned as-is. |
| `func (p Point3f) String() string` | String renders the point as (x, y, z). |
| `func (p Point3f) Sub(o Point3f) Point3f` | Sub returns the vector difference p-o. |

</details>

<details>
<summary><code>Point3i</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Pt3i(x, y, z int) Point3i` | Pt3i builds a Point3i from x, y and z. |
| `func (p Point3i) Add(o Point3i) Point3i` | Add returns the vector sum p+o. |
| `func (p Point3i) Cross(o Point3i) Point3i` | Cross returns the 3D cross product p×o. |
| `func (p Point3i) Dot(o Point3i) int` | Dot returns the integer dot product p·o. |
| `func (p Point3i) Equals(o Point3i) bool` | Equals reports whether p and o are the same point. |
| `func (p Point3i) Mul(s int) Point3i` | Mul returns the point scaled by s. |
| `func (p Point3i) Norm() float64` | Norm returns the Euclidean length of the position vector. |
| `func (p Point3i) String() string` | String renders the point as (x, y, z). |
| `func (p Point3i) Sub(o Point3i) Point3i` | Sub returns the vector difference p-o. |

</details>

<details>
<summary><code>Quatd</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Lerpd(a, b Quatd, t float64) Quatd` | Lerpd returns the straight-line interpolation (1-t)·a + t·b, mirroring cv::Quat::lerp. |
| `func NewQuatd(w, x, y, z float64) Quatd` | NewQuatd builds a Quatd from its four components. |
| `func Nlerpd(a, b Quatd, t float64) Quatd` | Nlerpd returns the normalized linear interpolation between a and b, mirroring cv::Quat::nlerp. |
| `func QuatdFromAxisAngle(axis Vec3d, angle float64) Quatd` | QuatdFromAxisAngle builds a unit quaternion rotating by angle radians about the given axis (which need not be normalized). |
| `func QuatdFromRotVec(rvec Vec3d) Quatd` | QuatdFromRotVec builds a unit quaternion from a Rodrigues rotation vector (unit axis scaled by the angle in radians), mirroring… |
| `func QuatdFromRotationMatrix(m Matx33d) Quatd` | QuatdFromRotationMatrix builds a unit quaternion from a 3x3 rotation matrix using the numerically stable branch of Shepperd's method. |
| `func QuatdIdentity() Quatd` | QuatdIdentity returns the identity rotation quaternion (1, 0, 0, 0). |
| `func Slerpd(a, b Quatd, t float64) Quatd` | Slerpd performs spherical linear interpolation between unit quaternions a and b by parameter t in [0, 1]. |
| `func (q Quatd) Add(o Quatd) Quatd` | Add returns the component-wise sum q+o. |
| `func (q Quatd) Angle() float64` | Angle returns the rotation angle in radians encoded by the quaternion, in [0, π], mirroring cv::Quat::getAngle. |
| `func (q Quatd) Axis() Vec3d` | Axis returns the unit rotation axis encoded by the quaternion, mirroring cv::Quat::getAxis. |
| `func (q Quatd) Conjugate() Quatd` | Conjugate returns the conjugate (w, -x, -y, -z). |
| `func (q Quatd) Dot(o Quatd) float64` | Dot returns the 4D dot product q·o. |
| `func (q Quatd) Equals(o Quatd) bool` | Equals reports whether q and o have identical components. |
| `func (q Quatd) Inverse() Quatd` | Inverse returns the multiplicative inverse conj(q)/\|q\|². |
| `func (q Quatd) IsNormalized(eps float64) bool` | IsNormalized reports whether \|q\| is within eps of 1, mirroring cv::Quat::isNormal. |
| `func (q Quatd) Mul(o Quatd) Quatd` | Mul returns the Hamilton product q*o. |
| `func (q Quatd) Neg() Quatd` | Neg returns -q, the component-wise negation. |
| `func (q Quatd) Norm() float64` | Norm returns the quaternion norm \|q\|. |
| `func (q Quatd) NormSq() float64` | NormSq returns the squared norm \|q\|². |
| `func (q Quatd) Normalize() Quatd` | Normalize returns the unit quaternion; a zero quaternion is returned as-is. |
| `func (q Quatd) RotateVector(v Vec3d) Vec3d` | RotateVector applies the rotation encoded by q to vector v. |
| `func (q Quatd) ScalarMul(s float64) Quatd` | ScalarMul returns the quaternion scaled by s. |
| `func (q Quatd) String() string` | String renders the quaternion. |
| `func (q Quatd) Sub(o Quatd) Quatd` | Sub returns the component-wise difference q-o. |
| `func (q Quatd) ToRotVec() Vec3d` | ToRotVec returns the Rodrigues rotation vector (the unit axis scaled by the angle in radians) equivalent to the quaternion, mirroring… |
| `func (q Quatd) ToRotationMatrix() Matx33d` | ToRotationMatrix returns the 3x3 rotation matrix of the unit quaternion. |

</details>

<details>
<summary><code>Quatf</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewQuatf(w, x, y, z float32) Quatf` | NewQuatf builds a Quatf from its four components. |
| `func QuatfFromAxisAngle(axis Vec3f, angle float64) Quatf` | QuatfFromAxisAngle builds a unit quaternion rotating by angle radians about the given axis (which need not be normalized). |
| `func QuatfFromQuatd(q Quatd) Quatf` | QuatfFromQuatd converts a double-precision quaternion to single precision. |
| `func QuatfIdentity() Quatf` | QuatfIdentity returns the identity rotation quaternion (1, 0, 0, 0). |
| `func Slerpf(a, b Quatf, t float64) Quatf` | Slerpf performs spherical linear interpolation between unit quaternions a and b by parameter t in [0, 1]. |
| `func (q Quatf) Add(o Quatf) Quatf` | Add returns the component-wise sum q+o. |
| `func (q Quatf) Conjugate() Quatf` | Conjugate returns the conjugate (w, -x, -y, -z). |
| `func (q Quatf) Dot(o Quatf) float64` | Dot returns the 4D dot product q·o, accumulated in float64. |
| `func (q Quatf) Equals(o Quatf) bool` | Equals reports whether q and o have identical components. |
| `func (q Quatf) Inverse() Quatf` | Inverse returns the multiplicative inverse conj(q)/\|q\|²; a zero quaternion is returned as-is. |
| `func (q Quatf) Mul(o Quatf) Quatf` | Mul returns the Hamilton product q*o. |
| `func (q Quatf) Norm() float64` | Norm returns the quaternion norm \|q\|. |
| `func (q Quatf) NormSq() float64` | NormSq returns the squared norm \|q\|², accumulated in float64. |
| `func (q Quatf) Normalize() Quatf` | Normalize returns the unit quaternion; a zero quaternion is returned as-is. |
| `func (q Quatf) RotateVector(v Vec3f) Vec3f` | RotateVector applies the rotation encoded by q to vector v. |
| `func (q Quatf) ScalarMul(s float32) Quatf` | ScalarMul returns the quaternion scaled by s. |
| `func (q Quatf) String() string` | String renders the quaternion. |
| `func (q Quatf) Sub(o Quatf) Quatf` | Sub returns the component-wise difference q-o. |
| `func (q Quatf) ToQuatd() Quatd` | ToQuatd converts to a double-precision quaternion. |
| `func (q Quatf) ToRotationMatrix() Matx33f` | ToRotationMatrix returns the 3x3 rotation matrix of the unit quaternion. |

</details>

<details>
<summary><code>RNG</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRNG(seed uint64) *RNG` | NewRNG creates an RNG seeded with the given value. |
| `func (r *RNG) FillGaussian(dst []float64, sigma float64)` | FillGaussian fills dst with independent normal samples of the given standard deviation and zero mean. |
| `func (r *RNG) FillUniformd(dst []float64, a, b float64)` | FillUniformd fills dst with independent uniform samples in [a, b). |
| `func (r *RNG) Gaussian(sigma float64) float64` | Gaussian returns a normally distributed value with mean 0 and the given standard deviation, using the Box–Muller transform. |
| `func (r *RNG) Next() uint32` | Next returns the next 32-bit pseudo-random value and advances the state. |
| `func (r *RNG) NextBool() bool` | NextBool returns a pseudo-random boolean. |
| `func (r *RNG) Uniformd(a, b float64) float64` | Uniformd returns a uniformly distributed float64 in the half-open range [a, b). |
| `func (r *RNG) Uniformf(a, b float32) float32` | Uniformf returns a uniformly distributed float32 in the half-open range [a, b). |
| `func (r *RNG) Uniformi(a, b int) int` | Uniformi returns a uniformly distributed integer in the half-open range [a, b). |

</details>

<details>
<summary><code>RNGMT19937</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRNGMT19937(seed uint32) *RNGMT19937` | NewRNGMT19937 creates a Mersenne Twister seeded with the given value. |
| `func (r *RNGMT19937) Next() uint32` | Next returns the next 32-bit pseudo-random value. |
| `func (r *RNGMT19937) Uniformd(a, b float64) float64` | Uniformd returns a uniformly distributed float64 in the half-open range [a, b). |
| `func (r *RNGMT19937) Uniformi(a, b int) int` | Uniformi returns a uniformly distributed integer in the half-open range [a, b). |

</details>

<details>
<summary><code>Range</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func AllRange() Range` | AllRange returns the special range covering an entire dimension, encoded as [MinInt, MaxInt) the way OpenCV's Range::all does. |
| `func NewRange(start, end int) Range` | NewRange builds a Range from start and end. |
| `func (r Range) Contains(i int) bool` | Contains reports whether i lies in [Start, End). |
| `func (r Range) Empty() bool` | Empty reports whether the range contains no elements. |
| `func (r Range) Equals(o Range) bool` | Equals reports whether two ranges are identical. |
| `func (r Range) Intersect(o Range) Range` | Intersect returns the overlap of r and o; an empty range when disjoint. |
| `func (r Range) Shift(delta int) Range` | Shift returns the range translated by delta. |
| `func (r Range) Size() int` | Size returns End-Start. |
| `func (r Range) String() string` | String renders the range as [Start, End). |

</details>

<details>
<summary><code>Rect2d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BoundingRect2d(pts []Point2d) Rect2d` | BoundingRect2d returns the smallest upright rectangle containing every point. |
| `func Rc2d(x, y, w, h float64) Rect2d` | Rc2d builds a Rect2d from its corner and size. |
| `func (r Rect2d) And(o Rect2d) Rect2d` | And returns the intersection of r and o; an empty rectangle when disjoint. |
| `func (r Rect2d) Area() float64` | Area returns Width*Height. |
| `func (r Rect2d) BR() Point2d` | BR returns the bottom-right corner. |
| `func (r Rect2d) Contains(p Point2d) bool` | Contains reports whether point p lies inside the rectangle. |
| `func (r Rect2d) Empty() bool` | Empty reports whether the rectangle has non-positive area. |
| `func (r Rect2d) Equals(o Rect2d) bool` | Equals reports whether two rectangles are identical. |
| `func (r Rect2d) Or(o Rect2d) Rect2d` | Or returns the smallest rectangle containing both r and o. |
| `func (r Rect2d) Shift(p Point2d) Rect2d` | Shift returns the rectangle translated by point p. |
| `func (r Rect2d) Size() Size2d` | Size returns the rectangle's size. |
| `func (r Rect2d) String() string` | String renders the rectangle as [WxH from (X, Y)]. |
| `func (r Rect2d) TL() Point2d` | TL returns the top-left corner. |

</details>

<details>
<summary><code>Rect2f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BoundingRect2f(pts []Point2f) Rect2f` | BoundingRect2f returns the smallest upright rectangle containing every point. |
| `func Rc2f(x, y, w, h float32) Rect2f` | Rc2f builds a Rect2f from its corner and size. |
| `func (r Rect2f) And(o Rect2f) Rect2f` | And returns the intersection of r and o; an empty rectangle when disjoint. |
| `func (r Rect2f) Area() float32` | Area returns Width*Height. |
| `func (r Rect2f) BR() Point2f` | BR returns the bottom-right corner. |
| `func (r Rect2f) Contains(p Point2f) bool` | Contains reports whether point p lies inside the rectangle. |
| `func (r Rect2f) Empty() bool` | Empty reports whether the rectangle has non-positive area. |
| `func (r Rect2f) Equals(o Rect2f) bool` | Equals reports whether two rectangles are identical. |
| `func (r Rect2f) Or(o Rect2f) Rect2f` | Or returns the smallest rectangle containing both r and o. |
| `func (r Rect2f) Shift(p Point2f) Rect2f` | Shift returns the rectangle translated by point p. |
| `func (r Rect2f) Size() Size2f` | Size returns the rectangle's size. |
| `func (r Rect2f) String() string` | String renders the rectangle as [WxH from (X, Y)]. |
| `func (r Rect2f) TL() Point2f` | TL returns the top-left corner. |

</details>

<details>
<summary><code>Rect2i</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BoundingRect2i(pts []Point2i) Rect2i` | BoundingRect2i returns the smallest upright rectangle containing every point. |
| `func Rc2i(x, y, w, h int) Rect2i` | Rc2i builds a Rect2i from its corner and size. |
| `func (r Rect2i) And(o Rect2i) Rect2i` | And returns the intersection of r and o; an empty rectangle when they do not overlap. |
| `func (r Rect2i) Area() int` | Area returns Width*Height. |
| `func (r Rect2i) BR() Point2i` | BR returns the bottom-right corner (exclusive: X+Width, Y+Height). |
| `func (r Rect2i) Contains(p Point2i) bool` | Contains reports whether point p lies inside the rectangle. |
| `func (r Rect2i) Empty() bool` | Empty reports whether the rectangle has non-positive area. |
| `func (r Rect2i) Equals(o Rect2i) bool` | Equals reports whether two rectangles are identical. |
| `func (r Rect2i) Or(o Rect2i) Rect2i` | Or returns the smallest rectangle containing both r and o. |
| `func (r Rect2i) Shift(p Point2i) Rect2i` | Shift returns the rectangle translated by point p. |
| `func (r Rect2i) Size() Size2i` | Size returns the rectangle's size. |
| `func (r Rect2i) String() string` | String renders the rectangle as [WxH from (X, Y)]. |
| `func (r Rect2i) TL() Point2i` | TL returns the top-left corner. |

</details>

<details>
<summary><code>RotatedRect</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewRotatedRect(center Point2f, size Size2f, angle float64) RotatedRect` | NewRotatedRect builds a RotatedRect from its centre, size and angle. |
| `func (r RotatedRect) Area() float64` | Area returns the rectangle's area (Width*Height). |
| `func (r RotatedRect) BoundingRect() Rect2i` | BoundingRect returns the smallest upright integer rectangle enclosing the rotated rectangle. |
| `func (r RotatedRect) Points() [4]Point2f` | Points returns the four corners of the rotated rectangle in the same order cv::RotatedRect::points uses: bottom-left, top-left, top-right,… |

</details>

<details>
<summary><code>Scalar</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewScalar(v ...float64) Scalar` | NewScalar builds a Scalar from up to four components; missing entries are zero and extra entries are ignored. |
| `func ScalarAll(v float64) Scalar` | ScalarAll returns a Scalar with every component set to v. |
| `func (s Scalar) Add(o Scalar) Scalar` | Add returns the element-wise sum s+o. |
| `func (s Scalar) Conj() Scalar` | Conj returns the quaternion-style conjugate (v0, -v1, -v2, -v3). |
| `func (s Scalar) Equals(o Scalar) bool` | Equals reports whether two scalars are identical. |
| `func (s Scalar) IsReal() bool` | IsReal reports whether the last three components are all zero. |
| `func (s Scalar) Mul(k float64) Scalar` | Mul returns the scalar scaled by k. |
| `func (s Scalar) MulElem(o Scalar) Scalar` | MulElem returns the element-wise product s*o. |
| `func (s Scalar) String() string` | String renders the scalar as [a, b, c, d]. |
| `func (s Scalar) Sub(o Scalar) Scalar` | Sub returns the element-wise difference s-o. |

</details>

<details>
<summary><code>Size2d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Sz2d(w, h float64) Size2d` | Sz2d builds a Size2d from width and height. |
| `func (s Size2d) Add(o Size2d) Size2d` | Add returns the element-wise sum of two sizes. |
| `func (s Size2d) Area() float64` | Area returns Width*Height. |
| `func (s Size2d) Aspect() float64` | Aspect returns Width/Height as a float64 (0 when Height is 0). |
| `func (s Size2d) Empty() bool` | Empty reports whether either dimension is non-positive. |
| `func (s Size2d) Equals(o Size2d) bool` | Equals reports whether two sizes are identical. |
| `func (s Size2d) Scale(k float64) Size2d` | Scale returns the size with both dimensions multiplied by k. |
| `func (s Size2d) String() string` | String renders the size as WxH. |
| `func (s Size2d) Sub(o Size2d) Size2d` | Sub returns the element-wise difference of two sizes. |

</details>

<details>
<summary><code>Size2f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Sz2f(w, h float32) Size2f` | Sz2f builds a Size2f from width and height. |
| `func (s Size2f) Add(o Size2f) Size2f` | Add returns the element-wise sum of two sizes. |
| `func (s Size2f) Area() float32` | Area returns Width*Height. |
| `func (s Size2f) Aspect() float64` | Aspect returns Width/Height as a float64 (0 when Height is 0). |
| `func (s Size2f) Empty() bool` | Empty reports whether either dimension is non-positive. |
| `func (s Size2f) Equals(o Size2f) bool` | Equals reports whether two sizes are identical. |
| `func (s Size2f) Scale(k float32) Size2f` | Scale returns the size with both dimensions multiplied by k. |
| `func (s Size2f) String() string` | String renders the size as WxH. |
| `func (s Size2f) Sub(o Size2f) Size2f` | Sub returns the element-wise difference of two sizes. |

</details>

<details>
<summary><code>Size2i</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func Sz2i(w, h int) Size2i` | Sz2i builds a Size2i from width and height. |
| `func (s Size2i) Add(o Size2i) Size2i` | Add returns the element-wise sum of two sizes. |
| `func (s Size2i) Area() int` | Area returns Width*Height. |
| `func (s Size2i) Aspect() float64` | Aspect returns Width/Height as a float64 (0 when Height is 0). |
| `func (s Size2i) Empty() bool` | Empty reports whether either dimension is non-positive. |
| `func (s Size2i) Equals(o Size2i) bool` | Equals reports whether two sizes are identical. |
| `func (s Size2i) Scale(k float64) Size2i` | Scale returns the size with both dimensions multiplied by k (truncated). |
| `func (s Size2i) String() string` | String renders the size as WxH. |
| `func (s Size2i) Sub(o Size2i) Size2i` | Sub returns the element-wise difference of two sizes. |

</details>

<details>
<summary><code>TermCriteria</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewTermCriteria(t TermCriteriaType, maxCount int, eps float64) TermCriteria` | NewTermCriteria builds a TermCriteria from its type flags, iteration cap and epsilon. |
| `func (t TermCriteria) IsValid() bool` | IsValid reports whether at least one stopping condition is enabled with a usable bound. |

</details>

<details>
<summary><code>Vec2b</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec2b(x uint8, y uint8) Vec2b` | NewVec2b builds a Vec2b from its 2 components. |
| `func (v Vec2b) Add(o Vec2b) Vec2b` | Add returns the element-wise sum v+o (no saturation; uint8 arithmetic wraps). |
| `func (v Vec2b) At(i int) uint8` | At returns element i of the vector. |
| `func (v Vec2b) Dot(o Vec2b) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec2b) Equals(o Vec2b) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec2b) Mul(s uint8) Vec2b` | Mul returns the vector scaled by s. |
| `func (v Vec2b) MulElem(o Vec2b) Vec2b` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec2b) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec2b) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec2b) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec2b) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec2b) Sub(o Vec2b) Vec2b` | Sub returns the element-wise difference v-o. |
| `func (v Vec2b) ToSlice() []uint8` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec2b) ToVec2d() Vec2d` | ToVec2d converts the components to float64. |
| `func (v Vec2b) ToVec2f() Vec2f` | ToVec2f converts the components to float32. |

</details>

<details>
<summary><code>Vec2d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec2d(x float64, y float64) Vec2d` | NewVec2d builds a Vec2d from its 2 components. |
| `func (v Vec2d) Add(o Vec2d) Vec2d` | Add returns the element-wise sum v+o (no saturation; float64 arithmetic wraps). |
| `func (v Vec2d) At(i int) float64` | At returns element i of the vector. |
| `func (v Vec2d) Dot(o Vec2d) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec2d) Equals(o Vec2d) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec2d) Mul(s float64) Vec2d` | Mul returns the vector scaled by s. |
| `func (v Vec2d) MulElem(o Vec2d) Vec2d` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec2d) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec2d) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec2d) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec2d) Normalize() Vec2d` | Normalize returns the unit vector in the direction of v. |
| `func (v Vec2d) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec2d) Sub(o Vec2d) Vec2d` | Sub returns the element-wise difference v-o. |
| `func (v Vec2d) ToSlice() []float64` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec2d) ToVec2f() Vec2f` | ToVec2f converts the components to float32. |

</details>

<details>
<summary><code>Vec2f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec2f(x float32, y float32) Vec2f` | NewVec2f builds a Vec2f from its 2 components. |
| `func (v Vec2f) Add(o Vec2f) Vec2f` | Add returns the element-wise sum v+o (no saturation; float32 arithmetic wraps). |
| `func (v Vec2f) At(i int) float32` | At returns element i of the vector. |
| `func (v Vec2f) Dot(o Vec2f) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec2f) Equals(o Vec2f) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec2f) Mul(s float32) Vec2f` | Mul returns the vector scaled by s. |
| `func (v Vec2f) MulElem(o Vec2f) Vec2f` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec2f) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec2f) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec2f) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec2f) Normalize() Vec2f` | Normalize returns the unit vector in the direction of v. |
| `func (v Vec2f) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec2f) Sub(o Vec2f) Vec2f` | Sub returns the element-wise difference v-o. |
| `func (v Vec2f) ToSlice() []float32` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec2f) ToVec2d() Vec2d` | ToVec2d converts the components to float64. |

</details>

<details>
<summary><code>Vec2i</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec2i(x int, y int) Vec2i` | NewVec2i builds a Vec2i from its 2 components. |
| `func (v Vec2i) Add(o Vec2i) Vec2i` | Add returns the element-wise sum v+o (no saturation; int arithmetic wraps). |
| `func (v Vec2i) At(i int) int` | At returns element i of the vector. |
| `func (v Vec2i) Dot(o Vec2i) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec2i) Equals(o Vec2i) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec2i) Mul(s int) Vec2i` | Mul returns the vector scaled by s. |
| `func (v Vec2i) MulElem(o Vec2i) Vec2i` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec2i) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec2i) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec2i) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec2i) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec2i) Sub(o Vec2i) Vec2i` | Sub returns the element-wise difference v-o. |
| `func (v Vec2i) ToSlice() []int` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec2i) ToVec2d() Vec2d` | ToVec2d converts the components to float64. |
| `func (v Vec2i) ToVec2f() Vec2f` | ToVec2f converts the components to float32. |

</details>

<details>
<summary><code>Vec2s</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec2s(x int16, y int16) Vec2s` | NewVec2s builds a Vec2s from its 2 components. |
| `func (v Vec2s) Add(o Vec2s) Vec2s` | Add returns the element-wise sum v+o (no saturation; int16 arithmetic wraps). |
| `func (v Vec2s) At(i int) int16` | At returns element i of the vector. |
| `func (v Vec2s) Dot(o Vec2s) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec2s) Equals(o Vec2s) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec2s) Mul(s int16) Vec2s` | Mul returns the vector scaled by s. |
| `func (v Vec2s) MulElem(o Vec2s) Vec2s` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec2s) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec2s) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec2s) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec2s) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec2s) Sub(o Vec2s) Vec2s` | Sub returns the element-wise difference v-o. |
| `func (v Vec2s) ToSlice() []int16` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec2s) ToVec2d() Vec2d` | ToVec2d converts the components to float64. |
| `func (v Vec2s) ToVec2f() Vec2f` | ToVec2f converts the components to float32. |

</details>

<details>
<summary><code>Vec2w</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec2w(x uint16, y uint16) Vec2w` | NewVec2w builds a Vec2w from its 2 components. |
| `func (v Vec2w) Add(o Vec2w) Vec2w` | Add returns the element-wise sum v+o (no saturation; uint16 arithmetic wraps). |
| `func (v Vec2w) At(i int) uint16` | At returns element i of the vector. |
| `func (v Vec2w) Dot(o Vec2w) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec2w) Equals(o Vec2w) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec2w) Mul(s uint16) Vec2w` | Mul returns the vector scaled by s. |
| `func (v Vec2w) MulElem(o Vec2w) Vec2w` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec2w) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec2w) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec2w) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec2w) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec2w) Sub(o Vec2w) Vec2w` | Sub returns the element-wise difference v-o. |
| `func (v Vec2w) ToSlice() []uint16` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec2w) ToVec2d() Vec2d` | ToVec2d converts the components to float64. |
| `func (v Vec2w) ToVec2f() Vec2f` | ToVec2f converts the components to float32. |

</details>

<details>
<summary><code>Vec3b</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec3b(x uint8, y uint8, z uint8) Vec3b` | NewVec3b builds a Vec3b from its 3 components. |
| `func (v Vec3b) Add(o Vec3b) Vec3b` | Add returns the element-wise sum v+o (no saturation; uint8 arithmetic wraps). |
| `func (v Vec3b) At(i int) uint8` | At returns element i of the vector. |
| `func (v Vec3b) Cross(o Vec3b) Vec3b` | Cross returns the 3D cross product v x o. |
| `func (v Vec3b) Dot(o Vec3b) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec3b) Equals(o Vec3b) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec3b) Mul(s uint8) Vec3b` | Mul returns the vector scaled by s. |
| `func (v Vec3b) MulElem(o Vec3b) Vec3b` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec3b) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec3b) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec3b) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec3b) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec3b) Sub(o Vec3b) Vec3b` | Sub returns the element-wise difference v-o. |
| `func (v Vec3b) ToSlice() []uint8` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec3b) ToVec3d() Vec3d` | ToVec3d converts the components to float64. |
| `func (v Vec3b) ToVec3f() Vec3f` | ToVec3f converts the components to float32. |

</details>

<details>
<summary><code>Vec3d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MatrixToRodriguesd(r Matx33d) Vec3d` | MatrixToRodriguesd converts a 3x3 rotation matrix to a rotation vector (axis scaled by angle in radians), the inverse of RodriguesToMatrixd. |
| `func NewVec3d(x float64, y float64, z float64) Vec3d` | NewVec3d builds a Vec3d from its 3 components. |
| `func (v Vec3d) Add(o Vec3d) Vec3d` | Add returns the element-wise sum v+o (no saturation; float64 arithmetic wraps). |
| `func (v Vec3d) At(i int) float64` | At returns element i of the vector. |
| `func (v Vec3d) Cross(o Vec3d) Vec3d` | Cross returns the 3D cross product v x o. |
| `func (v Vec3d) Dot(o Vec3d) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec3d) Equals(o Vec3d) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec3d) Mul(s float64) Vec3d` | Mul returns the vector scaled by s. |
| `func (v Vec3d) MulElem(o Vec3d) Vec3d` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec3d) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec3d) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec3d) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec3d) Normalize() Vec3d` | Normalize returns the unit vector in the direction of v. |
| `func (v Vec3d) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec3d) Sub(o Vec3d) Vec3d` | Sub returns the element-wise difference v-o. |
| `func (v Vec3d) ToSlice() []float64` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec3d) ToVec3f() Vec3f` | ToVec3f converts the components to float32. |

</details>

<details>
<summary><code>Vec3f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec3f(x float32, y float32, z float32) Vec3f` | NewVec3f builds a Vec3f from its 3 components. |
| `func (v Vec3f) Add(o Vec3f) Vec3f` | Add returns the element-wise sum v+o (no saturation; float32 arithmetic wraps). |
| `func (v Vec3f) At(i int) float32` | At returns element i of the vector. |
| `func (v Vec3f) Cross(o Vec3f) Vec3f` | Cross returns the 3D cross product v x o. |
| `func (v Vec3f) Dot(o Vec3f) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec3f) Equals(o Vec3f) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec3f) Mul(s float32) Vec3f` | Mul returns the vector scaled by s. |
| `func (v Vec3f) MulElem(o Vec3f) Vec3f` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec3f) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec3f) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec3f) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec3f) Normalize() Vec3f` | Normalize returns the unit vector in the direction of v. |
| `func (v Vec3f) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec3f) Sub(o Vec3f) Vec3f` | Sub returns the element-wise difference v-o. |
| `func (v Vec3f) ToSlice() []float32` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec3f) ToVec3d() Vec3d` | ToVec3d converts the components to float64. |

</details>

<details>
<summary><code>Vec3i</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec3i(x int, y int, z int) Vec3i` | NewVec3i builds a Vec3i from its 3 components. |
| `func (v Vec3i) Add(o Vec3i) Vec3i` | Add returns the element-wise sum v+o (no saturation; int arithmetic wraps). |
| `func (v Vec3i) At(i int) int` | At returns element i of the vector. |
| `func (v Vec3i) Cross(o Vec3i) Vec3i` | Cross returns the 3D cross product v x o. |
| `func (v Vec3i) Dot(o Vec3i) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec3i) Equals(o Vec3i) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec3i) Mul(s int) Vec3i` | Mul returns the vector scaled by s. |
| `func (v Vec3i) MulElem(o Vec3i) Vec3i` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec3i) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec3i) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec3i) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec3i) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec3i) Sub(o Vec3i) Vec3i` | Sub returns the element-wise difference v-o. |
| `func (v Vec3i) ToSlice() []int` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec3i) ToVec3d() Vec3d` | ToVec3d converts the components to float64. |
| `func (v Vec3i) ToVec3f() Vec3f` | ToVec3f converts the components to float32. |

</details>

<details>
<summary><code>Vec3s</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec3s(x int16, y int16, z int16) Vec3s` | NewVec3s builds a Vec3s from its 3 components. |
| `func (v Vec3s) Add(o Vec3s) Vec3s` | Add returns the element-wise sum v+o (no saturation; int16 arithmetic wraps). |
| `func (v Vec3s) At(i int) int16` | At returns element i of the vector. |
| `func (v Vec3s) Cross(o Vec3s) Vec3s` | Cross returns the 3D cross product v x o. |
| `func (v Vec3s) Dot(o Vec3s) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec3s) Equals(o Vec3s) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec3s) Mul(s int16) Vec3s` | Mul returns the vector scaled by s. |
| `func (v Vec3s) MulElem(o Vec3s) Vec3s` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec3s) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec3s) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec3s) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec3s) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec3s) Sub(o Vec3s) Vec3s` | Sub returns the element-wise difference v-o. |
| `func (v Vec3s) ToSlice() []int16` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec3s) ToVec3d() Vec3d` | ToVec3d converts the components to float64. |
| `func (v Vec3s) ToVec3f() Vec3f` | ToVec3f converts the components to float32. |

</details>

<details>
<summary><code>Vec3w</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec3w(x uint16, y uint16, z uint16) Vec3w` | NewVec3w builds a Vec3w from its 3 components. |
| `func (v Vec3w) Add(o Vec3w) Vec3w` | Add returns the element-wise sum v+o (no saturation; uint16 arithmetic wraps). |
| `func (v Vec3w) At(i int) uint16` | At returns element i of the vector. |
| `func (v Vec3w) Cross(o Vec3w) Vec3w` | Cross returns the 3D cross product v x o. |
| `func (v Vec3w) Dot(o Vec3w) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec3w) Equals(o Vec3w) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec3w) Mul(s uint16) Vec3w` | Mul returns the vector scaled by s. |
| `func (v Vec3w) MulElem(o Vec3w) Vec3w` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec3w) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec3w) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec3w) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec3w) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec3w) Sub(o Vec3w) Vec3w` | Sub returns the element-wise difference v-o. |
| `func (v Vec3w) ToSlice() []uint16` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec3w) ToVec3d() Vec3d` | ToVec3d converts the components to float64. |
| `func (v Vec3w) ToVec3f() Vec3f` | ToVec3f converts the components to float32. |

</details>

<details>
<summary><code>Vec4b</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec4b(x uint8, y uint8, z uint8, w uint8) Vec4b` | NewVec4b builds a Vec4b from its 4 components. |
| `func (v Vec4b) Add(o Vec4b) Vec4b` | Add returns the element-wise sum v+o (no saturation; uint8 arithmetic wraps). |
| `func (v Vec4b) At(i int) uint8` | At returns element i of the vector. |
| `func (v Vec4b) Dot(o Vec4b) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec4b) Equals(o Vec4b) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec4b) Mul(s uint8) Vec4b` | Mul returns the vector scaled by s. |
| `func (v Vec4b) MulElem(o Vec4b) Vec4b` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec4b) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec4b) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec4b) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec4b) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec4b) Sub(o Vec4b) Vec4b` | Sub returns the element-wise difference v-o. |
| `func (v Vec4b) ToSlice() []uint8` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec4b) ToVec4d() Vec4d` | ToVec4d converts the components to float64. |
| `func (v Vec4b) ToVec4f() Vec4f` | ToVec4f converts the components to float32. |

</details>

<details>
<summary><code>Vec4d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec4d(x float64, y float64, z float64, w float64) Vec4d` | NewVec4d builds a Vec4d from its 4 components. |
| `func (v Vec4d) Add(o Vec4d) Vec4d` | Add returns the element-wise sum v+o (no saturation; float64 arithmetic wraps). |
| `func (v Vec4d) At(i int) float64` | At returns element i of the vector. |
| `func (v Vec4d) Dot(o Vec4d) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec4d) Equals(o Vec4d) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec4d) Mul(s float64) Vec4d` | Mul returns the vector scaled by s. |
| `func (v Vec4d) MulElem(o Vec4d) Vec4d` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec4d) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec4d) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec4d) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec4d) Normalize() Vec4d` | Normalize returns the unit vector in the direction of v. |
| `func (v Vec4d) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec4d) Sub(o Vec4d) Vec4d` | Sub returns the element-wise difference v-o. |
| `func (v Vec4d) ToSlice() []float64` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec4d) ToVec4f() Vec4f` | ToVec4f converts the components to float32. |

</details>

<details>
<summary><code>Vec4f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec4f(x float32, y float32, z float32, w float32) Vec4f` | NewVec4f builds a Vec4f from its 4 components. |
| `func (v Vec4f) Add(o Vec4f) Vec4f` | Add returns the element-wise sum v+o (no saturation; float32 arithmetic wraps). |
| `func (v Vec4f) At(i int) float32` | At returns element i of the vector. |
| `func (v Vec4f) Dot(o Vec4f) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec4f) Equals(o Vec4f) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec4f) Mul(s float32) Vec4f` | Mul returns the vector scaled by s. |
| `func (v Vec4f) MulElem(o Vec4f) Vec4f` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec4f) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec4f) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec4f) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec4f) Normalize() Vec4f` | Normalize returns the unit vector in the direction of v. |
| `func (v Vec4f) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec4f) Sub(o Vec4f) Vec4f` | Sub returns the element-wise difference v-o. |
| `func (v Vec4f) ToSlice() []float32` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec4f) ToVec4d() Vec4d` | ToVec4d converts the components to float64. |

</details>

<details>
<summary><code>Vec4i</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec4i(x int, y int, z int, w int) Vec4i` | NewVec4i builds a Vec4i from its 4 components. |
| `func (v Vec4i) Add(o Vec4i) Vec4i` | Add returns the element-wise sum v+o (no saturation; int arithmetic wraps). |
| `func (v Vec4i) At(i int) int` | At returns element i of the vector. |
| `func (v Vec4i) Dot(o Vec4i) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec4i) Equals(o Vec4i) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec4i) Mul(s int) Vec4i` | Mul returns the vector scaled by s. |
| `func (v Vec4i) MulElem(o Vec4i) Vec4i` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec4i) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec4i) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec4i) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec4i) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec4i) Sub(o Vec4i) Vec4i` | Sub returns the element-wise difference v-o. |
| `func (v Vec4i) ToSlice() []int` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec4i) ToVec4d() Vec4d` | ToVec4d converts the components to float64. |
| `func (v Vec4i) ToVec4f() Vec4f` | ToVec4f converts the components to float32. |

</details>

<details>
<summary><code>Vec4s</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec4s(x int16, y int16, z int16, w int16) Vec4s` | NewVec4s builds a Vec4s from its 4 components. |
| `func (v Vec4s) Add(o Vec4s) Vec4s` | Add returns the element-wise sum v+o (no saturation; int16 arithmetic wraps). |
| `func (v Vec4s) At(i int) int16` | At returns element i of the vector. |
| `func (v Vec4s) Dot(o Vec4s) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec4s) Equals(o Vec4s) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec4s) Mul(s int16) Vec4s` | Mul returns the vector scaled by s. |
| `func (v Vec4s) MulElem(o Vec4s) Vec4s` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec4s) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec4s) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec4s) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec4s) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec4s) Sub(o Vec4s) Vec4s` | Sub returns the element-wise difference v-o. |
| `func (v Vec4s) ToSlice() []int16` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec4s) ToVec4d() Vec4d` | ToVec4d converts the components to float64. |
| `func (v Vec4s) ToVec4f() Vec4f` | ToVec4f converts the components to float32. |

</details>

<details>
<summary><code>Vec4w</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec4w(x uint16, y uint16, z uint16, w uint16) Vec4w` | NewVec4w builds a Vec4w from its 4 components. |
| `func (v Vec4w) Add(o Vec4w) Vec4w` | Add returns the element-wise sum v+o (no saturation; uint16 arithmetic wraps). |
| `func (v Vec4w) At(i int) uint16` | At returns element i of the vector. |
| `func (v Vec4w) Dot(o Vec4w) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec4w) Equals(o Vec4w) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec4w) Mul(s uint16) Vec4w` | Mul returns the vector scaled by s. |
| `func (v Vec4w) MulElem(o Vec4w) Vec4w` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec4w) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec4w) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec4w) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec4w) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec4w) Sub(o Vec4w) Vec4w` | Sub returns the element-wise difference v-o. |
| `func (v Vec4w) ToSlice() []uint16` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec4w) ToVec4d() Vec4d` | ToVec4d converts the components to float64. |
| `func (v Vec4w) ToVec4f() Vec4f` | ToVec4f converts the components to float32. |

</details>

<details>
<summary><code>Vec6d</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec6d(x float64, y float64, z float64, w float64, a float64, b float64) Vec6d` | NewVec6d builds a Vec6d from its 6 components. |
| `func (v Vec6d) Add(o Vec6d) Vec6d` | Add returns the element-wise sum v+o (no saturation; float64 arithmetic wraps). |
| `func (v Vec6d) At(i int) float64` | At returns element i of the vector. |
| `func (v Vec6d) Dot(o Vec6d) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec6d) Equals(o Vec6d) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec6d) Mul(s float64) Vec6d` | Mul returns the vector scaled by s. |
| `func (v Vec6d) MulElem(o Vec6d) Vec6d` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec6d) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec6d) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec6d) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec6d) Normalize() Vec6d` | Normalize returns the unit vector in the direction of v. |
| `func (v Vec6d) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec6d) Sub(o Vec6d) Vec6d` | Sub returns the element-wise difference v-o. |
| `func (v Vec6d) ToSlice() []float64` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec6d) ToVec6f() Vec6f` | ToVec6f converts the components to float32. |

</details>

<details>
<summary><code>Vec6f</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec6f(x float32, y float32, z float32, w float32, a float32, b float32) Vec6f` | NewVec6f builds a Vec6f from its 6 components. |
| `func (v Vec6f) Add(o Vec6f) Vec6f` | Add returns the element-wise sum v+o (no saturation; float32 arithmetic wraps). |
| `func (v Vec6f) At(i int) float32` | At returns element i of the vector. |
| `func (v Vec6f) Dot(o Vec6f) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec6f) Equals(o Vec6f) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec6f) Mul(s float32) Vec6f` | Mul returns the vector scaled by s. |
| `func (v Vec6f) MulElem(o Vec6f) Vec6f` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec6f) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec6f) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec6f) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec6f) Normalize() Vec6f` | Normalize returns the unit vector in the direction of v. |
| `func (v Vec6f) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec6f) Sub(o Vec6f) Vec6f` | Sub returns the element-wise difference v-o. |
| `func (v Vec6f) ToSlice() []float32` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec6f) ToVec6d() Vec6d` | ToVec6d converts the components to float64. |

</details>

<details>
<summary><code>Vec6i</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec6i(x int, y int, z int, w int, a int, b int) Vec6i` | NewVec6i builds a Vec6i from its 6 components. |
| `func (v Vec6i) Add(o Vec6i) Vec6i` | Add returns the element-wise sum v+o (no saturation; int arithmetic wraps). |
| `func (v Vec6i) At(i int) int` | At returns element i of the vector. |
| `func (v Vec6i) Dot(o Vec6i) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec6i) Equals(o Vec6i) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec6i) Mul(s int) Vec6i` | Mul returns the vector scaled by s. |
| `func (v Vec6i) MulElem(o Vec6i) Vec6i` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec6i) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec6i) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec6i) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec6i) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec6i) Sub(o Vec6i) Vec6i` | Sub returns the element-wise difference v-o. |
| `func (v Vec6i) ToSlice() []int` | ToSlice returns the components as a freshly allocated slice. |
| `func (v Vec6i) ToVec6d() Vec6d` | ToVec6d converts the components to float64. |
| `func (v Vec6i) ToVec6f() Vec6f` | ToVec6f converts the components to float32. |

</details>

<details>
<summary><code>Vec8i</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewVec8i(x int, y int, z int, w int, a int, b int, g int, h int) Vec8i` | NewVec8i builds a Vec8i from its 8 components. |
| `func (v Vec8i) Add(o Vec8i) Vec8i` | Add returns the element-wise sum v+o (no saturation; int arithmetic wraps). |
| `func (v Vec8i) At(i int) int` | At returns element i of the vector. |
| `func (v Vec8i) Dot(o Vec8i) float64` | Dot returns the dot product of v and o accumulated in float64. |
| `func (v Vec8i) Equals(o Vec8i) bool` | Equals reports whether v and o have identical components. |
| `func (v Vec8i) Mul(s int) Vec8i` | Mul returns the vector scaled by s. |
| `func (v Vec8i) MulElem(o Vec8i) Vec8i` | MulElem returns the element-wise (Hadamard) product v*o. |
| `func (v Vec8i) Norm() float64` | Norm returns the Euclidean (L2) length of the vector. |
| `func (v Vec8i) NormL1() float64` | NormL1 returns the L1 (Manhattan) norm as a float64. |
| `func (v Vec8i) NormSq() float64` | NormSq returns the squared L2 norm (v dot v) in float64. |
| `func (v Vec8i) String() string` | String renders the vector as [a, b, ...]. |
| `func (v Vec8i) Sub(o Vec8i) Vec8i` | Sub returns the element-wise difference v-o. |
| `func (v Vec8i) ToSlice() []int` | ToSlice returns the components as a freshly allocated slice. |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/core).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
