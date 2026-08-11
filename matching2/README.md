# matching2 — implements descriptor matching and multiple-view geometry on top of the parent cv

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/matching2.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/matching2)

Package matching2 implements descriptor matching and multiple-view geometry on
top of the parent cv module, using only the Go standard library.

- Descriptor matching. Brute-force matching with the L1, L2 and Hamming norms
(`BFMatcher`, `HammingMatcher`), k-nearest-neighbour and radius matching,
Lowe's ratio test (`RatioTest`), symmetric cross-check (`CrossCheck`) and an
approximate FLANN-style kd-tree matcher (`KDTree`, `FLANNMatcher`).

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/matching2"
```

## Usage

This is the package's own `ExampleBFMatcher_KnnMatch`, so it compiles and its output is
asserted on every `go test ./matching2/`.

```go
train := [][]float64{{0, 0}, {5, 0}, {10, 0}}
	query := [][]float64{{0.5, 0}}
	knn := matching2.NewBFMatcher(matching2.NormL2).KnnMatch(query, train, 2)
	fmt.Println(len(knn[0]), knn[0][0].TrainIdx, knn[0][1].TrainIdx)

	fmt.Println(len(matching2.RatioTest(knn, 0.5)))
```

```
2 0 1
1
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func ApplyHomography(H [3][3]float64, p core.Point2d) core.Point2d` | ApplyHomography maps a single point through the homography H, returning the dehomogenized image (H·(p,1) divided by its third coordinate). |
| `func DecomposeEssentialMat(E [3][3]float64) (R1, R2 [3][3]float64, t [3]float64)` | DecomposeEssentialMat factors an essential matrix into the two possible rotations (R1, R2) and the translation direction t (unit length, sign… |
| `func EpipolarConstraint(F [3][3]float64, p1, p2 core.Point2d) float64` | EpipolarConstraint returns the algebraic residual x2ᵀ·F·x1 for a correspondence, which is exactly zero for a noise-free inlier. |
| `func EpipolarLine(F [3][3]float64, p core.Point2d, whichImage int) [3]float64` | EpipolarLine returns the epipolar line induced in the other image by a point. |
| `func Epipoles(F [3][3]float64) (e1, e2 core.Point2d, ok bool)` | Epipoles returns the epipoles e1 (in image 1) and e2 (in image 2) of the fundamental matrix F. e1 is the right null vector of F and e2 is the left… |
| `func EssentialFromFundamental(F, K [3][3]float64) [3][3]float64` | EssentialFromFundamental converts a fundamental matrix F and shared intrinsic matrix K into the corresponding essential matrix E = Kᵀ·F·K, with… |
| `func FindEssentialMat(pts1, pts2 []core.Point2d, K [3][3]float64) ([3][3]float64, bool)` | FindEssentialMat estimates the 3×3 essential matrix relating two calibrated views from at least eight correspondences given in pixel coordinates,… |
| `func FindFundamentalMat(pts1, pts2 []core.Point2d) ([3][3]float64, bool)` | FindFundamentalMat estimates the 3×3 fundamental matrix F relating two views from at least eight point correspondences, using Hartley's normalized… |
| `func FindHomographyDLT(src, dst []core.Point2d) ([3][3]float64, bool)` | FindHomographyDLT estimates the 3×3 projective transform H mapping the source points onto the destination points using the normalized Direct Linear… |
| `func FloatDistance(norm NormType, a, b []float64) float64` | FloatDistance computes the distance between two float descriptors under the given norm. |
| `func FundamentalFromEssential(E, K [3][3]float64) ([3][3]float64, bool)` | FundamentalFromEssential converts an essential matrix E and shared intrinsic matrix K into the corresponding fundamental matrix F =… |
| `func HammingDistance(a, b []byte) int` | HammingDistance returns the number of differing bits between two equal-length bit-packed binary descriptors. |
| `func HomographyReprojectionError(H [3][3]float64, src, dst core.Point2d) float64` | HomographyReprojectionError returns the forward reprojection error ‖H·src − dst‖ for a single correspondence, in destination pixels. |
| `func L1Distance(a, b []float64) float64` | L1Distance returns the Manhattan distance between two equal-length float descriptors. |
| `func L2Distance(a, b []float64) float64` | L2Distance returns the Euclidean distance between two equal-length float descriptors. |
| `func L2SquaredDistance(a, b []float64) float64` | L2SquaredDistance returns the squared Euclidean distance between two equal-length float descriptors. |
| `func Mat3Det(m [3][3]float64) float64` | Mat3Det returns the determinant of a 3×3 matrix. |
| `func Mat3Identity() [3][3]float64` | Mat3Identity returns the 3×3 identity matrix. |
| `func Mat3Inverse(m [3][3]float64) ([3][3]float64, bool)` | Mat3Inverse returns the inverse of a 3×3 matrix and reports whether it is invertible. |
| `func Mat3Mul(a, b [3][3]float64) [3][3]float64` | Mat3Mul returns the matrix product a·b of two 3×3 matrices. |
| `func Mat3Scale(a [3][3]float64, s float64) [3][3]float64` | Mat3Scale returns the 3×3 matrix a with every entry multiplied by s. |
| `func Mat3Transpose(a [3][3]float64) [3][3]float64` | Mat3Transpose returns the transpose aᵀ of a 3×3 matrix. |
| `func Mat3VecMul(a [3][3]float64, v [3]float64) [3]float64` | Mat3VecMul returns the matrix–vector product a·v, where v is a 3-vector. |
| `func MatrixToRodrigues(R [3][3]float64) [3]float64` | MatrixToRodrigues converts a 3×3 rotation matrix into its axis-angle rotation vector, the inverse of `RodriguesToMatrix`. |
| `func MeanReprojectionError(objPts []core.Point3d, imgPts []core.Point2d, R [3][3]float64, t [3]float64, K [3][3]float64) float64` | MeanReprojectionError returns the mean of `ReprojectionErrors`, a single-number summary of pose quality. |
| `func MinMaxDistance(matches []DMatch) (min, max float64)` | MinMaxDistance returns the smallest and largest distance among the matches, or (0, 0) when matches is empty. |
| `func NormalizeL2(d []float64) []float64` | NormalizeL2 returns a copy of the descriptor scaled to unit Euclidean length. |
| `func NormalizePoints2D(pts []core.Point2d) ([]core.Point2d, [3][3]float64)` | NormalizePoints2D applies Hartley's isotropic normalization: it returns points translated so their centroid is the origin and scaled so their mean… |
| `func PerspectiveTransform(H [3][3]float64, pts []core.Point2d) []core.Point2d` | PerspectiveTransform maps every point in pts through the homography H, returning the transformed points in the same order. |
| `func PointLineDistance(line [3]float64, p core.Point2d) float64` | PointLineDistance returns the perpendicular distance from point p to the line (a, b, c) representing a·x + b·y + c = 0. |
| `func ProjectPoints(objPts []core.Point3d, R [3][3]float64, t [3]float64, K [3][3]float64) []core.Point2d` | ProjectPoints projects world points into the image under pose (R, t) and intrinsics K, returning one image point per world point. |
| `func ProjectionMatrix(K, R [3][3]float64, t [3]float64) [3][4]float64` | ProjectionMatrix assembles the 3×4 camera projection matrix P = K·[R \| t] from the intrinsic matrix K, rotation R and translation t. |
| `func RecoverPose(E [3][3]float64, pts1, pts2 []core.Point2d, K [3][3]float64) (R [3][3]float64, t [3]float64, good int)` | RecoverPose recovers the relative camera rotation R and unit translation t between two calibrated views from the essential matrix E and the matched… |
| `func ReprojectionErrors(objPts []core.Point3d, imgPts []core.Point2d, R [3][3]float64, t [3]float64, K [3][3]float64) []float64` | ReprojectionErrors returns the per-point reprojection error, in pixels, between the observed image points and the projections of the world points… |
| `func RodriguesToMatrix(rvec [3]float64) [3][3]float64` | RodriguesToMatrix converts an axis-angle rotation vector (whose direction is the rotation axis and whose magnitude is the rotation angle in radians)… |
| `func SampsonDistance(F [3][3]float64, p1, p2 core.Point2d) float64` | SampsonDistance returns the first-order geometric (Sampson) approximation of the reprojection error for a correspondence under the fundamental matrix… |
| `func SolvePnPDLT(objPts []core.Point3d, imgPts []core.Point2d, K [3][3]float64) (R [3][3]float64, t [3]float64, ok bool)` | SolvePnPDLT estimates the camera pose (rotation R and translation t) that maps the 3-D world points objPts to the 2-D image points imgPts under the… |
| `func SymmetricTransferError(H [3][3]float64, src, dst core.Point2d) float64` | SymmetricTransferError returns the symmetric transfer error for a correspondence: the sum of the forward reprojection error ‖H·src − dst‖ and… |
| `func TriangulatePoint(P1, P2 [3][4]float64, p1, p2 core.Point2d) core.Point3d` | TriangulatePoint reconstructs the 3-D world point that projects to p1 in the first camera (matrix P1) and to p2 in the second (matrix P2) using the… |
| `func TriangulatePoints(P1, P2 [3][4]float64, pts1, pts2 []core.Point2d) []core.Point3d` | TriangulatePoints triangulates a batch of correspondences with `TriangulatePoint`, returning one reconstructed world point per pair. |

### Types

| Type | What it is |
| --- | --- |
| `BFMatcher` | BFMatcher is a brute-force matcher for floating-point descriptors: it compares every query descriptor against every train descriptor under the… |
| `DMatch` | DMatch records a correspondence between a query descriptor and a train descriptor, mirroring OpenCV's cv::DMatch and the type of the same name in the… |
| `FLANNMatcher` | FLANNMatcher matches float descriptors using a `KDTree` built over the train set, giving the same results as a brute-force L2 matcher but with… |
| `HammingMatcher` | HammingMatcher is a brute-force matcher for bit-packed binary descriptors, such as those produced by ORB and BRIEF, compared with the Hamming… |
| `KDTree` | KDTree is a static k-d tree over float vectors (descriptors or points) that answers nearest-neighbour, k-nearest and radius queries under the squared… |
| `Line2D` | Line2D represents the 2-D line a·x + b·y + c = 0 with (a, b) a unit normal, so that a·px + b·py + c is the signed perpendicular distance from a… |
| `NormType` | NormType selects the distance used to compare floating-point descriptors. |
| `Pose` | Pose bundles a camera rotation and translation, the `SolvePnPRansac` model type. |
| `RANSACResult` | RANSACResult holds the outcome of a generic robust fit: the recovered model, the boolean inlier mask over the input observations and the inlier count. |

<details>
<summary><code>BFMatcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewBFMatcher(norm NormType) *BFMatcher` | NewBFMatcher returns a brute-force float matcher using the given norm with cross-check disabled. |
| `func (m *BFMatcher) KnnMatch(query, train [][]float64, k int) [][]DMatch` | KnnMatch returns, for each query descriptor, its k nearest train descriptors as a slice of `DMatch` sorted by ascending distance (ties broken by… |
| `func (m *BFMatcher) Match(query, train [][]float64) []DMatch` | Match returns the single best train match for each query descriptor. |
| `func (m *BFMatcher) RadiusMatch(query, train [][]float64, maxDistance float64) [][]DMatch` | RadiusMatch returns, for each query descriptor, every train descriptor within maxDistance, sorted by ascending distance. |

</details>

<details>
<summary><code>DMatch</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CrossCheck(forward, backward []DMatch) []DMatch` | CrossCheck keeps only the matches in forward (query→train) that are confirmed by backward (train→query): a pair (q, t) survives when backward… |
| `func FilterMatchesByDistance(matches []DMatch, maxDistance float64) []DMatch` | FilterMatchesByDistance returns the subset of matches whose distance does not exceed maxDistance, preserving input order. |
| `func RatioTest(knn [][]DMatch, ratio float64) []DMatch` | RatioTest applies Lowe's ratio test to the output of a k-nearest-neighbour match with k ≥ 2: it keeps a query's best match only when the best… |
| `func SortMatchesByDistance(matches []DMatch) []DMatch` | SortMatchesByDistance returns a copy of matches sorted by ascending distance, breaking ties by lower train index. |

</details>

<details>
<summary><code>FLANNMatcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewFLANNMatcher(train [][]float64) *FLANNMatcher` | NewFLANNMatcher builds a kd-tree index over the train descriptors and returns a matcher that queries it. |
| `func (f *FLANNMatcher) KnnMatch(query [][]float64, k int) [][]DMatch` | KnnMatch returns, for each query descriptor, its k nearest train descriptors sorted by ascending Euclidean distance, in query order. |
| `func (f *FLANNMatcher) Match(query [][]float64) []DMatch` | Match returns the single nearest train descriptor for each query descriptor, one `DMatch` per query in query order. |

</details>

<details>
<summary><code>HammingMatcher</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func NewHammingMatcher() *HammingMatcher` | NewHammingMatcher returns a brute-force Hamming matcher with cross-check disabled. |
| `func (m *HammingMatcher) KnnMatch(query, train [][]byte, k int) [][]DMatch` | KnnMatch returns, for each binary query descriptor, its k nearest train descriptors sorted by ascending Hamming distance. |
| `func (m *HammingMatcher) Match(query, train [][]byte) []DMatch` | Match returns the best train match for each binary query descriptor, mirroring `BFMatcher.Match` but with the Hamming distance. |

</details>

<details>
<summary><code>KDTree</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func BuildKDTree(points [][]float64) *KDTree` | BuildKDTree constructs a balanced k-d tree over the given points, which must all share the same non-zero dimension. |
| `func (t *KDTree) Dim() int` | Dim returns the dimension of the stored points, or 0 for an empty tree. |
| `func (t *KDTree) KNearest(query []float64, k int) (indices []int, distances []float64)` | KNearest returns the indices of the k points closest to query together with their squared distances, both ordered nearest-first. |
| `func (t *KDTree) Nearest(query []float64) (int, float64)` | Nearest returns the index of the point closest to query and the squared Euclidean distance to it. |
| `func (t *KDTree) RadiusSearch(query []float64, radius float64) (indices []int, distances []float64)` | RadiusSearch returns the indices of every point within radius of query and their squared distances, ordered nearest-first. |
| `func (t *KDTree) Size() int` | Size returns the number of points stored in the tree. |

</details>

<details>
<summary><code>Line2D</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FitLine(points []core.Point2d) (Line2D, bool)` | FitLine returns the total-least-squares line through the given points (at least two), minimising the sum of squared perpendicular distances. |
| `func (l Line2D) Distance(p core.Point2d) float64` | Distance returns the absolute perpendicular distance from point p to the line. |

</details>

<details>
<summary><code>RANSACResult</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func FindFundamentalMatRANSAC(pts1, pts2 []core.Point2d, threshold float64, iters int, seed int64) RANSACResult[[3][3]float64]` | FindFundamentalMatRANSAC robustly estimates the fundamental matrix from correspondences containing outliers. |
| `func FindHomographyRANSAC(src, dst []core.Point2d, threshold float64, iters int, seed int64) RANSACResult[[3][3]float64]` | FindHomographyRANSAC robustly estimates a homography from correspondences that may contain outliers. |
| `func FitLineRANSAC(points []core.Point2d, threshold float64, iters int, seed int64) RANSACResult[Line2D]` | FitLineRANSAC robustly fits a line to points, tolerating outliers. |
| `func LMedS[M any]( n, sampleSize, iters int, seed int64, fit func(sample []int) (M, bool), residual func(model M, i int) float64, ) RANSACResult[M]` | LMedS performs least-median-of-squares robust fitting over n observations. |
| `func RANSAC[M any]( n, sampleSize, iters, minInliers int, seed int64, fit func(sample []int) (M, bool), inliers func(model M) []bool, refit func(sample []int) (M, bool), ) RANSACResult[M]` | RANSAC performs Random Sample Consensus over n observations. |
| `func SolvePnPRansac(objPts []core.Point3d, imgPts []core.Point2d, K [3][3]float64, threshold float64, iters int, seed int64) RANSACResult[Pose]` | SolvePnPRansac robustly estimates the camera pose from correspondences that may contain outliers. |

</details>

### Constants

`DefaultRANSACSeed`

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/matching2).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
