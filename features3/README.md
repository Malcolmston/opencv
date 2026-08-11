# features3 — provides classic feature detection and description primitives on top

[![Go Reference](https://pkg.go.dev/badge/github.com/malcolmston/opencv/features3.svg)](https://pkg.go.dev/github.com/malcolmston/opencv/features3)

Package features3 provides classic feature detection and description
primitives on top of the standard-library-only OpenCV port
github.com/malcolmston/opencv (imported here as cv). It complements the
module's features2d package with a self-contained collection of corner and
interest-point operators, binary descriptors, non-linear region detectors and
scale-space blob detectors, all written against the Go standard library with
no third-party dependencies, no cgo and no GPU code.

Every operator consumes the module's central image type, cv.Mat. One-channel
(grayscale) and three-channel (RGB) images are accepted; three-channel input
is converted to grayscale with the BT.601 luma weights before processing.
Locations are reported with cv.Point (integer) and cv.Point2f (sub-pixel), and
dense response images use cv.FloatMat, matching the rest of the module.

## Install

```sh
go get github.com/malcolmston/opencv@v0.8.0
```

```go
import "github.com/malcolmston/opencv/features3"
```

## Usage

This is the package's own `ExampleCensusTransform3x3`, so it compiles and its output is
asserted on every `go test ./features3/`.

```go
img := cv.NewMat(3, 3, 1)
	img.SetTo(10)
	img.Set(1, 1, 0, 200)

	c := features3.CensusTransform3x3(img)
	fmt.Printf("%08b\n", c[1*3+1])
```

```
11111111
```

## Exported surface

### Functions

| Function | What it does |
| --- | --- |
| `func AGASTScore(img *cv.Mat, x, y, threshold int) int` | AGASTScore returns the adaptive AGAST cornerness score at (x, y): the largest intensity threshold for which the pixel still satisfies the OAST-16… |
| `func BitCount(x uint64) int` | BitCount returns the number of set bits (population count) of x. |
| `func CensusTransform3x3(img *cv.Mat) []uint8` | CensusTransform3x3 computes the 3×3 census (local binary) transform of an image. |
| `func CensusTransform5x5(img *cv.Mat) []uint32` | CensusTransform5x5 computes the 5×5 census transform of an image, packing the twenty-four neighbour comparisons of each pixel into a uint32 (bit i… |
| `func ComputeBRIEF(img *cv.Mat, kps []KeyPoint, pattern *BRIEFPattern) [][]byte` | ComputeBRIEF computes an (unsteered) BRIEF binary descriptor for each keypoint using pattern. |
| `func ComputeORB(img *cv.Mat, kps []KeyPoint, pattern *BRIEFPattern) [][]byte` | ComputeORB computes an ORB-style steered BRIEF binary descriptor for each keypoint. |
| `func ConvertKeyPointsToPoints(kps []KeyPoint) []cv.Point` | ConvertKeyPointsToPoints returns the rounded integer locations of the given keypoints. |
| `func CornerEigenVals(img *cv.Mat, blockSize int) (lambda1, lambda2 *cv.FloatMat)` | CornerEigenVals computes both eigenvalues of the windowed structure tensor at every pixel, returning the larger eigenvalue field lambda1 and the… |
| `func CornerMinEigenVal(img *cv.Mat, blockSize int) *cv.FloatMat` | CornerMinEigenVal computes the Shi–Tomasi corner measure: the smaller eigenvalue of the windowed structure tensor at each pixel, returned as a… |
| `func CornerSubPix(img *cv.Mat, corners []cv.Point2f, winSize, maxIter int, epsilon float64) []cv.Point2f` | CornerSubPix refines integer corner locations to sub-pixel accuracy using the classic iterative method: the refined point is the location where… |
| `func DifferenceOfGaussians(img *cv.Mat, sigma1, sigma2 float64) *cv.FloatMat` | DifferenceOfGaussians returns the Difference-of-Gaussians of an image: the image blurred at sigma1 minus the image blurred at sigma2, as a… |
| `func FASTScore(img *cv.Mat, x, y int) float64` | FASTScore returns the FAST cornerness score at (x, y): the sum of absolute differences between the centre and the sixteen circle pixels. |
| `func GaussianScaleSpace(img *cv.Mat, sigmas []float64) []*cv.FloatMat` | GaussianScaleSpace returns the image blurred at each of the given sigmas as a slice of cv.FloatMats, the shared building block of the scale-space… |
| `func HammingDistance(a, b []byte) int` | HammingDistance returns the number of differing bits between two equal-length byte slices (their Hamming distance). |
| `func HammingDistanceUint64(a, b uint64) int` | HammingDistanceUint64 returns the Hamming distance between two 64-bit words. |
| `func HarrisResponse(img *cv.Mat, blockSize int, k float64) *cv.FloatMat` | HarrisResponse computes the Harris corner response of an image. |
| `func HessianDeterminant(img *cv.Mat, sigma float64) *cv.FloatMat` | HessianDeterminant returns the scale-normalised determinant of the Hessian sigma^4*(Lxx*Lyy - Lxy^2) of an image blurred at the given sigma, as a… |
| `func IntensityCentroidOrientation(img *cv.Mat, x, y, radius int) float64` | IntensityCentroidOrientation returns the orientation, in degrees in [0, 360), of the patch of the given radius around (x, y), computed from the… |
| `func IsAGASTCorner(img *cv.Mat, x, y, threshold int) bool` | IsAGASTCorner reports whether pixel (x, y) is a corner under the AGAST (Adaptive and Generic Accelerated Segment Test) OAST-16 criterion: at least… |
| `func IsFASTCorner(img *cv.Mat, x, y, threshold int) bool` | IsFASTCorner reports whether pixel (x, y) passes the FAST-9 segment test with the given intensity threshold: at least nine contiguous pixels on the… |
| `func LaplacianResponse(img *cv.Mat, sigma float64) *cv.FloatMat` | LaplacianResponse returns the scale-normalised Laplacian-of-Gaussian response sigma^2*(Lxx+Lyy) of an image blurred at the given sigma, as a… |
| `func LocalMaxima(resp *cv.FloatMat, halfWindow int, threshold float64) []cv.Point` | LocalMaxima returns the integer coordinates of the strict local maxima of a response image within a (2*halfWindow+1) window whose value is at least… |
| `func PreCornerDetect(img *cv.Mat) *cv.FloatMat` | PreCornerDetect computes OpenCV's preCornerDetect feature map, the quantity Ix^2*Iyy + Iy^2*Ixx - 2*Ix*Iy*Ixy formed from first and second image… |
| `func SUSANResponse(img *cv.Mat, radius int, t, geometricThreshold float64) *cv.FloatMat` | SUSANResponse computes the SUSAN (Smallest Univalue Segment Assimilating Nucleus) corner response of an image. |
| `func SortKeyPointsByResponse(kps []KeyPoint)` | SortKeyPointsByResponse sorts keypoints in place in descending order of Response, breaking ties by position for a deterministic ordering. |

### Types

| Type | What it is |
| --- | --- |
| `BRIEFPattern` | BRIEFPattern is a fixed set of intensity-comparison point pairs sampled inside a square patch, used to compute BRIEF and steered-BRIEF (ORB) binary… |
| `Blob` | Blob is a scale-space interest region: a location and the Gaussian scale (sigma) at which it responded most strongly, as returned by `LoGBlobs`,… |
| `CensusField` | CensusField is a dense census transform with a configurable window, storing one bit-packed uint64 code per pixel. |
| `KeyPoint` | KeyPoint is a salient image point produced by a detector in this package. |
| `MSERRegion` | MSERRegion is a Maximally Stable Extremal Region: a connected set of pixels that keeps a nearly constant area across a range of intensity thresholds,… |
| `Match` | Match records a correspondence between a query descriptor and a train descriptor, mirroring OpenCV's cv::DMatch. |

<details>
<summary><code>BRIEFPattern</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DefaultBRIEFPattern() *BRIEFPattern` | DefaultBRIEFPattern returns the standard 256-bit BRIEF pattern in a 31×31 patch generated with a fixed seed, matching the descriptor length used by… |
| `func GenerateBRIEFPattern(patchSize, numPairs int, seed int64) *BRIEFPattern` | GenerateBRIEFPattern builds a BRIEF sampling pattern of numPairs comparisons inside a patchSize×patchSize patch. |
| `func (p *BRIEFPattern) NumBits() int` | NumBits returns the number of comparison bits (descriptor length in bits). |
| `func (p *BRIEFPattern) NumBytes() int` | NumBytes returns the number of bytes each packed descriptor occupies. |

</details>

<details>
<summary><code>Blob</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func DoGBlobs(img *cv.Mat, minSigma, maxSigma float64, numScales int, threshold float64) []Blob` | DoGBlobs detects blobs with the Difference-of-Gaussians approximation to the Laplacian. |
| `func DoHBlobs(img *cv.Mat, minSigma, maxSigma float64, numScales int, threshold float64) []Blob` | DoHBlobs detects blobs with the scale-normalised determinant of the Hessian. |
| `func LoGBlobs(img *cv.Mat, minSigma, maxSigma float64, numScales int, threshold float64) []Blob` | LoGBlobs detects blobs with the scale-normalised Laplacian-of-Gaussian. |
| `func (b Blob) Area() float64` | Area returns the area of the blob disc, pi*Radius^2. |
| `func (b Blob) Radius() float64` | Radius returns the characteristic blob radius sigma*sqrt(2), the radius of the bright/dark region a Laplacian-of-Gaussian of this scale responds to. |
| `func (b Blob) ToKeyPoint() KeyPoint` | ToKeyPoint converts the blob to a KeyPoint whose Size is the blob diameter (2*Radius) and whose Response carries the operator response. |

</details>

<details>
<summary><code>CensusField</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func CensusTransform(img *cv.Mat, halfWindow int) *CensusField` | CensusTransform computes the census transform of an image over a (2*halfWindow+1) square window, comparing every neighbour with the centre and… |
| `func ModifiedCensusTransform(img *cv.Mat, halfWindow int) *CensusField` | ModifiedCensusTransform computes the modified census transform, which compares each neighbour with the mean intensity of the window rather than the… |
| `func (c *CensusField) At(x, y int) uint64` | At returns the packed census code at column x, row y. |
| `func (c *CensusField) Hamming(x1, y1, x2, y2 int) int` | Hamming returns the Hamming distance between the census codes at (x1, y1) and (x2, y2), a matching cost in the range [0, Bits]. |

</details>

<details>
<summary><code>KeyPoint</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func AGASTKeyPoints(img *cv.Mat, threshold int, nonmaxSuppression bool) []KeyPoint` | AGASTKeyPoints detects corners with the AGAST OAST-16 test. |
| `func ConvertPointsToKeyPoints(pts []cv.Point, size float64) []KeyPoint` | ConvertPointsToKeyPoints wraps integer points as keypoints with the given size, zero response and unset orientation. |
| `func FASTKeyPoints(img *cv.Mat, threshold int, nonmaxSuppression bool) []KeyPoint` | FASTKeyPoints detects corners with the FAST-9 accelerated segment test. |
| `func FilterByBorder(kps []KeyPoint, rows, cols, border int) []KeyPoint` | FilterByBorder returns the keypoints lying at least border pixels away from every edge of a rows×cols image, preserving their order. |
| `func FilterByResponse(kps []KeyPoint, minResponse float64) []KeyPoint` | FilterByResponse returns the keypoints whose Response is at least minResponse, preserving their order. |
| `func GoodFeaturesToTrack(img *cv.Mat, maxCorners int, qualityLevel, minDistance float64, blockSize int) []KeyPoint` | GoodFeaturesToTrack finds strong corners with the Shi–Tomasi measure, the Go analogue of OpenCV's cv::goodFeaturesToTrack. |
| `func HarrisCorners(img *cv.Mat, blockSize int, k, qualityLevel float64, maxCorners int) []KeyPoint` | HarrisCorners detects corners with the Harris measure. |
| `func KeypointNMS(kps []KeyPoint, minDistance float64) []KeyPoint` | KeypointNMS applies greedy radial non-maximum suppression to a keypoint list. |
| `func NewKeyPoint(x, y, response float64) KeyPoint` | NewKeyPoint returns a KeyPoint at the given column/row with the supplied response, a size of 7, an unset orientation (-1) and octave 0. |
| `func NonMaxSuppressionFloat(resp *cv.FloatMat, halfWindow int, threshold float64) []KeyPoint` | NonMaxSuppressionFloat scans a response image and returns the local maxima as keypoints. |
| `func RetainBest(kps []KeyPoint, n int) []KeyPoint` | RetainBest returns the n strongest keypoints by Response. |
| `func SUSANCorners(img *cv.Mat, radius int, t, geometricThreshold, qualityLevel float64, maxCorners int) []KeyPoint` | SUSANCorners detects corners with the SUSAN operator. |
| `func ShiTomasiCorners(img *cv.Mat, blockSize int, qualityLevel float64, maxCorners int) []KeyPoint` | ShiTomasiCorners detects corners with the Shi–Tomasi minimum-eigenvalue measure. |
| `func (k KeyPoint) DistanceTo(other KeyPoint) float64` | DistanceTo returns the Euclidean distance in pixels between two keypoints. |
| `func (k KeyPoint) Point() cv.Point` | Point returns the keypoint location rounded to the nearest integer cv.Point. |

</details>

<details>
<summary><code>MSERRegion</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MSERRegions(img *cv.Mat, delta, minArea, maxArea int, maxVariation float64) []MSERRegion` | MSERRegions extracts Maximally Stable Extremal Regions from an image, the Go analogue of OpenCV's cv::MSER. |
| `func (r MSERRegion) Area() int` | Area returns the number of pixels in the region. |
| `func (r MSERRegion) BoundingRect() image.Rectangle` | BoundingRect returns the axis-aligned bounding box of the region as an image.Rectangle (Max exclusive). |
| `func (r MSERRegion) Centroid() cv.Point2f` | Centroid returns the mean location of the region's pixels. |

</details>

<details>
<summary><code>Match</code> — constructors and methods</summary>

| Signature | What it does |
| --- | --- |
| `func MatchBinaryDescriptors(query, train [][]byte, maxDistance int) []Match` | MatchBinaryDescriptors matches each query descriptor to its nearest train descriptor by Hamming distance (brute force). |

</details>

Full signatures, doc comments and every runnable example are on
[pkg.go.dev](https://pkg.go.dev/github.com/malcolmston/opencv/features3).

## License

MIT, as part of [`github.com/malcolmston/opencv`](..).
An independent re-implementation, not affiliated with or endorsed by the
original project.
